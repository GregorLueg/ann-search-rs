#!/usr/bin/env bash
set -euo pipefail

run_benchmark() {
    local name=$1
    shift
    echo "Running ${name} benchmarks..."
    cargo run --example "gridsearch_${name}" --release "$@"
}

run_quantised() {
    local variant=$1
    shift
    cargo run --example "gridsearch_${variant}" --release --features quantised -- "$@"
}

run_common_patterns() {
    local run_fn=$1
    local name=$2
    shift 2
    
    echo "Running ${name} benchmarks..."
    $run_fn "$@" --distance euclidean
    $run_fn "$@" --distance cosine
    $run_fn "$@" --distance euclidean --data correlated
    $run_fn "$@" --distance euclidean --data lowrank
}

run_standard() {
    echo "=== Running standard benchmarks ==="
    for algo in annoy hnsw ivf lsh nndescent; do
        run_common_patterns run_benchmark "${algo}" "${algo}"
    done
}

run_quantised_benchmarks() {
    echo "=== Running quantised benchmarks ==="
    
    # IVF-BF16 and IVF-SQ8
    for variant in bf16 sq8; do
        run_common_patterns run_quantised "${variant}" "${variant}"
    done

    # Higher dimensions for SQ8
    for dim in 96 128; do
        echo "Running SQ8 benchmarks (dim=${dim})..."
        run_quantised sq8 --distance euclidean --dim ${dim}
        run_quantised sq8 --distance euclidean --dim ${dim} --data correlated
        run_quantised sq8 --distance euclidean --dim ${dim} --data lowrank
    done
    
    # IVF-PQ and IVF-OPQ
    for variant in ivf-pq ivf-opq; do
        for dim in 128 192; do
            echo "Running IVF-${variant} benchmarks (dim=${dim})..."
            run_quantised ${variant} --distance euclidean --dim ${dim}
            run_quantised ${variant} --distance euclidean --dim ${dim} --data correlated
            run_quantised ${variant} --distance euclidean --dim ${dim} --data lowrank
        done
    done
}

run_gpu_benchmarks() {
    echo "=== Running GPU benchmarks ==="
    run_common_patterns "cargo run --example gridsearch_gpu --release --features gpu --" "GPU"
    
    echo "Running GPU benchmarks (larger data sets)..."
    for n_cells in 250000 500000; do
        cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-cells ${n_cells} --dim 64
        cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-cells ${n_cells} --dim 64
    done
    
    echo "Running GPU benchmarks (more dimensions)..."
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --dim 128 --data correlated
}

run_binary_benchmarks() {
    echo "=== Running binary benchmarks ===" 
    
    for variant in binary rabitq; do
        run_common_patterns "cargo run --example gridsearch_${variant} --release --features binary --" "$(echo ${variant} | tr '[:lower:]' '[:upper:]')"
    done

    echo "Running RaBitQ benchmarks (more dimensions)..."
    for dim in 128; do
        cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim ${dim}
        cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim ${dim} --data correlated
        cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim ${dim} --data lowrank
    done
}

[ $# -eq 0 ] && { echo "Usage: $0 [--standard] [--quantised] [--gpu] [--binary] [--all]"; exit 1; }

RUN_STANDARD=false RUN_QUANTISED=false RUN_GPU=false RUN_BINARY=false

for arg in "$@"; do
    case $arg in
        --standard) RUN_STANDARD=true ;;
        --quantised) RUN_QUANTISED=true ;;
        --gpu) RUN_GPU=true ;;
        --binary) RUN_BINARY=true ;;
        --all) RUN_STANDARD=true RUN_QUANTISED=true RUN_GPU=true RUN_BINARY=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

$RUN_STANDARD && run_standard
$RUN_QUANTISED && run_quantised_benchmarks
$RUN_GPU && run_gpu_benchmarks
$RUN_BINARY && run_binary_benchmarks

echo "All benchmarks complete."