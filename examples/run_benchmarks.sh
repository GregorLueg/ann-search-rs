#!/usr/bin/env bash
set -euo pipefail

run_benchmark() {
    local name=$1
    shift
    echo "Running ${name} benchmarks..."
    cargo run --example "gridsearch_${name}" --release -- "$@"
}

run_quantised() {
    local variant=$1
    shift
    echo "Running IVF-${variant} benchmarks..."
    cargo run --example "gridsearch_ivf_${variant}" --release --features quantised -- "$@"
}

run_standard() {
    echo "=== Running standard benchmarks ==="
    for algo in annoy hnsw ivf lsh nndescent; do
        run_benchmark "${algo}" --distance euclidean
        run_benchmark "${algo}" --distance cosine
    done
}

run_quantised_benchmarks() {
    echo "=== Running quantised benchmarks ==="
    # IVF-BF16
    run_quantised bf16 --distance euclidean
    run_quantised bf16 --distance cosine

    # IVF-SQ8
    run_quantised sq8 --distance euclidean
    run_quantised sq8 --distance euclidean --dim 96
    run_quantised sq8 --distance euclidean --dim 128 --data correlated
    
    # IVF-PQ
    run_quantised pq --distance euclidean --dim 128 --data correlated
    run_quantised pq --distance euclidean --dim 192 --data correlated
    
    # IVF-OPQ
    run_quantised opq --distance euclidean --dim 128 --data correlated
    run_quantised opq --distance euclidean --dim 192 --data correlated
}

run_gpu_benchmarks() {
    echo "=== Running GPU benchmarks ==="
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean
    cargo run --example gridsearch_gpu --release --features gpu -- --distance cosine
    
    echo "Running GPU benchmarks (larger data set)..."
    cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-cells 250000 --dim 64
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-cells 250000 --dim 64
    cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-cells 500000 --dim 64
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-cells 500000 --dim 64
    
    echo "Running GPU benchmarks (more dimensions)..."
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --dim 128 --data correlated
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--standard] [--quantised] [--gpu] [--all]"
    exit 1
fi

RUN_STANDARD=false
RUN_QUANTISED=false
RUN_GPU=false

for arg in "$@"; do
    case $arg in
        --standard)
            RUN_STANDARD=true
            ;;
        --quantised)
            RUN_QUANTISED=true
            ;;
        --gpu)
            RUN_GPU=true
            ;;
        --all)
            RUN_STANDARD=true
            RUN_QUANTISED=true
            RUN_GPU=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--standard] [--quantised] [--gpu] [--all]"
            exit 1
            ;;
    esac
done

$RUN_STANDARD && run_standard
$RUN_QUANTISED && run_quantised_benchmarks
$RUN_GPU && run_gpu_benchmarks

echo "All benchmarks complete."