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
    cargo run --example "gridsearch_${variant}" --release --features quantised "$@"
}

run_common_patterns() {
    local run_fn=$1
    local name=$2
    shift 2

    echo "Running ${name} benchmarks..."
    $run_fn "$@" -- --distance euclidean
    $run_fn "$@" -- --distance cosine
    $run_fn "$@" -- --distance euclidean --data correlated
    $run_fn "$@" -- --distance euclidean --data lowrank
    $run_fn "$@" -- --distance euclidean --data lowrank --dim 128 --intrinsic-dim 32
}

run_standard() {
    echo "=== Running standard benchmarks ==="
    for algo in annoy balltree hnsw ivf kd_forest lsh nndescent vamana; do
        run_common_patterns run_benchmark "${algo}" "${algo}"
    done
}

run_quantised_benchmarks() {
    echo "=== Running quantised benchmarks ==="

    for variant in bf16 sq8; do
        run_common_patterns run_quantised "${variant}" "${variant}"
    done

    for variant in pq opq; do
        for dim in 128 256 512; do
            echo "Running ${variant} benchmarks (dim=${dim})..."
            run_quantised ${variant} -- --distance euclidean --dim ${dim} --data correlated --n-samples 50000
            run_quantised ${variant} -- --distance euclidean --dim ${dim} --data lowrank --n-samples 50000 --intrinsic-dim $((dim / 4))
            run_quantised ${variant} -- --distance euclidean --dim ${dim} --data quantisation --n-samples 50000 --n-cluster 50 --intrinsic-dim {UPDATE}
        done
    done
}

run_gpu_benchmarks() {
    echo "=== Running GPU benchmarks ==="
    run_common_patterns "cargo run --example gridsearch_gpu --release --features gpu" "GPU (IVF)"
    run_common_patterns "cargo run --example gridsearch_cagra --release --features gpu" "GPU (Cagra)"

    echo "Running GPU benchmarks (larger data sets)..."
    for n_samples in 250000 500000; do
        for n_dim in 64 128; do
            cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-samples ${n_samples} --dim ${n_dim} --data lowrank --intrinsic-dim $((n_dim / 4))
            cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-samples ${n_samples} --dim ${n_dim} --data lowrank --intrinsic-dim $((n_dim / 4))
            cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --n-samples ${n_samples} --dim ${n_dim} --data lowrank --intrinsic-dim $((n_dim / 4))
        done
    done

    echo "Running kNN specific benchmarks ..."
    for n_samples in 250000 500000 1000000; do
        for n_dim in 32 64; do
            cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples ${n_samples} --dim ${n_dim} --intrinsic-dim $((n_dim / 4))
        done
    done
}

run_binary_benchmarks() {
    echo "=== Running binary benchmarks ==="

    for variant in binary rabitq; do
        for n_dim in 256 512 1024; do
            cargo run --example gridsearch_${variant} --release --features binary -- --dim ${n_dim} --data correlated --n-samples 50000
            cargo run --example gridsearch_${variant} --release --features binary -- --dim ${n_dim} --data lowrank --n-samples 50000 --intrinsic-dim $((n_dim / 4))
            cargo run --example gridsearch_${variant} --release --features binary -- --dim ${n_dim} --data quantisation --n-samples 50000 --n-cluster 50
        done
    done
}

[ $# -eq 0 ] && { echo "Usage: $0 [--standard] [--quantised] [--gpu] [--binary] [--all]"; exit 1; }

RUN_STANDARD=false
RUN_QUANTISED=false
RUN_GPU=false
RUN_BINARY=false

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
