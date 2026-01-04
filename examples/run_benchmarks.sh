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
        run_benchmark "${algo}" --distance euclidean --data correlated
        run_benchmark "${algo}" --distance euclidean --data lowrank
    done
}

run_quantised_benchmarks() {
    echo "=== Running quantised benchmarks ==="
    # IVF-BF16
    run_quantised bf16 --distance euclidean
    run_quantised bf16 --distance cosine
    run_quantised bf16 --distance euclidean --data correlated
    run_quantised bf16 --distance euclidean --data lowrank

    # IVF-SQ8
    run_quantised sq8 --distance euclidean
    # more dimensions
    run_quantised sq8 --distance euclidean --dim 96
    run_quantised sq8 --distance euclidean --dim 96 --data correlated
    run_quantised sq8 --distance euclidean --dim 96 --data lowrank
    # even more dimensions
    run_quantised sq8 --distance euclidean --dim 128
    run_quantised sq8 --distance euclidean --dim 128 --data correlated
    run_quantised sq8 --distance euclidean --dim 128 --data lowrank
    
    # IVF-PQ
    run_quantised pq --distance euclidean --dim 128
    run_quantised pq --distance euclidean --dim 128 --data correlated
    run_quantised pq --distance euclidean --dim 128 --data lowrank

    run_quantised pq --distance euclidean --dim 192
    run_quantised pq --distance euclidean --dim 192 --data correlated
    run_quantised pq --distance euclidean --dim 192 --data lowrank
    
    # IVF-OPQ
    run_quantised opq --distance euclidean --dim 128 
    run_quantised opq --distance euclidean --dim 128 --data correlated
    run_quantised opq --distance euclidean --dim 128 --data lowrank

    run_quantised opq --distance euclidean --dim 192 
    run_quantised opq --distance euclidean --dim 192 --data correlated
    run_quantised opq --distance euclidean --dim 192 --data lowrank
}

run_gpu_benchmarks() {
    echo "=== Running GPU benchmarks ==="
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean
    cargo run --example gridsearch_gpu --release --features gpu -- --distance cosine
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data correlated
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank
    
    echo "Running GPU benchmarks (larger data set)..."
    cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-cells 250000 --dim 64
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-cells 250000 --dim 64
    cargo run --example gridsearch_ivf --release --features gpu -- --distance euclidean --n-cells 500000 --dim 64
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --n-cells 500000 --dim 64
    
    echo "Running GPU benchmarks (more dimensions)..."
    cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --dim 128 --data correlated
}

run_binary_benchmarks() {
    echo "=== Running binary benchmarks ===" 

    echo "Running simple binarisations"
    cargo run --example gridsearch_binary --release --features binary -- --distance euclidean
    cargo run --example gridsearch_binary --release --features binary -- --distance cosine
    cargo run --example gridsearch_binary --release --features binary -- --distance euclidean --data correlated
    cargo run --example gridsearch_binary --release --features binary -- --distance euclidean --data lowrank

    echo "Running RaBitQ binarisations"
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean
    cargo run --example gridsearch_rabitq --release --features binary -- --distance cosine
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --data correlated
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --data lowrank

    # more dimensions
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim 128
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim 128 --data correlated
    cargo run --example gridsearch_rabitq --release --features binary -- --distance euclidean --dim 128 --data lowrank
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--standard] [--quantised] [--gpu] [--binary] [--all]"
    exit 1
fi

RUN_STANDARD=false
RUN_QUANTISED=false
RUN_GPU=false
RUN_BINARY=false

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
        --binary)
            RUN_BINARY=true
            ;;
        --all)
            RUN_STANDARD=true
            RUN_QUANTISED=true
            RUN_GPU=true
            RUN_BINARY=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--standard] [--quantised] [--gpu] [--binary] [--all]"
            exit 1
            ;;
    esac
done

$RUN_STANDARD && run_standard
$RUN_QUANTISED && run_quantised_benchmarks
$RUN_GPU && run_gpu_benchmarks
$RUN_BINARY && run_binary_benchmarks

echo "All benchmarks complete."