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
    cargo run --example "gridsearch_ivf_${variant}" --release -- "$@"
}

# Basic algorithms with cosine and euclidean
for algo in annoy hnsw ivf lsh nndescent gpu; do
    run_benchmark "${algo}" --distance euclidean
    run_benchmark "${algo}" --distance cosine
done

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

# GPU benchmarks - larger datasets
echo "Running GPU benchmarks (larger data set)..."
cargo run --example gridsearch_ivf --release -- --distance euclidean --n-cells 250000 --dim 64
cargo run --example gridsearch_gpu --release -- --distance euclidean --n-cells 250000 --dim 64

# GPU benchmarks - more dimensions
echo "Running GPU benchmarks (more dimensions)..."
cargo run --example gridsearch_gpu --release -- --distance euclidean --dim 128 --data correlated

echo "All benchmarks complete."