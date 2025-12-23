#!/usr/bin/env bash
set -euo pipefail

run_benchmark() {
    local name=$1
    shift
    echo "Running ${name} benchmarks..."
    cargo run --example "gridsearch_${name,,}" --release -- "$@"
}

# Basic algorithms with cosine and euclidean
for algo in annoy hnsw ivf lsh nndescent; do
    run_benchmark "${algo^^}" --distance cosine
    run_benchmark "${algo^^}" --distance euclidean
done

# Algorithms with quantisations

IVF-SQ8
echo "Running IVF-SQ8 benchmarks..."
cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean
cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean --dim 96
cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean --dim 128 --data correlated

# IVF-PQ
echo "Running IVF-PQ benchmarks..."
cargo run --example gridsearch_ivf_pq --release -- --distance euclidean --dim 128 --data correlated
cargo run --example gridsearch_ivf_pq --release -- --distance euclidean --dim 192 --data correlated

# IVF-OPQ
echo "Running IVF-OPQ benchmarks..."
cargo run --example gridsearch_ivf_opq --release -- --distance euclidean --dim 128 --data correlated
cargo run --example gridsearch_ivf_opq --release -- --distance euclidean --dim 192 --data correlated

echo "All benchmarks complete."