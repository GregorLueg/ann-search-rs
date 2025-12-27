#!/usr/bin/env bash
set -euo pipefail

# run_benchmark() {
#     local name=$1
#     shift
#     echo "Running ${name} benchmarks..."
#     cargo run --example "gridsearch_${name,,}" --release -- "$@"
# }

# # Basic algorithms with cosine and euclidean
# for algo in annoy hnsw ivf lsh nndescent; do
#     run_benchmark "${algo^^}" --distance euclidean
#     run_benchmark "${algo^^}" --distance cosine
# done

# # Algorithms with quantisations

# # IVF-SQ8
# echo "Running IVF-SQ8 benchmarks..."
# cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean
# cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean --dim 96
# cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean --dim 128 --data correlated

# # IVF-PQ
# echo "Running IVF-PQ benchmarks..."
# cargo run --example gridsearch_ivf_pq --release -- --distance euclidean --dim 128 --data correlated
# cargo run --example gridsearch_ivf_pq --release -- --distance euclidean --dim 192 --data correlated

# IVF-OPQ
echo "Running IVF-OPQ benchmarks..."
# cargo run --example gridsearch_ivf_opq --release -- --distance euclidean --dim 128 --data correlated
# cargo run --example gridsearch_ivf_opq --release -- --distance euclidean --dim 192 --data correlated

# Algorithms with GPU acceleration
echo "Running GPU benchmarks..."
cargo run --example gridsearch_gpu --release -- --distance euclidean
cargo run --example gridsearch_gpu --release -- --distance cosine

# Comparison with larger datasets
echo "Running GPU benchmarks (larger data set)..."
cargo run --example gridsearch_ivf --release -- --distance euclidean --n-cells 250000 --dim 64
cargo run --example gridsearch_gpu --release -- --distance euclidean --n-cells 250000 --dim 64

# More dimensions
echo "Running GPU benchmarks (more dimensions)..."
cargo run --example gridsearch_gpu --release -- --distance euclidean --dim 128 --data correlated

echo "All benchmarks complete."