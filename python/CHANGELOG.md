# News

Changes to the `ann-search` Python package. The Rust crate it wraps,
`ann-search-rs`, has its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.5.1

Requires `ann-search-rs` 0.11.1

- Takes in the minor updates from `ann-search-rs` without any impact on the
  Python side.

## 0.5.0

Requires `ann-search-rs` 0.11.0

- Experimental MLX estimators for Apple Silicon: `ExhaustiveMlxIndex`,
  `IvfMlxIndex` and `CagraMlxIndex`, plus `mlx_available()`. Source builds only
  (`brew install mlx-c`, then `maturin develop --release --features mlx`); the
  published wheel does not carry them. Same constraints as the GPU estimators,
  and a fitted index only works from the thread that built it.
- `NNDescentIndex` seeds by joining every forest leaf and defaults to scanpy's
  tree count, `min(5 + round(sqrt(n) / 20), 64)`. On 2.86M Tahoe PCA cells at
  64 trees: build 50.3 s to 28.0 s, recall@15 0.976 to 0.990.
- The GPU and MLX estimators build with lower peak memory, and the MLX ones
  hand MLX's buffer cache back once fitted.
- `CagraGpuIndex` picks up the GPU NN-Descent fixes: no duplicate neighbour ids
  from the random initialisation, and mutual-neighbour duplicates dropped before
  the local join.

## 0.4.1

Requires `ann-search-rs` 0.10.1

- Improved k-means on GPU and CPU.

## 0.4.0

Requires `ann-search-rs` 0.10.0

- Improved speed across all GPU-based indices
- Improved query speed + bug fix for HNSW RaBitQ at `ex_bit = 0`.
- Breaking change: older saved indices cannot be loaded with this update
  anymore.

## 0.3.2

Requires `ann-search-rs` 0.9.3

- Take the speeed improvements for various CPU-based indices (BallTree, Annoy,
  NNDescent) from the Rust package.

## 0.3.1

Requires `ann-search-rs` 0.9.1 (will be 0.9.2 if compiled from source).

- Improved k-means clustering and `Accelerate` framework enabled on MacOS,
  yielding even faster GEMM which impacts several of the indices.

## 0.3.0

Requires `ann-search-rs` 0.9.0.

- Takes in the RaBitQ rework: a fast Hadamard rotation and fast-scan distance
  estimation. RaBitQ indices saved with an earlier version cannot be loaded.
- New `QgIndex` and `HnswRaBitQIndex`, the two RaBitQ graph indices. The wheel
  now builds the crate's `binary` feature, which adds `statrs` (and with it
  `nalgebra`) to the dependency tree.
- `VamanaIndex` runs its first pass at the crate's narrow default beam instead
  of `l_build`, following the core change to `build_vamana_index`.

## 0.2.4

Requires `ann-search-rs` 0.8.5.

- Takes in the improved GPU speeds for the indices.

## 0.2.3

Requires `ann-search-rs` 0.8.4.

- Picks up the GPU query fix: `ExhaustiveGpuIndex`, `IvfGpuIndex` and
  `CagraGpuIndex` rejected any query whose `n_features` was not a multiple of
  four with a dimension-mismatch error.

## 0.2.2

Requires `ann-search-rs` 0.8.3.

- Take the x86_64 updates forward in terms of SIMD.

## 0.2.1

Requires `ann-search-rs` 0.8.2.

- Documentation updates where out-of-date things were claimed.
- Wiring in the faster RNN from the Rust parent package.

## 0.2.0

Requires `ann-search-rs` 0.8.1.

- The eleven quantised indices are now bound: `ExhaustiveBf16Index`,
  `IvfBf16Index`, `ExhaustiveSq8Index`, `IvfSq8Index`, `HnswSq8uIndex`,
  `ExhaustivePqIndex`, `IvfPqIndex`, `ExhaustiveOpqIndex`, `IvfOpqIndex`,
  `SoarPqIndex`, `SoarOpqIndex`. Same four-method surface as the rest, save /
  load / pickle included, float32 and float64 both supported. The binary
  indices are not bound yet.
- Docs corrected against the regenerated benchmark tables. HNSW is now the
  cheapest graph index to build rather than the most expensive, which reversed
  three separate claims in `choosing.md`.

## 0.1.0

Requires `ann-search-rs` 0.7.0. First release on PyPI.

- Python bindings under `python/`, built with PyO3 and maturin. scikit-learn
  shaped estimators over the CPU indices, plus the synthetic generators.
