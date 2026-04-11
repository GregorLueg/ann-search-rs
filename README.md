[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ann-search-rs

Various approximate nearest neighbour/vector searches implemented in Rust
(with focus on computational biology methods).

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Gridsearches and performance](#running-the-grid-searches)
- [FEATURE: quantisation](#quantised-indices)
- [FEATURE: GPU acceleration](#gpu)
- [FEATURE: Binary indices](#binarised-indices)

## Description

Extracted function for approximate nearest neighbour searches specifically
with single cell in mind from [bixverse](https://github.com/GregorLueg/bixverse),
a R/Rust package designed for computational biology, that has a ton of
functionality for single cell. Within all of the single cell functions, kNN
generations are ubiqituos, thus, I want to expose the APIs to other packages.
Feel free to use these implementations where you might need approximate nearest
neighbour searches. This work is based on the great work from others who
figured out how to design these algorithms and is just an implementation into
Rust of many of these. Over time, I started getting interested into vector
searches and implement WAY more indices and new stuff into this than initially
anticipated. If you want to see what changed, please check this
[one out](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/news.md)

## Features

- **Multiple ANN algorithms**:
  - *Annoy (Approximate Nearest Neighbours Oh Yeah)*
  - *BallTree*
  - *Exhaustive flat index*
  - *HNSW (Hierarchical Navigable Small World)*
  - *IVF (Inverted File index)*
  - *Kd forest (based on Kd trees)*
  - *LSH (Locality Sensitive Hashing)*
  - *NNDescent (Nearest Neighbour Descent)*
  (heavily inspired by [PyNNDescent](https://github.com/lmcinnes/pynndescent)).
  - *Vanama (the graph powering DiskANN)*

- **Distance metrics**:
  - Euclidean
  - Cosine
  - More to come maybe... ?

- **High performance**: Optimised implementations with SIMD, heavy
  multi-threading were possible and optimised structures for memory access.

- **Quantised indices** (optional feature):
  - *BF16* (brain floating point 16 quantisation for exhaustive and IVF)
  - *SQ8* (int8 quantisation for exhaustive and IVF)
  - *PQ* (product quantisation for IVF)
  - *OPQ* (optimised product quantisation for IVF)

- **GPU-accelerated indices** (optional feature):
  - *Exhaustive flat index with GPU acceleration*
  - *IVF (Inverted File index) with GPU acceleration*
  - *CAGRA style index*

- **(Near) Binarised indices** (optional feature):
  - *Binary* (different types of binary quantisations for exhaustive and IVF
    indices.)
  - *RaBitQ* (RaBitQ quantisation for exhaustive and IVF indices.)
  - *TurboQuant* (TurboQuant quantisation for exhaustive and IVF indices.)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

## Example Usage

Below shows an example on how to use for example the HNSW index and query it.

### HNSW

```rust
use ann_search_rs::{build_hnsw_index, query_hnsw_index};
use faer::Mat;

// Build the HNSW index
let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
let hnsw_idx = build_hnsw_index(
  mat.as_ref(),
  16,             // m
  100,            // ef_construction
  "euclidean",    // distance metric
  42,             // seed
  false           // verbosity
);

// Query the HNSW index
// In this case we are doing a full self query
let query = Mat::from_fn(10, 128, |_, _| rand::random::<f32>());
let (hnsw_indices, hnsw_dists) = query_hnsw_index(
  mat.as_ref(),
  &hnsw_idx,
  15,             // k
  200,            // ef_search
  true,           // return distances
  false           // verbosity
);
```

The package provides a number of different approximate nearest neighbour
searches. The overall design is very similar and if you wish details on usage,
please refer to the `examples/*.rs` section which shows you the grid searches
across various parameters per given index. This and the documentation is a
good starting point to understand how the crate works.

## Performance and parameters

### Synthetic data sets

**GaussianNoise**

Generates simple Gaussian clusters with variable sizes and standard deviations.
Each cluster is a blob centred in the full dimensional space. Useful for basic
benchmarking where clusters are well-separated and occupy the entire ambient
space.

**Correlated**

Creates clusters with subspace structure where each cluster only activates a
subset of dimensions. Additionally introduces explicit correlation patterns
where groups of dimensions are linear combinations of source dimensions.
Designed to test methods that exploit inter-dimensional correlations and sparse
activation patterns.

**LowRank**

Generates data that lives in a low-dimensional subspace (intrinsic_dim) and
embeds it via random rotation into high-dimensional space (embedding_dim).
Simulates the manifold hypothesis where high-dimensional data actually lies on a
lower-dimensional manifold. Adds minimal isotropic noise to model measurement
error.

**QuantisationStress**

Combines power-law eigenvalue spectrum with norm-stratified clusters, randomly
rotated out of axis alignment. Variance decays as 1/(i+1)^decay across principal
components, so most information concentrates in a small subspace. Cluster pairs
share directions but sit at very different radii (2, 8, 20), meaning points with
near-identical angular signatures can have wildly different true distances. The
random rotation ensures no coordinate axis is privileged. Specifically targets
failure modes of aggressive quantisation: sign binarisation wastes bits on noise
dimensions and cannot distinguish radially-separated clusters, axis-aligned
product quantisation mixes informative and uninformative dimensions within
sub-vectors, and low-bit methods generally lose angular resolution in the
informative subspace. Use spectral_decay=1.5 for moderate concentration (~80% of
variance in top 25% of principal components) or 2.0 for aggressive (~90% in top
15%). Recommended with `dim=128` or `dim=256` and `n_clusters=50+`.

### Running the grid searches

To identify good basic thresholds, there are a set of different gridsearch
scripts available. These can be run via

```bash
# Run with default parameters
cargo run --example gridsearch_annoy --release

# Override specific parameters
cargo run --example gridsearch_annoy --release -- --n-samples 500000 --dim 32 --distance euclidean

# Available parameters with their defaults:
# --n-samples 150_000
# --dim 32
# --n-clusters 25
# --k 15
# --seed 42
# --distance cosine
# --data gaussian
```

Every index is trained on 150k samples with 32 dimensions distance and 25 distinct
clusters (of different sizes each). Then the index is tested against a subset of
10% of samples with a little Gaussian noise added and for full kNN self
generation. Below are the results shown for `Annoy` with the GaussianNoise
data sets.

```
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Rel dist err    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.28     1_602.87     1_606.15       1.0000       0.0000        18.31
Exhaustive (self)                                          3.28    16_690.33    16_693.60       1.0000       0.0000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt5-s:auto (query)                                  81.01        65.71       146.72       0.7006       0.0302        34.82
Annoy-nt5-s:10x (query)                                   81.01        39.97       120.98       0.5252       0.0617        34.82
Annoy-nt5-s:5x (query)                                    81.01        25.88       106.89       0.3732       0.1068        34.82
Annoy-nt5 (self)                                          81.01       349.56       430.57       0.7005       0.0325        34.82
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt10-s:auto (query)                                102.08       123.94       226.02       0.8910       0.0082        50.18
Annoy-nt10-s:10x (query)                                 102.08        77.19       179.27       0.7415       0.0243        50.18
Annoy-nt10-s:5x (query)                                  102.08        48.77       150.85       0.5626       0.0533        50.18
Annoy-nt10 (self)                                        102.08       719.81       821.89       0.8902       0.0089        50.18
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt15-s:auto (query)                                159.61       181.20       340.81       0.9582       0.0027        50.79
Annoy-nt15-s:10x (query)                                 159.61       115.06       274.67       0.8546       0.0114        50.79
Annoy-nt15-s:5x (query)                                  159.61        73.07       232.68       0.6907       0.0311        50.79
Annoy-nt15 (self)                                        159.61     1_084.09     1_243.71       0.9571       0.0029        50.79
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt25-s:auto (query)                                248.71       275.95       524.66       0.9928       0.0004        81.51
Annoy-nt25-s:10x (query)                                 248.71       182.43       431.15       0.9508       0.0031        81.51
Annoy-nt25-s:5x (query)                                  248.71       119.58       368.29       0.8410       0.0126        81.51
Annoy-nt25 (self)                                        248.71     1_868.17     2_116.89       0.9925       0.0004        81.51
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt50-s:auto (query)                                462.06       507.49       969.55       0.9998       0.0000       143.57
Annoy-nt50-s:10x (query)                                 462.06       343.51       805.57       0.9957       0.0002       143.57
Annoy-nt50-s:5x (query)                                  462.06       232.82       694.88       0.9644       0.0021       143.57
Annoy-nt50 (self)                                        462.06     3_747.19     4_209.25       0.9998       0.0000       143.57
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt75-s:auto (query)                                674.52       788.23     1_462.75       1.0000       0.0000       178.63
Annoy-nt75-s:10x (query)                                 674.52       541.05     1_215.57       0.9995       0.0000       178.63
Annoy-nt75-s:5x (query)                                  674.52       434.04     1_108.56       0.9912       0.0004       178.63
Annoy-nt75 (self)                                        674.52     6_092.44     6_766.96       1.0000       0.0000       178.63
--------------------------------------------------------------------------------------------------------------------------------
Annoy-nt100-s:auto (query)                               889.04     1_041.54     1_930.58       1.0000       0.0000       267.69
Annoy-nt100-s:10x (query)                                889.04       728.21     1_617.26       0.9999       0.0000       267.69
Annoy-nt100-s:5x (query)                                 889.04       538.96     1_428.00       0.9975       0.0001       267.69
Annoy-nt100 (self)                                       889.04     7_866.57     8_755.61       1.0000       0.0000       267.69
--------------------------------------------------------------------------------------------------------------------------------
```

Detailed benchmarks on all the "standard" indices can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_general.md).
Every index was tested on every data set with 32 dimensions (mimicking typical
single cell scenarios) and against the lowrank data set with 128 dimensions.

## Quantised indices

The crate also provides some quantised approximate nearest neighbour searches,
designed for very large data sets where memory (and query time) starts becoming
constraining. There are a total of four different quantisation methods available
(plus some binary quantisation, see further below). The crate does NOT provide
re-ranking on the full vectors (yet) for these quantised indices.

- *BF16*: An exhaustive search and IVF index are available with BF16
  quantisation. In this case the `f32` or `f64` are transformed during storage
  into `bf16` floats. These keep the range of `f32`; however, they reduce
  precision.
- *SQ8*: A scalar quantisation to `i8`. Exhaustive and IVF indices are provided.
  For each dimensions in the data, the min and max values are being computed and
  the respective data points are projected to integers between `-128` to `127`.
  This enables fast integer math; however, this comes at cost of recall of the
  real nearest neighbours.
- *PQ*: Uses product quantisation. Useful when the dimensions of the vectors
  are incredibly large and one needs to compress the index in memory even
  further. Only useful when dim ≥ 128 in most cases and ideal for very large
  dimensions. Exhaustive and IVF are available with product quantisation.
  Exhaustive PQ is not recommend due to worse performance across the board
  compared to IVF-PQ – the index was added for completeness.
- *OPQ*: Uses optimised product quantisation. Tries to de-correlate the
  residuals and can in times improve the Recall. Please see the benchmarks.
  Same indices available as for PQ.

The benchmarks can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_quantised.md).
If you wish to use these, please add the `"quantised"` feature:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["quantised"] }
```

## GPU

Three indices are also implemented in GPU-accelerated versions. A
GPU-accelerated exhaustive and IVF index. And a new addition with release
`0.2.6` a [CAGRA-style index](https://arxiv.org/abs/2308.15136). Under the hood,
this use [cubecl](https://github.com/tracel-ai/cubecl) with wgpu backend (which
makes them largely agnostic to the type of hardware), for details please check
[here](https://burn.dev/books/cubecl/getting-started/installation.html)). The
benchmarks can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_gpu.md).
To unlock GPU-acceleration, please use:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["gpu"] }
```

## Binarised indices

For the most extreme compression needs, binary indices are also provided. There
are two approaches for binarisation available in the crate:

- Bitwise binarisation either leveraging a SimHash random projection, PCA
  hashing or signed-based binarisation.
- [RaBitQ](https://arxiv.org/abs/2405.12497) binarisation while storing
  additional data for approximate distance calculations.

These can be used with Exhaustive or IVF indices and you have the option to
store the original vectors on-disk to allow for subsequent re-ranking. This
can drastically improve the Recall. To enable the feature, please use:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["binary"] }
```

The benchmarks can be found [here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_binary.md).

## Licence

MIT License

Copyright (c) 2025 Gregor Alexander Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
