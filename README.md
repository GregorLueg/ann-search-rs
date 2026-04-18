[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ann-search-rs

Various approximate nearest neighbour/vector searches implemented in Rust (with
focus on computational biology applications, very specifically single cell).

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
  - *KmKnn (k-means kNN)*
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
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.11     1_546.34     1_549.45       1.0000          1.0000        18.31
Exhaustive (self)                                          3.11    15_706.15    15_709.27       1.0000          1.0000        18.31
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt5-s:auto (query)                                  80.90        64.18       145.08       0.7006          1.0326        34.82
Annoy-nt5-s:10x (query)                                   80.90        40.09       120.99       0.5252          1.0665        34.82
Annoy-nt5-s:5x (query)                                    80.90        24.65       105.55       0.3732          1.1150        34.82
Annoy-nt5 (self)                                          80.90       357.33       438.23       0.7005          1.0328        34.82
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt10-s:auto (query)                                 99.17       122.28       221.46       0.8910          1.0089        50.18
Annoy-nt10-s:10x (query)                                  99.17        78.29       177.47       0.7415          1.0263        50.18
Annoy-nt10-s:5x (query)                                   99.17        48.12       147.29       0.5626          1.0575        50.18
Annoy-nt10 (self)                                         99.17       716.11       815.28       0.8902          1.0090        50.18
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt15-s:auto (query)                                165.86       177.81       343.68       0.9582          1.0029        50.79
Annoy-nt15-s:10x (query)                                 165.86       116.37       282.23       0.8546          1.0124        50.79
Annoy-nt15-s:5x (query)                                  165.86        72.10       237.97       0.6907          1.0336        50.79
Annoy-nt15 (self)                                        165.86     1_088.84     1_254.70       0.9571          1.0030        50.79
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt25-s:auto (query)                                246.44       276.97       523.41       0.9928          1.0004        81.51
Annoy-nt25-s:10x (query)                                 246.44       180.98       427.42       0.9508          1.0034        81.51
Annoy-nt25-s:5x (query)                                  246.44       119.28       365.72       0.8410          1.0137        81.51
Annoy-nt25 (self)                                        246.44     1_852.87     2_099.31       0.9925          1.0004        81.51
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt50-s:auto (query)                                483.07       513.34       996.42       0.9998          1.0000       143.57
Annoy-nt50-s:10x (query)                                 483.07       345.13       828.20       0.9957          1.0002       143.57
Annoy-nt50-s:5x (query)                                  483.07       231.50       714.57       0.9644          1.0023       143.57
Annoy-nt50 (self)                                        483.07     3_705.56     4_188.63       0.9998          1.0000       143.57
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt75-s:auto (query)                                676.99       812.03     1_489.03       1.0000          1.0000       178.63
Annoy-nt75-s:10x (query)                                 676.99       569.81     1_246.80       0.9995          1.0000       178.63
Annoy-nt75-s:5x (query)                                  676.99       438.39     1_115.39       0.9912          1.0005       178.63
Annoy-nt75 (self)                                        676.99     6_165.07     6_842.06       1.0000          1.0000       178.63
-----------------------------------------------------------------------------------------------------------------------------------
Annoy-nt100-s:auto (query)                               906.98     1_046.57     1_953.55       1.0000          1.0000       267.69
Annoy-nt100-s:10x (query)                                906.98       701.32     1_608.30       0.9999          1.0000       267.69
Annoy-nt100-s:5x (query)                                 906.98       632.66     1_539.64       0.9975          1.0001       267.69
Annoy-nt100 (self)                                       906.98     8_425.02     9_332.00       1.0000          1.0000       267.69
-----------------------------------------------------------------------------------------------------------------------------------
```

Detailed benchmarks on all the "standard" CPU-based indices can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_standard.md).
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
