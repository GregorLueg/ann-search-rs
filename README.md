[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![docs.rs](https://img.shields.io/docsrs/ann-search-rs)](https://docs.rs/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ann-search-rs

Various approximate nearest neighbour/vector searches implemented in Rust (with
focus on computational biology applications, very specifically single cell). The
search algorithms are designed for high in-memory performance. Indices can be
saved to disk and loaded back in via the `serialise` feature. Longer term, I
might add the option to add/remove vectors from some of the indices.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Gridsearches and performance](#running-the-grid-searches)
- [FEATURE: quantisation](#quantised-indices)
- [FEATURE: GPU acceleration](#gpu)
- [FEATURE: Binary indices](#binarised-indices)
- [FEATURE: Saving and loading](#saving-and-loading-indices)

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
[one out](https://github.com/GregorLueg/ann-search-rs/blob/main/CHANGELOG.md)

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
  - *Navigating Spread-out Graph (NSG)*
  - *Relative NN-Descent*
  - *Vanama (the graph powering DiskANN)*

- **Distance metrics**:
  - Euclidean
  - Cosine
  - Manhattan (support for a subset of the approximate nearest neighbour
    searches).

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
  - *TurboQuant* (Turbo Quantisation for exhaustive and IVF indices.)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*"
```

### Note

With version `"0.4.2"` some breaking API changes were introduced: this harmonise
several of the functions and avoid panics in favour of errors. A key change
was also the update to cubecl `"0.1.0"` which changes quite a few APIs for the
GPU-accelerated version.

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

Isotropic Gaussian clusters with variable sizes and per-cluster standard
deviations, centred across the full ambient space. A fraction of points
(`DEFAULT_BRIDGE_FRACTION`: 20%) are placed on thin Gaussian tubes interpolating
between each cluster and its nearest neighbour, so clusters are connected rather
than fully isolated. Useful as a baseline where structure is axis-agnostic and
mostly well-separated.

**Correlated**

Well-separated clusters, each an arbitrarily-oriented ellipsoid with a low-rank
power-law covariance (cluster-local anisotropy), plus a globally-shared off-axis
subspace carrying inter-dimension correlation. The structured variance is
deliberately not aligned with the coordinate axes, so a learned rotation (OPQ)
can recover it while axis-aligned PQ cannot. `correlation_strength` splits
variance between the shared global subspace (1.0) and the cluster-local one
(0.0). Designed to separate rotation-aware from axis-aligned quantisers.

**LowRank**

Data that genuinely lives in a low-dimensional subspace (`intrinsic_dim`),
isometrically embedded into the full space (`dim`) via an orthonormal rotation,
with minimal isotropic noise for measurement error. Clusters follow a two-level
lineage hierarchy (roots, then leaves around each root), and a fraction of points
(`DEFAULT_TRAJECTORY_FRACTION`: 15%) lie on quadratic Bezier trajectories
chaining leaves within a lineage. Models the manifold hypothesis: locally
low-dimensional, globally curved, so a single global rotation cannot flatten it.

**CellEmbedding**

Approximates the geometry of foundation-model cell embeddings (Geneformer/scGPT
style) at 256-768 dimensions. Combines five properties, each targeting a distinct
quantisation failure mode: a strong shared anisotropy cone (large mean offset,
high pairwise cosine) that defeats cosine- and sign-based methods; a few
axis-aligned rogue dimensions with outsized variance that dominate dot products
and starve PQ codebooks; per-cell-type low-rank oriented subspaces whose union is
full-rank, so no single OPQ rotation fully aligns them; differentiation
trajectories between related types (curved manifold); and per-cell lognormal norm
variation standing in for library size/sequencing depth. Recommended with
dim=256 to dim=768 and n_clusters=25+. Note the constants here are
plausible rather than fitted to real embeddings — if you have a real
Geneformer/scGPT matrix, fit the cone, rogue-dim magnitudes and eigenspectrum to
it before trusting benchmark conclusions.

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
Exhaustive (query)                                         3.07     1_512.56     1_515.63       1.0000          1.0000        18.31
Exhaustive (self)                                          3.07    15_883.64    15_886.71       1.0000          1.0000        18.31
Annoy-nt5-s:auto (query)                                  81.78        63.93       145.71       0.7410          1.0235        31.96
Annoy-nt5-s:10x (query)                                   81.78        39.44       121.22       0.5881          1.0505        31.96
Annoy-nt5-s:5x (query)                                    81.78        24.35       106.13       0.4530          1.0933        31.96
Annoy-nt5 (self)                                          81.78       350.63       432.41       0.7393          1.0237        31.96
Annoy-nt10-s:auto (query)                                117.23       112.86       230.09       0.8963          1.0068        44.46
Annoy-nt10-s:10x (query)                                 117.23        76.48       193.71       0.7595          1.0205        44.46
Annoy-nt10-s:5x (query)                                  117.23        46.14       163.37       0.6031          1.0464        44.46
Annoy-nt10 (self)                                        117.23       690.51       807.74       0.8954          1.0068        44.46
Annoy-nt15-s:auto (query)                                170.86       158.72       329.59       0.9552          1.0024        60.83
Annoy-nt15-s:10x (query)                                 170.86       106.63       277.50       0.8532          1.0101        60.83
Annoy-nt15-s:5x (query)                                  170.86        68.97       239.83       0.7026          1.0278        60.83
Annoy-nt15 (self)                                        170.86     1_040.97     1_211.83       0.9537          1.0025        60.83
Annoy-nt25-s:auto (query)                                258.99       253.76       512.74       0.9895          1.0005        70.08
Annoy-nt25-s:10x (query)                                 258.99       169.26       428.24       0.9386          1.0033        70.08
Annoy-nt25-s:5x (query)                                  258.99       111.95       370.93       0.8206          1.0128        70.08
Annoy-nt25 (self)                                        258.99     1_690.02     1_949.00       0.9891          1.0005        70.08
Annoy-nt50-s:auto (query)                                510.87       458.09       968.96       0.9994          1.0000       120.71
Annoy-nt50-s:10x (query)                                 510.87       313.73       824.60       0.9890          1.0004       120.71
Annoy-nt50-s:5x (query)                                  510.87       218.49       729.36       0.9364          1.0032       120.71
Annoy-nt50 (self)                                        510.87     3_246.34     3_757.21       0.9994          1.0000       120.71
Annoy-nt75-s:auto (query)                                714.24       723.99     1_438.22       0.9999          1.0000       218.84
Annoy-nt75-s:10x (query)                                 714.24       548.36     1_262.59       0.9974          1.0001       218.84
Annoy-nt75-s:5x (query)                                  714.24       437.78     1_152.01       0.9725          1.0012       218.84
Annoy-nt75 (self)                                        714.24     5_215.59     5_929.82       1.0000          1.0000       218.84
Annoy-nt100-s:auto (query)                               912.43       905.49     1_817.92       1.0000          1.0000       221.97
Annoy-nt100-s:10x (query)                                912.43       662.54     1_574.97       0.9993          1.0000       221.97
Annoy-nt100-s:5x (query)                                 912.43       567.15     1_479.58       0.9868          1.0005       221.97
Annoy-nt100 (self)                                       912.43     7_335.82     8_248.24       1.0000          1.0000       221.97
-----------------------------------------------------------------------------------------------------------------------------------
```

Detailed benchmarks on all the "standard" CPU-based indices can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_standard.md).
Every index was tested on every data set with 32 dimensions (mimicking typical
single cell scenarios) and against the cell embedding data set with 128
dimensions.

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
  hashing or sign-based binarisation. Sign bits taken in the global frame tell
  you which cluster a point sits in, not where it sits inside that cluster, and
  the latter is what a kNN search needs. SimHash and PCA hashing centre on a
  per-feature mean; sign-based centres only on an IVF index, where it takes the
  residual against each vector's own cell centroid. Both are trades rather than
  free wins, see `CHANGELOG.md` for the regimes where each helps and hurts.
- [RaBitQ](https://arxiv.org/abs/2405.12497) binarisation while storing
  additional data for approximate distance calculations.
- [TurboQuant](https://arxiv.org/abs/2504.19874) a data-oblivious quantiser that
  randomly rotates each vector and applies a per-coordinate scalar quantiser, so
  there are no learned codebooks and indexing is near-instant.

These can be used with Exhaustive or IVF indices and you have the option to
store the original vectors on-disk to allow for subsequent re-ranking. This
can drastically improve the Recall. To enable the feature, please use:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["binary"] }
```

The benchmarks can be found [here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_binary.md).

## Saving and loading indices

Building an index over a few million cells takes minutes. Doing it again on
every process start gets old fast. The `serialise` feature adds `save_index` and
`load_index` to every CPU, quantised and binary index:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["serialise"] }
```

```rust
use ann_search_rs::cpu::hnsw::HnswIndex;
use ann_search_rs::{build_hnsw_index, load_index, save_index};

let index = build_hnsw_index(mat.as_ref(), 16, 200, "cosine", 42, false);
save_index(&index, "my_index")?;

let index: HnswIndex<f32> = load_index("my_index")?;
```

The `IndexIo` trait is in the prelude, so with
`use ann_search_rs::prelude::*;` the method form
`index.save_index("my_index")` and `HnswIndex::<f32>::load_index("my_index")`
works directly. The index types themselves live in their own modules, not the
prelude.

An index is a *directory*, not a single file. `index.bin` holds the payload
(serde plus bincode, little-endian, variable-length integers so the file moves
between 32- and 64-bit machines). The binary indices additionally carry their
on-disk re-ranking store into the same directory, so a saved index is
self-contained and can be moved. Saving into the directory the store already
lives in skips the copy. Note the store files are raw native-endian dumps, so a
bundle carrying one does not survive a move between a little- and a big-endian
machine, even though `index.bin` on its own would.

```
my_index/
  index.bin
  vectors_flat.bin   # binary indices with a re-ranking store
  norms.bin
```

`index.bin` opens with a header recording the format version, the index kind and
the float width. Loading an HNSW index as an IVF one, or an `f32` index as
`f64`, gives you a typed error rather than garbage.

Saving is atomic in the sense that matters: every file goes to a temporary name
first, and `index.bin` is the last thing renamed into place. A save that dies
half way leaves the previous bundle alone, or leaves no `index.bin` and fails
loudly on the next load. It never leaves a directory that loads clean and
answers wrongly.

The GPU indices are not covered. They hold live device handles, and
`IvfIndexGpu` keeps its centroids on the GPU with no host copy, so persisting
them needs more than a derive. Watch this space.

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
