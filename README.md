[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ann-search-rs

Various approximate nearest neighbour/vector searches implemented in Rust.
Helper library to be used in other libraries.

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
  - *HNSW (Hierarchical Navigable Small World)*
  - *NNDescent (Nearest Neighbour Descent)*
  (heavily inspired by [PyNNDescent](https://github.com/lmcinnes/pynndescent)).
  - *LSH (Locality Sensitive Hashing)*
  - *IVF (Inverted File index)*
  - *Exhaustive flat index*
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

- **Binarised indices** (optional feature):
  - *Binary* (different types of binary quantisations for exhaustive and IVF
    indices.)
  - *RaBitQ* (RaBitQ quantisation for exhaustive and IVF indices.)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

To note, I have changed some of the interfaces between versions.

## Roadmap

- ~~First GPU support~~ (Implemented with version `0.2.1` of the crate).
- ~~Binary indices~~ (Also implemented with version `0.2.1`).
- ~~Proper SIMD~~ (Implemented with `0.2.2` via the [wide crate](https://docs.rs/wide/latest/wide/)).
- Option to save indices on-disk and maybe do on-disk querying ... ? The binary
  indices already use some aspects of on-disk storage.
- More GPU support for other indices. TBD, needs to warrant the time investment.

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
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.58     1_725.77     1_729.35       1.0000     0.000000        18.31
Exhaustive (self)                                          3.58    17_651.75    17_655.33       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
HNSW-M16-ef50-s50 (query)                                922.59        58.86       981.46       0.9991     0.000799        51.60
HNSW-M16-ef50-s75 (query)                                922.59        81.45     1_004.05       0.9998     0.000221        51.60
HNSW-M16-ef50-s100 (query)                               922.59       106.08     1_028.68       0.9999     0.000087        51.60
HNSW-M16-ef50 (self)                                     922.59     1_085.24     2_007.84       0.9999     0.000083        51.60
HNSW-M16-ef100-s50 (query)                             1_688.76        68.75     1_757.50       0.9997     0.000272        51.60
HNSW-M16-ef100-s75 (query)                             1_688.76       101.03     1_789.79       0.9999     0.000063        51.60
HNSW-M16-ef100-s100 (query)                            1_688.76       144.57     1_833.33       1.0000     0.000027        51.60
HNSW-M16-ef100 (self)                                  1_688.76     1_205.42     2_894.17       1.0000     0.000009        51.60
HNSW-M16-ef200-s50 (query)                             3_149.52        70.57     3_220.09       0.9997     0.000191        51.60
HNSW-M16-ef200-s75 (query)                             3_149.52        98.13     3_247.65       1.0000     0.000014        51.60
HNSW-M16-ef200-s100 (query)                            3_149.52       127.51     3_277.03       1.0000     0.000005        51.60
HNSW-M16-ef200 (self)                                  3_149.52     1_236.76     4_386.28       1.0000     0.000004        51.60
--------------------------------------------------------------------------------------------------------------------------------
HNSW-M24-ef100-s50 (query)                             1_620.61        67.83     1_688.44       0.9998     0.000110        51.60
HNSW-M24-ef100-s75 (query)                             1_620.61        93.32     1_713.93       1.0000     0.000008        51.60
HNSW-M24-ef100-s100 (query)                            1_620.61       120.86     1_741.48       1.0000     0.000000        51.60
HNSW-M24-ef100 (self)                                  1_620.61     1_226.84     2_847.46       1.0000     0.000004        51.60
HNSW-M24-ef200-s50 (query)                             2_814.24        67.77     2_882.02       0.9999     0.000075        51.60
HNSW-M24-ef200-s75 (query)                             2_814.24        99.43     2_913.67       1.0000     0.000005        51.60
HNSW-M24-ef200-s100 (query)                            2_814.24       130.41     2_944.65       1.0000     0.000001        51.60
HNSW-M24-ef200 (self)                                  2_814.24     1_319.07     4_133.32       1.0000     0.000006        51.60
HNSW-M24-ef300-s50 (query)                             4_291.04        87.58     4_378.62       0.9999     0.000073        51.60
HNSW-M24-ef300-s75 (query)                             4_291.04       101.72     4_392.76       1.0000     0.000005        51.60
HNSW-M24-ef300-s100 (query)                            4_291.04       126.36     4_417.40       1.0000     0.000001        51.60
HNSW-M24-ef300 (self)                                  4_291.04     1_236.42     5_527.46       1.0000     0.000001        51.60
--------------------------------------------------------------------------------------------------------------------------------
HNSW-M32-ef200-s50 (query)                             2_779.77        81.97     2_861.74       0.9995     9.232197        83.60
HNSW-M32-ef200-s75 (query)                             2_779.77       100.55     2_880.32       1.0000     0.000018        83.60
HNSW-M32-ef200-s100 (query)                            2_779.77       135.59     2_915.36       1.0000     0.000013        83.60
HNSW-M32-ef200 (self)                                  2_779.77     1_258.12     4_037.89       1.0000     0.000002        83.60
HNSW-M32-ef300-s50 (query)                             3_827.90        73.96     3_901.86       1.0000     0.000034        83.60
HNSW-M32-ef300-s75 (query)                             3_827.90        99.55     3_927.44       1.0000     0.000005        83.60
HNSW-M32-ef300-s100 (query)                            3_827.90       125.16     3_953.06       1.0000     0.000000        83.60
HNSW-M32-ef300 (self)                                  3_827.90     1_272.95     5_100.85       1.0000     0.000003        83.60
--------------------------------------------------------------------------------------------------------------------------------
```

Detailed benchmarks on all the standard benchmarks can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_general.md).
Every index was tested on every data set.

## Quantised indices

The crate also provides some quantised approximate nearest neighbour searches,
designed for very large data sets where memory and time both start becoming
incredibly constraining. There are a total of four different quantisation
methods available (plus some binary quantisation, see further below). The crate
does NOT provide re-ranking on the full vectors (yet).

- *BF16*: An exhaustive search and IVF index are available with BF16
  quantisation. In this case the `f32` or `f64` are transformed during storage
  into `bf16` floats. These keep the range of `f32`; however, they reduce
  precision.
- *SQ8*: A scalar quantisation to `i8`. Exhaustive and IVF indices are provided.
  For each dimensions in the data, the min and max values are being computed and
  the respective data points are projected to integers between `-128` to `127`.
  This enables fast integer math; however, this comes at cost of precision.
- *PQ*: Uses product quantisation. Useful when the dimensions of the vectors
  are incredibly large and one needs to compress the index in memory even
  further. Only useful when dim ≥ 128 in most cases and ideal for very large
  dimensions. Only IVF is available with product quantisation.
- *OPQ*: Uses optimised product quantisation. Tries to de-correlate the
  residuals and can in times improve the Recall. Please see the benchmarks.
  Only IVF is available with optimised product quantisation.

The benchmarks can be found
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_quantised.md).
If you wish to use these, please add the `"quantised"` feature:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["quantised"] }
```

## GPU

Two indices are also implemented in GPU-accelerated versions. The exhaustive
search and the IVF index. Under the hood, this uses
[cubecl](https://github.com/tracel-ai/cubecl) with wgpu backend (system agnostic,
for details please check [here](https://burn.dev/books/cubecl/getting-started/installation.html)).
Let's first look at the indices compared against exhaustive (CPU). You can
of course provide other backends.

The benchmarks can be found [here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_gpu.md).
To unlock GPU-acceleration, please use:

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["gpu"] }
```

There is for sure room for improvement in terms of the design of the indices,
but they do the job as is. Longer term, I will add smarter design(s) to avoid
the CPU to GPU and back copying of data.

## Binarised indices

For the extreme compression needs, binary indices are also provided. There
are two approaches for binarisation

- Bitwise binarisation either leveraging a SimHash random projection approach
  or ITQ via PCA.
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
