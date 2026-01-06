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
Rust of some of these. Over time, I started getting interested into vector
searches and implement WAY more indices and new stuff into this than initially
anticipated.

## Features

- **Multiple ANN algorithms**:
  - *Annoy (Approximate Nearest Neighbours Oh Yeah)*
  - *HNSW (Hierarchical Navigable Small World)*
  - *NNDescent (Nearest Neighbour Descent)*
  (heavily inspired by [PyNNDescent](https://github.com/lmcinnes/pynndescent)).
  - *LSH (Locality Sensitive Hashing)*
  - *IVF (Inverted File index)*
  - *Exhaustive flat index*

- **Distance metrics**:
  - Euclidean
  - Cosine
  - More to come maybe... ?

- **High performance**: Optimised implementations with SIMD-friendly code,
heavy multi-threading were possible and optimised structures for memory access.

- **Quantised indices** (optional feature):
  - *BF16* (brain floating point 16 quantisation for exhaustive and IVF)
  - *SQ8* (int8 quantisation for exhaustive and IVF)
  - *PQ* (product quantisation for IVF)
  - *OPQ* (optimised product quantisation for IVF)

- **GPU-accelerated indices** (optional feature):
  - *Exhaustive flat index with GPU acceleration*
  - *IVF (Inverted File index) with GPU acceleration*

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
cargo run --example gridsearch_annoy --release -- --n-cells 500000 --dim 32 --distance euclidean

# Available parameters with their defaults:
# --n-cells 150_000
# --dim 32
# --n-clusters 25
# --k 15
# --seed 42
# --distance cosine
# --data gaussian
```

Every index is trained on 150k cells with 32 dimensions distance and 25 distinct 
clusters (of different sizes each). Then the index is tested against a subset of
10% of cells with a little Gaussian noise added and for full kNN self 
generation. Below are the results shown for `Annoy` with the GaussianNoise
data sets.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.21     2_350.81     2_354.02       1.0000     0.000000        18.31
Exhaustive (self)                                     3.21    23_238.47    23_241.67       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
Annoy-nt5-s:auto (query)                             75.49       119.15       194.64       0.6834    40.648110        33.67
Annoy-nt5-s:10x (query)                              75.49        57.14       132.63       0.5240    40.565845        33.67
Annoy-nt5-s:5x (query)                               75.49        36.03       111.52       0.3732    40.431273        33.67
Annoy-nt5 (self)                                     75.49       900.10       975.59       0.6838    40.084992        33.67
Annoy-nt10-s:auto (query)                           106.36       169.65       276.02       0.8810    40.727710        49.03
Annoy-nt10-s:10x (query)                            106.36       106.73       213.09       0.7412    40.683223        49.03
Annoy-nt10-s:5x (query)                             106.36        66.57       172.94       0.5626    40.594216        49.03
Annoy-nt10 (self)                                   106.36     1_687.77     1_794.13       0.8804    40.163717        49.03
Annoy-nt15-s:auto (query)                           153.18       241.72       394.90       0.9524    40.748981        49.65
Annoy-nt15-s:10x (query)                            153.18       155.48       308.66       0.8546    40.723903        49.65
Annoy-nt15-s:5x (query)                             153.18        97.11       250.29       0.6907    40.662767        49.65
Annoy-nt15 (self)                                   153.18     2_415.14     2_568.32       0.9516    40.184645        49.65
Annoy-nt25-s:auto (query)                           241.19       357.15       598.34       0.9908    40.758739        80.37
Annoy-nt25-s:10x (query)                            241.19       238.03       479.22       0.9508    40.750722        80.37
Annoy-nt25-s:5x (query)                             241.19       145.40       386.59       0.8410    40.720771        80.37
Annoy-nt25 (self)                                   241.19     3_557.66     3_798.85       0.9906    40.194411        80.37
Annoy-nt50-s:auto (query)                           447.88       586.98     1_034.87       0.9997    40.760798       142.43
Annoy-nt50-s:10x (query)                            447.88       414.37       862.25       0.9957    40.760169       142.43
Annoy-nt50-s:5x (query)                             447.88       276.64       724.53       0.9644    40.754108       142.43
Annoy-nt50 (self)                                   447.88     5_870.01     6_317.89       0.9997    40.196443       142.43
Annoy-nt75-s:auto (query)                           664.56       853.30     1_517.86       1.0000    40.760868       177.49
Annoy-nt75-s:10x (query)                            664.56       618.60     1_283.16       0.9995    40.760801       177.49
Annoy-nt75-s:5x (query)                             664.56       412.71     1_077.27       0.9912    40.759442       177.49
Annoy-nt75 (self)                                   664.56     8_523.23     9_187.79       1.0000    40.196503       177.49
Annoy-nt100-s:auto (query)                          871.80     1_084.39     1_956.19       1.0000    40.760873       266.55
Annoy-nt100-s:10x (query)                           871.80       778.07     1_649.87       0.9999    40.760864       266.55
Annoy-nt100-s:5x (query)                            871.80       541.85     1_413.65       0.9975    40.760539       266.55
Annoy-nt100 (self)                                  871.80    10_826.24    11_698.04       1.0000    40.196506       266.55
---------------------------------------------------------------------------------------------------------------------------
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