[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ann-search-rs

Various approximate nearest neighbour searches implemented in Rust. Helper
library to be used in other libraries. 

## Description

Extracted function for approximate nearest neighbour searches specifically
with single cell in mind from [bixverse](https://github.com/GregorLueg/bixverse),
a R/Rust package designed for computational biology, that has a ton of 
functionality for single cell. Within all of the single cell functions, kNN 
generations are ubiqituos, thus, I want to expose the APIs to other packages.
Feel free to use these implementations where you might need approximate nearest
neighbour searches. This work is based on the great work from others who
figured out how to design these algorithms and is just an implementation into
Rust of some of these.

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
  - *IVF-SQ8* (with scalar quantisation)
  - *IVF-PQ* (with product quantisation)
  - *IVF-OPQ* (with optimised product quantisation)

- **GPU-accelerated indices** (optional feature):
  - *Exhaustive flat index with GPU acceleration*
  - *IVF (Inverted File index) with GPU acceleration*

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

To note, I have changed some of the interfaces between versions.

## Roadmap

- ~~First GPU support~~ (Implemented with version `0.2.1` of the crate).
- Option to save indices on-disk and maybe do on-disk querying ... ? 
- More GPU support for other indices. TBD, needs to warrant the time investment.
  For the use cases of the author this crate suffices atm more than enough.
  Additionally, need to figure out better ways to do the kernel magic as the
  CPU to GPU transfers are quite costly and costing performance.

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
across various parameters per given index.

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
generation. Below are the results shown for `Annoy`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive (query)                          3.22      2323.61      2326.83       1.0000     0.000000
Exhaustive (self)                           3.22     24526.99     24530.21       1.0000     0.000000
Annoy-nt5-s:auto (query)                   74.71        90.94       165.65       0.6392    81.632962
Annoy-nt5-s:10x (query)                    74.71        59.54       134.25       0.5153    81.536609
Annoy-nt5-s:5x (query)                     74.71        37.26       111.97       0.3708    81.343675
Annoy-nt5 (self)                           74.71       830.23       904.94       0.6408    81.683927
Annoy-nt10-s:auto (query)                 109.86       155.44       265.30       0.8471    81.764138
Annoy-nt10-s:10x (query)                  109.86       108.39       218.25       0.7365    81.712452
Annoy-nt10-s:5x (query)                   109.86        67.11       176.97       0.5593    81.582371
Annoy-nt10 (self)                         109.86      1518.47      1628.33       0.8479    81.814313
Annoy-nt15-s:auto (query)                 161.83       215.60       377.43       0.9338    81.803269
Annoy-nt15-s:10x (query)                  161.83       153.45       315.28       0.8539    81.774152
Annoy-nt15-s:5x (query)                   161.83        92.75       254.59       0.6896    81.684499
Annoy-nt15 (self)                         161.83      2132.04      2293.87       0.9338    81.853076
Annoy-nt25-s:auto (query)                 253.70       321.63       575.33       0.9853    81.822332
Annoy-nt25-s:10x (query)                  253.70       233.69       487.39       0.9511    81.812750
Annoy-nt25-s:5x (query)                   253.70       155.44       409.15       0.8398    81.769091
Annoy-nt25 (self)                         253.70      3745.23      3998.93       0.9854    81.872143
Annoy-nt50-s:auto (query)                 546.70      1187.52      1734.22       0.9995    81.826775
Annoy-nt50-s:10x (query)                  546.70       440.92       987.62       0.9959    81.826054
Annoy-nt50-s:5x (query)                   546.70       294.44       841.14       0.9654    81.817842
Annoy-nt50 (self)                         546.70      5910.40      6457.10       0.9994    81.876491
Annoy-nt75-s:auto (query)                 706.48       882.82      1589.31       1.0000    81.826905
Annoy-nt75-s:10x (query)                  706.48       622.27      1328.75       0.9995    81.826820
Annoy-nt75-s:5x (query)                   706.48       413.66      1120.15       0.9909    81.824865
Annoy-nt75 (self)                         706.48      7958.33      8664.81       1.0000    81.876624
Annoy-nt100-s:auto (query)                927.31      1051.21      1978.51       1.0000    81.826908
Annoy-nt100-s:10x (query)                 927.31       803.39      1730.70       0.9999    81.826897
Annoy-nt100-s:5x (query)                  927.31       552.50      1479.80       0.9974    81.826428
Annoy-nt100 (self)                        927.31     10121.62     11048.93       1.0000    81.876631
----------------------------------------------------------------------------------------------------
```

Detailed benchmarks on all the standard benchmarks can be found 
[here](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_general.md)

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
  This enables fast integer math; however, this comes at cost of precision. Only 
  IVF is available with SQ8 quantisation.
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
To unlock GPU-acceleration, please use

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["gpu"] } 
```

There is for sure room for improvement in terms of the design of the indices,
but they do the job as is. Longer term, I will add smarter design(s) to avoid
the CPU to GPU and back copying of data.

## Binarised indices

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