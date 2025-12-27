[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)

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
  false.          // verbosity
);
```

The package provides a number of different approximate nearest neighbour
searches. The overall design is very similar and if you wish details on usage,
please refer to the `examples/*.rs` section which shows you the grid searches
across various parameters per given index.

## Performance and parameters

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
# --seed 10101
# --distance cosine
# --data gaussian
```

For every index, 150k cells with 32 dimensions distance and 25 distinct clusters 
(of different sizes each) in the synthetic data has been run. The results for 
the different indices are show below. For details on the synthetic data 
function, see `./examples/commons/mod.rs`. This was run on an M1 Max MacBoo Pro
with 64 GB of unified memory, see one example below:

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.02     23204.51     23208.53       1.0000     0.000000
Annoy-nt5-s:auto                           73.62       794.92       868.54       0.6408    81.683927
Annoy-nt5-s:10x                            73.62       570.38       644.00       0.5175    81.587548
Annoy-nt5-s:5x                             73.62       359.21       432.82       0.3713    81.392453
Annoy-nt10-s:auto                         100.36      1519.07      1619.43       0.8479    81.814313
Annoy-nt10-s:10x                          100.36      1061.24      1161.60       0.7378    81.763178
Annoy-nt10-s:5x                           100.36       669.30       769.66       0.5603    81.632177
Annoy-nt15-s:auto                         158.77      2145.18      2303.94       0.9338    81.853076
Annoy-nt15-s:10x                          158.77      1528.20      1686.96       0.8542    81.824079
Annoy-nt15-s:5x                           158.77       952.47      1111.24       0.6898    81.734461
Annoy-nt25-s:auto                         234.09      3317.58      3551.68       0.9854    81.872143
Annoy-nt25-s:10x                          234.09      2407.78      2641.87       0.9512    81.862444
Annoy-nt25-s:5x                           234.09      1448.60      1682.69       0.8394    81.818440
Annoy-nt50-s:auto                         450.88      5865.30      6316.18       0.9994    81.876491
Annoy-nt50-s:10x                          450.88      4471.22      4922.10       0.9959    81.875733
Annoy-nt50-s:5x                           450.88      2975.32      3426.20       0.9647    81.867209
Annoy-nt75-s:auto                         687.39      8105.55      8792.94       1.0000    81.876624
Annoy-nt75-s:10x                          687.39      6273.23      6960.62       0.9995    81.876551
Annoy-nt75-s:5x                           687.39      4329.62      5017.01       0.9909    81.874592
Annoy-nt100-s:auto                        900.40     10331.99     11232.39       1.0000    81.876631
Annoy-nt100-s:10x                         900.40      8045.77      8946.17       0.9999    81.876621
Annoy-nt100-s:5x                          900.40      5620.07      6520.48       0.9974    81.876118
----------------------------------------------------------------------------------------------------
```

Detailed benchmarks on the standard benchmarks can be found [here](/docs/benchmarks_general.md)

## Quantised indices

The crate also provides some quantised approximate nearest neighbour searches, 
designed for very large data sets where memory and time both start becoming 
incredibly constraining. There are a total of three different quantisation
methods available:

- *IVF-SQ8*: Uses a scalar quantisation and transforms the different feature
  dimensions to `i8` and leverages very fast integer math to compute the nearest
  neighbours (at a loss of precision).
- *IVF-PQ*: Uses product quantisation. Useful when the dimensions of the vectors
  are incredibly large and one needs to compress the index in memory. However,
  as you can see below, this comes at a cost of Recall.
- *IVF-OPQ*: Uses optimised product quantisation. Tries to de-correlate the
  residuals 

The benchmarks can be found [here](/docs/benchmarks_quantised.md). If you wish
to use these, please add the "quantised" feature

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

The benchmarks can be found [here](/docs/benchmarks_gpu.md). To unlock GPU-
acceleration, please use

```toml
[dependencies]
ann-search-rs = { version = "*", features = ["gpu"] } 
```

There is for sure room for improvement in terms of the design of the indices,
but they do the job as is. Longer term, I will add smarter design(s) to avoid
the CPU to GPU and back copying of data.

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