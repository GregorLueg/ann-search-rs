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
  - [**Annoy (Approximate Nearest Neighbours Oh Yeah)**](https://github.com/spotify/annoy).
  A version is implemented here with some modifications.
  - [**HNSW (Hierarchical Navigable Small World)**](https://arxiv.org/abs/1603.09320). 
  A version with some slight modifications has been implemented in this package,
  attempting rapid index generation.
  - **NNDescent (Nearest Neighbour Descent)** 
  (heavily inspired by [PyNNDescent](https://github.com/lmcinnes/pynndescent)).
  - [**FANNG**](https://openaccess.thecvf.com/content_cvpr_2016/papers/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.pdf).
  A version with some modifications in terms of starting node generation and
  some parallel operations in the index generation for speed purposes.

- **Distance metrics**:
  - Euclidean
  - Cosine
  - More to come maybe... ?

- **High performance**: Optimised implementations with SIMD-friendly code,
heavy multi-threading were possible and optimised structures for memory access.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

## Usage

Below shows an example on how to use for example the HNSW index and query it.

### HNSW

```rust
use ann_search_rs::{build_hnsw_index, query_hnsw_index, Dist, parse_ann_dist};
use faer::Mat;

// Build the HNSW index
let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
let hnsw_idx = build_hnsw_index(
  mat.as_ref(), 
  16,             // m
  200,            // ef_construction
  "euclidean",    // distance metric
  42,             // seed
  false           // verbosity
);

// Query the HNSW index
let query = Mat::from_fn(10, 128, |_, _| rand::random::<f32>());
let (hnsw_indices, hnsw_dists) = query_hnsw_index(
  mat.as_ref(), 
  &hnsw_idx, 
  15,             // k
  400,            // ef_search
  true,           // return distances
  false.          // verbosity
);
```

### Annoy

`ann-search-rs` also offers other versions of approximate nearest neighbour
searches, such as a more custom implementation of Annoy. The principle remains
the same:
- Generation of random trees splitting the data on a hyperplane (tree 
  generation is executed in parallel).
- A difference is that the search allows to look at different branches in case
  of close boundaries. 

```rust
use ann_search_rs::{build_annoy_index, query_annoy_index, Dist, parse_ann_dist};
use faer::Mat;

// Build the Annoy index
let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
let annoy_idx = build_annoy_index(
  mat.as_ref(), 
  100,            // number of trees to generate
  42,             // seed
);

// Query the Annoy index
let query = Mat::from_fn(10, 128, |_, _| rand::random::<f32>());
let (annoy_indices, annoy_dists) = query_annoy_index(
  mat.as_ref(), 
  15,             // k
  "euclidean",    // distance metric
  100,            // search budget
  true,           // return distances
  false.          // verbosity
);
```

## Performance 

Different indices show in parts very different index build times and querying 
times. Below are some of the results observed (please check the examples to run
them). You can do so with

```rust
// This would run the smallest benchmark
cargo run --example benchmarks_10k_8dim --release
```

All of the benchmarks below were run with k = 15 and Euclidean distances. The
exhaustive search serves as baseline.

### 10k samples, 20 distinct clusters, 8 dimensions

This could represent a smaller, low complexity single cell result. For Annoy
the search budget is set to `n_tree * 2`. The number of random entry points
in FANNG is set to `25`.

| Method | Build (ms) | Query (ms) | Recall@k | Dist Error |
|--------|------------|------------|----------|------------|
| Exhaustive | 0.10 | 65.15 | 1.0000 | 0.000000 |
| Annoy-5 | 3.42 | 7.92 | 0.6439 | 0.360428 |
| Annoy-10 | 5.17 | 16.45 | 0.8462 | 0.095916 |
| Annoy-25 | 11.69 | 43.87 | 0.9835 | 0.005743 |
| Annoy-50 | 18.88 | 85.08 | 0.9993 | 0.000164 |
| Annoy-100 | 36.05 | 215.27 | 1.0000 | 0.000001 |
| HNSW-M16-ef100-s50 | 132.56 | 26.57 | 1.0000 | 0.000000 |
| HNSW-M16-ef100-s100 | 121.33 | 78.81 | 1.0000 | 0.000000 |
| HNSW-M16-ef200-s100 | 190.20 | 74.99 | 1.0000 | 0.000000 |
| HNSW-M16-ef200-s200 | 189.60 | 229.11 | 1.0000 | 0.000000 |
| HNSW-M32-ef200-s100 | 416.97 | 71.33 | 1.0000 | 0.000000 |
| HNSW-M32-ef200-s200 | 418.46 | 238.91 | 1.0000 | 0.000000 |
| NNDescent-i10-r0.5 | 153.64 | 0.00 | 0.9981 | 0.000457 |
| NNDescent-i25-r0.5 | 187.17 | 0.00 | 0.9984 | 0.000378 |
| NNDescent-i25-r1 | 146.04 | 0.00 | 0.9984 | 0.000369 |
| FANNG-c100-s10 | 1364.58 | 9.41 | 0.6962 | 0.731422 |
| FANNG-c200-s20 | 1380.16 | 14.65 | 0.9991 | 0.001719 |
| FANNG-c500-s20 | 1363.54 | 20.83 | 0.9999 | 0.000394 |
| FANNG-c1000-s20 | 1356.45 | 21.50 | 0.9999 | 0.000394 |

### 100k samples, 20 distinct clusters, 16 dimensions

Medium complexity data set with more dimensions needed. For Annoy
the search budget is set to `n_tree * 2`. The number of random entry points
in FANNG is set to `25`.

| Method | Build (ms) | Query (ms) | Recall@k | Dist Error |
|--------|------------|------------|----------|------------|
| Exhaustive | 1.56 | 7862.67 | 1.0000 | 0.000000 |
| Annoy-5 | 51.25 | 157.99 | 0.2583 | 3.234698 |
| Annoy-10 | 64.84 | 382.32 | 0.4042 | 1.675017 |
| Annoy-25 | 155.14 | 1034.79 | 0.6772 | 0.502793 |
| Annoy-50 | 296.94 | 2368.09 | 0.8704 | 0.131538 |
| Annoy-100 | 549.23 | 5326.01 | 0.9738 | 0.017791 |
| HNSW-M16-ef100-s50 | 2106.30 | 578.04 | 0.9967 | 0.002410 |
| HNSW-M16-ef100-s100 | 2100.62 | 1696.65 | 0.9994 | 0.000439 |
| HNSW-M16-ef200-s100 | 3715.10 | 1667.15 | 0.9999 | 0.000071 |
| HNSW-M16-ef200-s200 | 3740.89 | 5359.63 | 1.0000 | 0.000005 |
| HNSW-M32-ef200-s100 | 7750.63 | 1719.12 | 1.0000 | 0.000004 |
| HNSW-M32-ef200-s200 | 7745.72 | 5650.26 | 1.0000 | 0.000000 |
| NNDescent-i10-r0.5 | 3331.30 | 0.00 | 0.9557 | 0.032429 |
| NNDescent-i25-r0.5 | 4757.38 | 0.00 | 0.9601 | 0.028696 |
| NNDescent-i25-r1 | 4769.00 | 0.00 | 0.9603 | 0.028480 |
| FANNG-c100-s20 | 24872.01 | 98.24 | 0.1399 | 6.873163 |
| FANNG-c200-s20 | 25332.87 | 195.48 | 0.7667 | 0.812314 |
| FANNG-c500-s20 | 24517.77 | 342.27 | 0.9534 | 0.044355 |
| FANNG-c1000-s20 | 24468.93 | 333.81 | 0.9545 | 0.042144 |

### 100k samples, 20 distinct clusters, 32 dimensions

Medium complexity data set with more dimensions needed. For Annoy
the search budget is set to `n_tree * 4`! The number of random entry points
in FANNG is set to `50`!

| Method | Build (ms) | Query (ms) | Recall@k | Dist Error |
|--------|------------|------------|----------|------------|
| Exhaustive | 2.58 | 17273.71 | 1.0000 | 0.000000 |
| Annoy-5 | 65.31 | 193.86 | 0.1260 | 13.152898 |
| Annoy-10 | 87.23 | 419.44 | 0.1809 | 8.786201 |
| Annoy-25 | 200.44 | 1247.87 | 0.3234 | 2.278832 |
| Annoy-50 | 359.95 | 2723.41 | 0.5010 | 0.874434 |
| Annoy-100 | 719.03 | 6332.21 | 0.7186 | 1.084700 |
| HNSW-M16-ef100-s50 | 3810.34 | 1075.20 | 0.9169 | 0.211469 |
| HNSW-M16-ef100-s100 | 3785.46 | 2876.53 | 0.9610 | 0.089046 |
| HNSW-M16-ef200-s100 | 6179.53 | 2871.37 | 0.9829 | 0.033648 |
| HNSW-M16-ef200-s200 | 6179.46 | 9565.31 | 0.9952 | 0.009195 |
| HNSW-M32-ef200-s100 | 18107.29 | 3361.91 | 0.9985 | 0.002412 |
| HNSW-M32-ef200-s200 | 19025.60 | 10221.43 | 0.9998 | 0.000352 |
| NNDescent-i10-r0.5 | 4157.08 | 0.00 | 0.7353 | 0.824006 |
| NNDescent-i25-r0.5 | 5806.16 | 0.00 | 0.7536 | 0.743641 |
| NNDescent-i25-r1 | 5979.18 | 0.00 | 0.7551 | 0.737854 |
| FANNG-c100-s20 | 50952.05 | 118.75 | 0.0648 | 18.678117 |
| FANNG-c200-s20 | 49876.50 | 191.53 | 0.3402 | 7.189320 |
| FANNG-c500-s20 | 49441.04 | 434.05 | 0.6655 | 1.470630 |
| FANNG-c1000-s20 | 48231.39 | 461.61 | 0.7007 | 1.134575 |

### Observations



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