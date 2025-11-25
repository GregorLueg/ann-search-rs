# ann-search-rs

Various approximate nearest neighbour searches implemented in Rust. Helper
library to be used in other libraries. 

## Description

Extracted function for approximate nearest neighbour searches specifically
with single cell in mind from [bixverse](https://github.com/GregorLueg/bixverse),
a R/Rust package designed for computational biology, that has a ton of 
functionality for single cell. Within all of the single cell functions, kNN 
generations are ubiqituos, thus, I want to expose the APIs to other packages.

## Features

- **Multiple ANN algorithms**:
  - [**Annoy (Approximate Nearest Neighbours Oh Yeah)**](https://github.com/spotify/annoy).
  A version is implemented here with some modifications.
  - [**HNSW (Hierarchical Navigable Small World)**](https://arxiv.org/abs/1603.09320). 
  A version with some slight modifications has been implemented in this package.
  - [**NNDescent (Nearest Neighbour Descent)** 
  (heavily inspired by [PyNNDescent](https://github.com/lmcinnes/pynndescent))].

- **Distance metrics**:
  - Euclidean
  - Cosine
  - More to come maybe...

- **High performance**: Optimised implementations with SIMD-friendly code,
heavy threading were possible and optimised structures for memory access.

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

## Usage

Below shows an example on how to use for example the HNSW index and query it.

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