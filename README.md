# ann-search-rs

Various approximate nearest neighbour searches implemented in Rust.

## Features

- **Multiple ANN algorithms**:
  - Annoy (Approximate Nearest Neighbours Oh Yeah)
  - HNSW (Hierarchical Navigable Small World)
  - NNDescent (Nearest Neighbour Descent)

- **Distance metrics**:
  - Euclidean
  - Cosine
  - (more to come maybe ... ?)

- **High performance**: Optimised implementations with SIMD-friendly code

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
ann-search-rs = "0.1.0"
```

## Usage
```rust
use ann_search_rs::{build_hnsw_index, query_hnsw_index, Dist, parse_ann_dist};
use faer::Mat;

// Build index
let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
let index = build_hnsw_index(data.as_ref(), "euclidean", 42);

// Query index
let query = Mat::from_fn(10, 128, |_, _| rand::random::<f32>());
let (indices, distances) = query_hnsw_index(
    query.as_ref(),
    &index,
    "euclidean",
    10,  // k neighbours
    true,  // return distances
    false  // verbose
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