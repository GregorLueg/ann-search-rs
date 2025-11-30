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
the search budget is set to `n_tree * 3`. The number of random entry points
in FANNG is set to `25`.

```
================================================================================
Benchmark: 10k cells, 8D
================================================================================
Method            Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------
Exhaustive              0.09        53.42        53.53       1.0000     0.000000
Annoy-5                 3.25         8.51        13.99       0.7230     0.238923
Annoy-10                4.59        17.62        24.32       0.9037     0.052017
Annoy-25                9.81        47.60        59.40       0.9940     0.001923
Annoy-50               18.92       114.09       135.10       0.9999     0.000017
Annoy-100              35.72       268.06       305.79       1.0000     0.000000
HNSW-M16-ef100-s50       122.31        24.54       148.98       1.0000     0.000000
HNSW-M16-ef100-s100       117.80        68.23       188.05       1.0000     0.000000
HNSW-M16-ef200-s100       181.14        68.31       251.52       1.0000     0.000000
HNSW-M16-ef200-s200       173.91       212.09       387.99       1.0000     0.000000
HNSW-M32-ef200-s100       380.94        64.86       447.84       1.0000     0.000000
HNSW-M32-ef200-s200       366.49       226.16       594.59       1.0000     0.000000
NNDescent-i10-r0.5       135.29         0.00       137.22       0.9981     0.000457
NNDescent-i25-r0.5       195.74         0.00       197.68       0.9984     0.000378
NNDescent-i25-r1       130.51         0.00       132.49       0.9984     0.000369
FANNG-c100-s20       1604.84         7.94      1614.96       0.6883     0.691898
FANNG-c200-s20       1565.52        13.29      1580.85       0.9992     0.001605
FANNG-c500-s20       1553.77        19.69      1575.40       0.9999     0.000327
FANNG-c1000-s20      1538.00        19.24      1559.24       0.9999     0.000327
--------------------------------------------------------------------------------
```

With low amounts of samples, the exhaustive search is actually the fastest.

### 100k samples, 20 distinct clusters, 16 dimensions

This could represent a smaller, low complexity single cell result. For Annoy
the search budget is set to `n_tree * 4`. The number of random entry points
in FANNG is set to `25`.

```
================================================================================
Benchmark: 100k cells, 16D
================================================================================
Method            Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------
Exhaustive              1.33      7010.78      7012.11       1.0000     0.000000
Annoy-5                42.89       195.91       261.75       0.3022     2.624925
Annoy-10               61.31       381.19       463.83       0.4702     1.282308
Annoy-25              142.47      1204.25      1367.55       0.7540     0.332057
Annoy-50              269.79      2758.16      3048.73       0.9212     0.070282
Annoy-100             514.36      6866.85      7403.22       0.9894     0.006388
HNSW-M16-ef100-s50      2093.57       554.63      2668.53       0.9966     0.002426
HNSW-M16-ef100-s100      1905.30      1518.20      3442.69       0.9995     0.000385
HNSW-M16-ef200-s100      3352.79      1738.16      5110.83       0.9999     0.000077
HNSW-M16-ef200-s200      3416.25      5103.71      8539.54       1.0000     0.000008
HNSW-M32-ef200-s100      7034.83      1585.24      8639.85       1.0000     0.000006
HNSW-M32-ef200-s200      7408.99      5056.07     12484.40       1.0000     0.000000
NNDescent-i10-r0.5      3015.45         0.00      3036.29       0.9557     0.032429
NNDescent-i25-r0.5      4648.03         0.00      4668.06       0.9601     0.028696
NNDescent-i25-r1      4634.53         0.00      4654.76       0.9603     0.028480
FANNG-c100-s20      26482.05        84.44     26589.99       0.1472     6.694286
FANNG-c200-s20      25840.38       145.22     26006.08       0.8138     0.691738
FANNG-c500-s20      25766.17       289.82     26075.90       0.9740     0.022659
FANNG-c1000-s20     25544.78       297.60     25863.04       0.9746     0.021176
--------------------------------------------------------------------------------
```

This is the point where we can observe the non-exhausive searches being faster
than the exhaustive search. There is a clear trade-off between index building
and querying time (usually you do both in one go in single cell). Especially
interesting here is FANNG. It has the longest index building time, but the 
query time is incredibly rapid. This can be useful in situations in which you
query the same data set again and again. HNSW is an obvious winner here with an
overall low combined index building time and query time. 

### 100k samples, 20 distinct clusters, 32 dimensions

Medium complexity data set with more dimensions needed. For Annoy
the search budget is set to `n_tree * 5`! The number of random entry points
in FANNG is set to `50`! The number of max iterations for NNDescent was 
increased to `50`.

```
================================================================================
Benchmark: 100k cells, 32D
================================================================================
Method            Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------
Exhaustive              2.76     14942.45     14945.22       1.0000     0.000000
Annoy-5                57.78       191.70       273.54       0.1272    13.014309
Annoy-10               71.35       426.70       520.05       0.1830     8.676938
Annoy-25              180.53      1257.21      1459.98       0.3276     4.440112
Annoy-50              349.63      2748.94      3121.66       0.5069     2.227598
Annoy-100             687.98      6353.03      7063.21       0.7249     0.846346
HNSW-M16-ef100-s50      3360.45      1005.46      4387.96       0.9172     0.212242
HNSW-M16-ef100-s100      3447.88      2595.75      6063.95       0.9610     0.089048
HNSW-M16-ef200-s100      5417.60      2553.00      7990.68       0.9829     0.033547
HNSW-M16-ef200-s200      5361.82      7728.92     13110.63       0.9952     0.009089
HNSW-M32-ef200-s100     15842.33      3120.11     18982.22       0.9985     0.002481
HNSW-M32-ef200-s200     15690.35      8830.62     24541.25       0.9998     0.000319
NNDescent-i10-r0.5      3839.63         0.00      3861.47       0.7353     0.824006
NNDescent-i25-r0.5      4901.87         0.00      4923.42       0.7536     0.743641
NNDescent-i25-r1      5001.44         0.00      5022.89       0.7551     0.737854
NNDescent-i50-r1      6850.52         0.00      6871.68       0.7574     0.728158
FANNG-c100-s20      55213.18        97.61     55333.11       0.0836    17.789756
FANNG-c200-s20      55087.03       205.13     55315.21       0.4968     5.433169
FANNG-c500-s20      57164.34       386.95     57572.76       0.8209     0.636383
FANNG-c1000-s20     58639.96       456.36     59118.27       0.8431     0.462000
--------------------------------------------------------------------------------
```

### 500k samples, 20 distinct clusters, 16 dimensions

This is a larger data set where exhaustive search clearly becomes prohibitively
expensive and the approximate methods shine. For Annoy the search budget is set 
to `n_tree * 5`! The number of random entry points in FANNG is set to `50`! The 
number of max iterations for NNDescent was increased to `50`.

```
```

### 500k samples, 20 distinct clusters, 32 dimensions

This is a larger data set where exhaustive search clearly becomes prohibitively
expensive and the approximate methods shine. For Annoy the search budget is set 
to `n_tree * 5`! The number of random entry points in FANNG is set to `50`! The 
number of max iterations for NNDescent was increased to `50`.

```

```

### Observations

The main focus of this library is to support single cell analysis. `HNSW` shines
here very clearly and has been established as the new default in
[bixverse](https://github.com/GregorLueg/bixverse). An interesting special case
is FANNG. The index generation is by far the longest; however, the query speed
is absurdly high compared to the other methods (particularly with more samples
and higher dimensionality). This indicates that this method could be particularly
interesting in cases where an index is generated ONCE and then it is queried
repeatedly.

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