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

### 50k samples, 20 distinct clusters, 16 dimensions

This could represent a smaller, low complexity single cell result. The search
budget for Annoy was set to `k * n_trees * 20` which is the default in this
library. You can control this also directly during querying. As one can 
appreciated, the exhaustive seach in this case actually fast and the overhead
introduced by some of the indices do not warrant (yet) the approximate searches.

```
========================================================================================================================
Benchmark: 50k cells, 16D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          0.98         1272.55         1273.53          1.0000        0.000000
Annoy-nt5                                          14.58           95.07          116.80          0.6910        0.529103
Annoy-nt10                                         19.66          167.54          194.05          0.8808        0.140608
Annoy-nt15                                         29.69          254.61          290.71          0.9512        0.047128
Annoy-nt25                                         47.62          494.40          548.33          0.9903        0.007390
Annoy-nt50                                         90.95         1133.85         1231.11          0.9997        0.000174
Annoy-nt100                                       173.24         2671.94         2851.36          1.0000        0.000001
HNSW-M16-ef100-s50                                760.22          201.23          967.64          0.9987        0.001071
HNSW-M16-ef100-s100                               757.24          631.50         1394.91          0.9998        0.000160
HNSW-M16-ef200-s100                              1392.59          630.04         2028.83          1.0000        0.000024
HNSW-M16-ef200-s200                              1383.44         2119.38         3509.03          1.0000        0.000000
HNSW-M32-ef200-s100                              2745.44          612.98         3364.59          1.0000        0.000002
HNSW-M32-ef200-s200                              2766.78         2123.93         4896.89          1.0000        0.000000
NNDescent-i10-c80-d0-ef100-conv:true              986.49          234.40         1227.25          0.9892        0.064742
NNDescent-i10-c100-d0-ef100-conv:true            1016.13          241.44         1263.87          0.9892        0.064742
NNDescent-i15-c100-d0-ef150-conv:true             977.28          331.31         1314.84          0.9934        0.047227
NNDescent-i20-c120-d0-ef200-conv:true             972.30          433.60         1412.14          0.9951        0.039946
NNDescent-i15-c100-d5-ef150-conv:true             978.53          334.37         1319.21          0.9877        0.058823
NNDescent-i20-c120-d5-ef200-conv:true             987.11          422.51         1415.93          0.9912        0.048542
NNDescent-i15-cauto-d0-ef150-conv:true            990.23          333.28         1329.87          0.9934        0.047227
NNDescent-i20-cauto-d0-ef200-conv:true            987.06          434.39         1427.69          0.9951        0.039946
------------------------------------------------------------------------------------------------------------------------
```

### 150k samples, 20 distinct clusters, 16 dimensions

This would be larger experiment with more cells/samples. In this case, we can
start seeing that the approximate methods start out-performing the brute force
approach with minimal loss in Recall.

```
========================================================================================================================
Benchmark: 150k cells, 16D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          1.92        11134.92        11136.84          1.0000        0.000000
Annoy-nt5                                          50.63          438.25          512.26          0.6205        0.615032
Annoy-nt10                                         75.46          935.60         1033.22          0.8249        0.193175
Annoy-nt15                                        115.82         1411.17         1548.61          0.9149        0.075544
Annoy-nt25                                        181.86         3515.82         3720.26          0.9775        0.015284
Annoy-nt50                                        409.70         6911.92         7342.32          0.9987        0.000619
Annoy-nt100                                       667.83        13678.17        14368.41          1.0000        0.000003
HNSW-M16-ef100-s50                               3780.39          943.59         4743.97          0.9976        0.001642
HNSW-M16-ef100-s100                              2672.94         2527.38         5220.02          0.9996        0.000304
HNSW-M16-ef200-s100                              4951.86         2494.69         7465.26          0.9999        0.000044
HNSW-M16-ef200-s200                              4900.15         8274.13        13194.70          1.0000        0.000015
HNSW-M32-ef200-s100                             10338.53         2811.32        13169.06          1.0000        0.000001
HNSW-M32-ef200-s200                             10590.15         9148.38        19758.62          1.0000        0.000000
NNDescent-i10-c80-d0-ef100-conv:true             6528.89         1070.13         7619.14          0.9888        0.057749
NNDescent-i10-c100-d0-ef100-conv:true            6681.78         1128.32         7829.34          0.9888        0.057749
NNDescent-i15-c100-d0-ef150-conv:true            6848.49         1495.28         8364.19          0.9933        0.041187
NNDescent-i20-c120-d0-ef200-conv:true            6549.09         2186.28         8755.06          0.9950        0.034261
NNDescent-i15-c100-d5-ef150-conv:true            7182.40         1427.71         8630.21          0.9877        0.050916
NNDescent-i20-c120-d5-ef200-conv:true            7137.99         1855.50         9016.30          0.9911        0.041659
NNDescent-i15-cauto-d0-ef150-conv:true           6928.29         1397.42         8344.62          0.9933        0.041187
NNDescent-i20-cauto-d0-ef200-conv:true           6663.95         1922.29         8605.54          0.9950        0.034261
------------------------------------------------------------------------------------------------------------------------
```

### 150k samples, 20 distinct clusters, 32 dimensions

```
========================================================================================================================
Benchmark: 150k cells, 32D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          3.78        23420.31        23424.10          1.0000        0.000000
Annoy-nt5                                          73.79          777.41          874.94          0.3406        3.964605
Annoy-nt10                                         99.83         1568.52         1689.88          0.5051        2.124384
Annoy-nt15                                        154.35         2461.31         2636.76          0.6250        1.317138
Annoy-nt25                                        241.50         4457.80         4720.33          0.7786        0.608047
Annoy-nt50                                        459.23         9393.67         9874.55          0.9329        0.134196
Annoy-nt100                                       917.44        20522.18        21459.07          0.9916        0.012543
HNSW-M16-ef100-s50                               4098.78         1536.38         5655.24          0.9155        0.211173
HNSW-M16-ef100-s100                              4094.51         3831.46         7946.62          0.9591        0.090794
HNSW-M16-ef200-s100                              7273.89         3839.24        11132.38          0.9829        0.032882
HNSW-M16-ef200-s200                              7275.73        11843.39        19138.13          0.9946        0.010489
HNSW-M32-ef200-s100                             18745.93         4516.78        23281.53          0.9985        0.002162
HNSW-M32-ef200-s200                             19008.54        13240.48        32268.02          0.9998        0.000376
NNDescent-i10-c80-d0-ef100                      10897.06         1215.50        12132.29          0.8645        1.309344
NNDescent-i10-c100-d0-ef100                     11169.37         1212.53        12402.27          0.8645        1.309892
NNDescent-i15-c100-d0-ef150                     11515.97         1765.52        13301.34          0.9084        0.913948
NNDescent-i20-c120-d0-ef200                     11282.29         2296.79        13598.78          0.9312        0.720815
NNDescent-i15-c100-d5-ef150                     11668.65         1735.10        13424.61          0.8890        1.023601
NNDescent-i20-c120-d5-ef200                     11284.43         2229.03        13532.99          0.9153        0.807323
NNDescent-i15-cauto-d0-ef150                    11677.55         1800.31        13497.88          0.9084        0.913948
NNDescent-i20-cauto-d0-ef200                    11289.27         2282.37        13592.39          0.9311        0.721020
------------------------------------------------------------------------------------------------------------------------
```

### 500k samples, 20 distinct clusters, 16 dimensions

With half a million samples, we are starting to approach the point where the
approximate indices really shine. Annoy with 25 trees yields a recall of ≥0.95
while being nearly 8 times faster. HNSW with m = 16, construction budget of 100
and search budget yields recalls of ≥0.99 while being 6 times faster.

```
========================================================================================================================
Benchmark: 500k cells, 16D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          5.22       123552.78       123558.01          1.0000        0.000000
Annoy-nt5                                         201.52         1931.08         2210.19          0.5583        0.665622
Annoy-nt10                                        283.31         4149.22         4504.57          0.7670        0.237472
Annoy-nt15                                        441.38         6931.79         7446.15          0.8725        0.103763
Annoy-nt25                                        685.97        15288.35        16045.18          0.9583        0.025873
Annoy-nt50                                       1297.49        34697.22        36063.85          0.9963        0.001585
Annoy-nt100                                      2605.33        62778.51        65455.00          0.9999        0.000020
HNSW-M16-ef100-s50                              10200.22         3877.98        14145.35          0.9959        0.002373
HNSW-M16-ef100-s100                             10036.96         9942.22        20046.84          0.9992        0.000430
HNSW-M16-ef200-s100                             19274.28         9917.03        29258.68          0.9998        0.000080
HNSW-M16-ef200-s200                             19125.83        29491.29        48686.38          1.0000        0.000010
HNSW-M32-ef200-s100                             36087.47        11189.44        47343.27          1.0000        0.000005
HNSW-M32-ef200-s200                             36057.10        32828.42        68953.52          1.0000        0.000000
NNDescent-i10-c80-d0-ef100                      51473.92         4143.93        55684.26          0.9876        0.051436
NNDescent-i10-c100-d0-ef100                     54967.43         4488.65        59523.22          0.9876        0.051436
NNDescent-i15-c100-d0-ef150                     56795.06         6158.77        63019.95          0.9928        0.035021
NNDescent-i20-c120-d0-ef200                     56464.46         7986.82        64518.64          0.9948        0.028398
NNDescent-i15-c100-d5-ef150                     55888.63         5942.58        61897.80          0.9867        0.043487
NNDescent-i20-c120-d5-ef200                     56262.82         7666.10        63994.42          0.9904        0.034668
NNDescent-i15-cauto-d0-ef150                    56245.77         6171.24        62482.87          0.9928        0.035021
NNDescent-i20-cauto-d0-ef200                    54508.51         7971.11        62546.09          0.9948        0.028398
------------------------------------------------------------------------------------------------------------------------
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