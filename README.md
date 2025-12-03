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
  some parallel operations in the index generation for speed purposes. This
  version is still being actively developed and optimised.

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
  100,            // ef_construction
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
  200,            // ef_search
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
  "euclidean".    // distance metric
  100,            // number of trees to generate
  42,             // seed
);

// Query the Annoy index
let query = Mat::from_fn(10, 128, |_, _| rand::random::<f32>());
let (annoy_indices, annoy_dists) = query_annoy_index(
  mat.as_ref(), 
  15,             // k
  None,           // search budget will use the default
  true,           // return distances
  false.          // verbosity
);
```

## Performance and recommendations

Different indices show in parts very different index build times and querying 
times. Below are some of the results observed (please check the examples to run
them). You can do so with

```rust
// This would run the smallest benchmark
cargo run --example benchmarks_50k_16dim --release
```

All of the benchmarks below were run with k = 15 and Euclidean distances. The
exhaustive search serves as baseline in terms of time and ground truth. Below
one can appreciate the trade-offs of index generation speed to querying speed
and `Recall@k` against the bruteforce ground truth.

### 50k samples, 20 distinct clusters, 16 dimensions

This could represent a smaller, low complexity single cell result (co-culture
for example with only few cell types present). The search budget for Annoy was 
set to `k * n_trees * 20` which is the default in this library. You can control 
this also directly during querying. As one can appreciated, the exhaustive seach 
in this case actually fast and the overhead introduced by some of the indices do 
not warrant (yet) the approximate searches. LSH being an exception here with
faster total time and decent enough `Recall@K`. 

```
========================================================================================================================
Benchmark: 50k cells, 16D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          1.08         1384.30         1385.38          1.0000        0.000000
Annoy-nt5                                          17.04           96.38          121.89          0.6910        0.529103
Annoy-nt10                                         22.86          188.24          219.02          0.8808        0.140608
Annoy-nt15                                         35.02          290.72          333.18          0.9512        0.047128
Annoy-nt25                                         53.14          554.12          614.66          0.9903        0.007390
Annoy-nt50                                        101.60         1254.64         1363.37          0.9997        0.000174
Annoy-nt100                                       196.92         2919.86         3123.92          1.0000        0.000001
HNSW-M16-ef100-s50                                843.82          222.33         1073.30          0.9987        0.000971
HNSW-M16-ef100-s100                               843.01          716.63         1566.80          0.9998        0.000161
HNSW-M16-ef200-s100                              1579.17          713.00         2299.37          1.0000        0.000028
HNSW-M16-ef200-s200                              1535.44         2371.54         3914.20          1.0000        0.000004
HNSW-M32-ef200-s100                              3088.59          703.10         3798.82          1.0000        0.000000
HNSW-M32-ef200-s200                              3132.10         2430.59         5570.06          1.0000        0.000000
NNDescent-nt12-s:auto-dp0                        1834.40           79.46         1921.36          0.9999        0.000075
NNDescent-nt24-s:auto-dp0                        1556.86           95.08         1659.59          0.9999        0.000036
NNDescent-nt:auto-s50-dp0                        1579.94          183.88         1771.16          0.9999        0.000022
NNDescent-nt:auto-s100-dp0                       1546.71          369.79         1923.72          1.0000        0.000004
NNDescent-nt:auto-s:auto-dp0                     1542.67           79.58         1629.61          0.9999        0.000043
NNDescent-nt:auto-s:auto-dp5                     1553.46           69.61         1630.31          0.9935        0.003909
NNDescent-nt:auto-s:auto-dp1                     1577.18           65.98         1650.61          0.9802        0.013460
LSH-nt20-bits8-candauto                            26.28          396.01          403.90          0.9023        0.114650
LSH-nt50-bits8-candauto                            53.47         1073.57         1080.83          0.9935        0.004933
LSH-nt100-bits8-candauto                          109.85         2118.77         2126.04          0.9998        0.000101
LSH-nt100-bits8-cand500*k                         104.14          699.85          707.43          0.9702        0.027213
LSH-nt20-bits10-candauto                           33.23          183.84          191.87          0.7511        0.386876
LSH-nt50-bits10-candauto                           72.11          449.22          456.75          0.9502        0.047295
LSH-nt100-bits10-candauto                         136.97          894.39          901.62          0.9936        0.004388
LSH-nt100-bits10-cand500*k                        134.38          839.36          846.59          0.9927        0.005078
LSH-nt20-bits12-candauto                           44.71          138.02          146.37          0.5737        0.936820
LSH-nt50-bits12-candauto                          112.53          340.06          348.06          0.8325        0.212174
LSH-nt100-bits12-candauto                         169.86          626.63          634.03          0.9540        0.040353
LSH-nt100-bits12-cand500*k                        176.61          635.10          642.59          0.9540        0.040353
LSH-nt20-bits16-candauto                           55.43          130.17          138.58          0.3030        3.530929
LSH-nt50-bits16-candauto                          118.12          299.11          307.28          0.5033        1.313877
LSH-nt100-bits16-candauto                         224.31          589.36          597.41          0.6993        0.502600
LSH-nt100-bits12-cand500*k                        173.70          642.93          650.38          0.9540        0.040353
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
Exhaustive                                          2.11        10605.66        10607.78          1.0000        0.000000
Annoy-nt5                                          47.96          391.29          460.78          0.6205        0.615032
Annoy-nt10                                         67.15          906.26          994.17          0.8249        0.193175
Annoy-nt15                                        101.47         1324.27         1445.39          0.9149        0.075544
Annoy-nt25                                        164.33         2897.72         3081.51          0.9775        0.015284
Annoy-nt50                                        314.84         5789.30         6122.86          0.9987        0.000619
Annoy-nt100                                       594.55        12796.15        13409.40          1.0000        0.000003
HNSW-M16-ef100-s50                               2522.09          863.78         3404.34          0.9976        0.001671
HNSW-M16-ef100-s100                              2572.20         2414.77         5005.59          0.9996        0.000261
HNSW-M16-ef200-s100                              4818.40         2415.37         7252.49          0.9999        0.000050
HNSW-M16-ef200-s200                              4797.08         7797.52        12613.11          1.0000        0.000015
HNSW-M32-ef200-s100                             11013.46         2735.21        13767.31          1.0000        0.000003
HNSW-M32-ef200-s200                              9561.14         8450.38        18030.49          1.0000        0.000000
NNDescent-nt12-s:auto-dp0                        6132.12          316.33         6467.76          0.9997        0.000190
NNDescent-nt24-s:auto-dp0                        5052.28          298.68         5370.07          0.9998        0.000111
NNDescent-nt:auto-s50-dp0                        4987.31          683.72         5690.20          0.9999        0.000069
NNDescent-nt:auto-s100-dp0                       5097.74         1611.50         6728.68          1.0000        0.000014
NNDescent-nt:auto-s:auto-dp0                     5262.47          295.12         5576.62          0.9998        0.000110
NNDescent-nt:auto-s:auto-dp5                     5048.57          273.75         5340.90          0.9934        0.003368
NNDescent-nt:auto-s:auto-dp1                     5121.47          245.82         5386.13          0.9806        0.010937
------------------------------------------------------------------------------------------------------------------------
```

### 150k samples, 20 distinct clusters, 32 dimensions

With increase of dimensionality, we can appreciate that some of the methods
start performing worse here. To note is that the synthetic data does not have
the most structure and likely the curse of dimensionality starts hitting here 
with the distances between points becoming smaller and smaller. In real data,
it is likely that you get comparable results to the 16 dimension case.

```
========================================================================================================================
Benchmark: 150k cells, 32D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          3.58        25537.45        25541.05          1.0000        0.000000
Annoy-nt5                                          79.88          835.85          941.94          0.3406        3.964605
Annoy-nt10                                        117.47         1686.52         1827.16          0.5051        2.124384
Annoy-nt15                                        200.75         2561.22         2783.92          0.6250        1.317138
Annoy-nt25                                        236.20         4431.42         4688.70          0.7786        0.608047
Annoy-nt50                                        449.16         9066.08         9535.61          0.9329        0.134196
Annoy-nt100                                       889.61        20282.40        21192.37          0.9916        0.012543
HNSW-M16-ef100-s50                               4464.44         1835.14         6320.40          0.9158        0.211107
HNSW-M16-ef100-s100                              4274.15         3994.12         8289.03          0.9591        0.091741
HNSW-M16-ef200-s100                              7698.71         3893.85        11613.68          0.9829        0.033833
HNSW-M16-ef200-s200                              7651.99        12421.97        20094.37          0.9947        0.010152
HNSW-M32-ef200-s100                             19441.81         4799.65        24261.89          0.9985        0.002314
HNSW-M32-ef200-s200                             19402.40        13332.85        32756.73          0.9998        0.000383
NNDescent-nt12-s:auto-dp0                        8113.14          484.76         8619.54          0.9888        0.019162
NNDescent-nt24-s:auto-dp0                        7645.77          476.40         8143.04          0.9901        0.016592
NNDescent-nt:auto-s50-dp0                        7737.22         1095.19         8854.85          0.9926        0.012222
NNDescent-nt:auto-s100-dp0                       7950.40         2464.86        10435.01          0.9963        0.005830
NNDescent-nt:auto-s:auto-dp0                     7696.00          475.05         8193.02          0.9902        0.016451
NNDescent-nt:auto-s:auto-dp5                     7947.59          441.83         8410.20          0.9754        0.035481
NNDescent-nt:auto-s:auto-dp1                     7907.91          421.40         8350.77          0.9566        0.065492
------------------------------------------------------------------------------------------------------------------------
```

### 500k samples, 20 distinct clusters, 16 dimensions

With half a million samples, we are starting to approach the point where the
approximate nearest neighbour searches and their indices really shine. Annoy 
with only 25 trees yields a recall of ≥0.95 while being nearly 8 times faster. 
HNSW with m = 16, construction budget of 100 and search budget yields recalls of 
≥0.99 while being 6 times faster. The NNDescent index can query 500k cells in
1.3 seconds which makes with a Recall of ≥0.99 making it perfect for repeat 
query situations (the fastest of all of them...).

```
========================================================================================================================
Benchmark: 500k cells, 16D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                          5.33       126361.75       126367.09          1.0000        0.000000
Annoy-nt5                                         201.38         1911.70         2189.58          0.5583        0.665622
Annoy-nt10                                        496.31         3957.17         4528.13          0.7670        0.237472
Annoy-nt15                                        444.59         7845.68         8365.09          0.8725        0.103763
Annoy-nt25                                        703.89        16091.30        16870.21          0.9583        0.025873
Annoy-nt50                                       1380.29        29163.84        30614.75          0.9963        0.001585
Annoy-nt100                                      2612.96        56457.11        59146.22          0.9999        0.000020
HNSW-M16-ef100-s50                              10181.98         3962.08        14214.04          0.9959        0.002304
HNSW-M16-ef100-s100                             10941.74        10808.54        21820.58          0.9992        0.000446
HNSW-M16-ef200-s100                             19808.71        10060.25        29937.25          0.9998        0.000094
HNSW-M16-ef200-s200                             20197.27        30133.18        50402.54          1.0000        0.000010
HNSW-M32-ef200-s100                             37082.05        11697.45        48848.24          1.0000        0.000007
HNSW-M32-ef200-s200                             37907.39        34205.35        72183.05          1.0000        0.000000
NNDescent-nt12-s:auto-dp0                       25029.66         1437.66        26545.26          0.9993        0.000330
NNDescent-nt24-s:auto-dp0                       23195.91         1471.51        24738.05          0.9995        0.000203
NNDescent-nt:auto-s50-dp0                       21590.28         3194.08        24854.19          0.9997        0.000098
NNDescent-nt:auto-s100-dp0                      22624.40         6052.20        28748.84          0.9999        0.000025
NNDescent-nt:auto-s:auto-dp0                    22737.14         1462.70        24271.19          0.9996        0.000167
NNDescent-nt:auto-s:auto-dp5                    22783.09         1347.97        24199.54          0.9934        0.002722
NNDescent-nt:auto-s:auto-dp1                    22749.89         1241.97        24062.96          0.9811        0.008758
------------------------------------------------------------------------------------------------------------------------
```

### 500k samples, 20 distinct clusters, 32 dimensions

With 32 dimensions, the observe (similar to the 150k sample case) a lower Recall
for most of the approximate methods; however, we observe way faster total times.

```
========================================================================================================================
Benchmark: 500k cells, 32D
========================================================================================================================
Method                                        Build (ms)      Query (ms)      Total (ms)        Recall@k      Dist Error
------------------------------------------------------------------------------------------------------------------------
Exhaustive                                         11.35       265557.34       265568.70          1.0000        0.000000
Annoy-nt5                                         294.68         3374.96         3751.03          0.2719        4.671276
Annoy-nt10                                        363.38         6880.77         7323.21          0.4032        2.748679
Annoy-nt15                                        632.57        11549.71        12260.28          0.5087        1.847431
Annoy-nt25                                        972.74        24784.11        25834.80          0.6611        0.985923
Annoy-nt50                                       1811.55        47893.26        49782.18          0.8561        0.298757
Annoy-nt100                                      3506.50        92872.64        96454.35          0.9678        0.048905
HNSW-M16-ef100-s50                              15398.46         6665.89        22138.81          0.8519        0.380373
HNSW-M16-ef100-s100                             15389.71        16653.15        32116.07          0.9134        0.194674
HNSW-M16-ef200-s100                             28183.15        16521.26        44776.38          0.9586        0.080924
HNSW-M16-ef200-s200                             28545.96        47679.65        76298.81          0.9830        0.031352
HNSW-M32-ef200-s100                             74946.94        22880.52        97898.26          0.9952        0.006908
HNSW-M32-ef200-s200                             75326.57        60464.15       135865.04          0.9990        0.001403
NNDescent-nt12-s:auto-dp0                       38998.73         2338.85        41420.46          0.9768        0.037383
NNDescent-nt24-s:auto-dp0                       37500.62         2346.83        39933.30          0.9793        0.032350
NNDescent-nt:auto-s50-dp0                       38157.88         4934.82        43176.62          0.9844        0.023504
NNDescent-nt:auto-s100-dp0                      36913.02         9528.57        46524.05          0.9910        0.013109
NNDescent-nt:auto-s:auto-dp0                    37277.52         2255.89        39616.89          0.9807        0.029912
NNDescent-nt:auto-s:auto-dp5                    38301.42         2105.58        40488.98          0.9668        0.046885
NNDescent-nt:auto-s:auto-dp1                    39716.33         2047.37        41847.42          0.9496        0.072063
------------------------------------------------------------------------------------------------------------------------
```

### 2m samples, 20 distinct clusters, 32 dimensions

This would be a case of a very large single cell data set. In this case, the
exhaustive search is becoming VERY slow and not recommended anymore. The 
approximate nearest neighbour searches are truly shining here.

```
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