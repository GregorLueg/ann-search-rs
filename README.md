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
  - [**LSH (Locality Sensitive Hashing)**](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) An
  approximate nearest neighbour search that can be very fast at the cost of
  precision. 
  - [**IVF (Inverted File index)**] This one leverages k-mean clustering to only
  search a subspace of the original index. There is also a scalar quantised
  version of this index for even higher speed/reduced memory fingerprint at the
  cost of Recall@k_neighbours.
    

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

## Roadmap

Longer term, I am considering GPU-acceleration (yet to be figured out how,
likely via the [Burn framework](https://burn.dev)). I am also considering some
further inspiration from the [Faiss library](https://faiss.ai) and combine 
quantisation methods with IVF vor REALLY large data sets. Let's see.

## Example Usage

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

The package provides a number of different approximate nearest neighbour
searches. The overall design is very similar and if you wish details on usage,
please refer to the `examples/*.rs` section which shows you the grid searches
across various parameters per given index.

## Performance across various parameters

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
# --n-clusters 20
# --k 15
# --seed 10101
# --distance cosine
```

For every index, 150k cells with 32 dimensions distance and 20 distinct clusters 
(of different sizes each) in the synthetic data has been run. The results for 
the different indices are show below. For details on the synthetic data 
function, see `./examples/commons/mod.rs`.

### Annoy

50 to 75 trees are already sufficient to achieve very high Recalls@k, while
being substantially faster than an exhaustive search. The search_budget is 
set as default to `k * n_trees * 20` which is quite a large one. This is 
reflected in `:auto`. Smaller multipliers with `5x` and `10x` are also shown.
Overall, index generation is very fast (highly parallelisable), but the search
budget needs to be quite decent to get good Recall@k (especially with few 
trees). The more trees you use the more you can reduce the search budget per 
given tree. It depends on your use case. Good allrounder. 

**Euclidean:**

Below are the results for the Euclidean distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.55     23334.34     23337.89       1.0000     0.000000
Annoy-nt5:auto                             76.32       850.30       926.62       0.6063    88.642058
Annoy-nt5:10x                              76.32       599.16       675.48       0.4849    88.529759
Annoy-nt5:5x                               76.32       383.37       459.69       0.3466    88.315789
Annoy-nt10:auto                           119.88      1593.12      1713.00       0.8226    88.791833
Annoy-nt10:10x                            119.88      1091.70      1211.58       0.7039    88.727727
Annoy-nt10:5x                             119.88       695.89       815.77       0.5274    88.579254
Annoy-nt15:auto                           175.85      2378.64      2554.50       0.9172    88.837623
Annoy-nt15:10x                            175.85      1783.97      1959.82       0.8257    88.799331
Annoy-nt15:5x                             175.85      1022.95      1198.80       0.6547    88.694040
Annoy-nt25:auto                           251.01      3552.79      3803.80       0.9799    88.861738
Annoy-nt25:10x                            251.01      2537.07      2788.08       0.9355    88.847629
Annoy-nt25:5x                             251.01      1794.35      2045.36       0.8105    88.792808
Annoy-nt50:auto                           471.70      6767.21      7238.91       0.9991    88.867683
Annoy-nt50:10x                            471.70      4670.27      5141.97       0.9930    88.866287
Annoy-nt50:5x                             471.70      3038.40      3510.11       0.9520    88.853961
Annoy-nt75:auto                           694.65      9111.69      9806.33       0.9999    88.867882
Annoy-nt75:10x                            694.65      6872.38      7567.03       0.9990    88.867708
Annoy-nt75:5x                             694.65      4861.24      5555.88       0.9861    88.864541
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.66     24842.81     24847.47       1.0000     0.000000
Annoy-nt5:auto                             76.06       323.74       399.80       0.3263     0.009069
Annoy-nt5:10x                              76.06       333.78       409.84       0.3263     0.009069
Annoy-nt5:5x                               76.06       320.63       396.69       0.3175     0.009340
Annoy-nt10:auto                           101.18       635.84       737.01       0.5007     0.004740
Annoy-nt10:10x                            101.18       606.73       707.91       0.5007     0.004740
Annoy-nt10:5x                             101.18       592.35       693.53       0.4943     0.004846
Annoy-nt15:auto                           156.41       867.01      1023.41       0.6276     0.002870
Annoy-nt15:10x                            156.41       861.91      1018.31       0.6276     0.002870
Annoy-nt15:5x                             156.41       850.43      1006.83       0.6230     0.002924
Annoy-nt25:auto                           249.97      1347.98      1597.95       0.7865     0.001278
Annoy-nt25:10x                            249.97      1326.26      1576.22       0.7865     0.001278
Annoy-nt25:5x                             249.97      1302.26      1552.23       0.7844     0.001296
Annoy-nt50:auto                           463.73      2553.34      3017.08       0.9419     0.000249
Annoy-nt50:10x                            463.73      2648.08      3111.81       0.9419     0.000249
Annoy-nt50:5x                             463.73      2650.64      3114.37       0.9415     0.000251
Annoy-nt75:auto                           689.00      4020.18      4709.19       0.9821     0.000063
Annoy-nt75:10x                            689.00      4082.52      4771.52       0.9821     0.000063
Annoy-nt75:5x                             689.00      4001.34      4690.34       0.9821     0.000063
----------------------------------------------------------------------------------------------------
```

### HNSW

HNSW has a trade off between `m` (connections between layers) and the 
`ef_construction` (the budget to generate good connections during construction).
One can appreciate, that higher `m` warrants bigger construction budgets. 
Overall, index generation takes a bit longer, but the query speed is (very) 
high with great recalls (if you took the time to generate the index).

**Euclidean:**

Below are the results for the Euclidean distance measure for HSNW.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.32     23632.97     23636.29       1.0000     0.000000
HNSW-M16-ef50-s50                        2277.61      1446.92      3724.53       0.9179     2.405528
HNSW-M16-ef50-s75                        2277.61      1992.09      4269.70       0.9494     1.321233
HNSW-M16-ef50-s100                       2277.61      2439.39      4717.00       0.9646     0.875401
HNSW-M16-ef100-s50                       3477.57      1560.06      5037.62       0.9487     1.827296
HNSW-M16-ef100-s75                       3477.57      2098.31      5575.88       0.9602     1.059740
HNSW-M16-ef100-s100                      3477.57      2580.29      6057.86       0.9657     0.779401
HNSW-M16-ef200-s50                       4636.76      1630.37      6267.13       0.9451     7.930138
HNSW-M16-ef200-s75                       4636.76      2186.08      6822.84       0.9540     4.781705
HNSW-M16-ef200-s100                      4636.76      2686.79      7323.56       0.9575     3.598514
HNSW-M24-ef100-s50                       6566.20      1964.25      8530.45       0.9853     2.229281
HNSW-M24-ef100-s75                       6566.20      2633.62      9199.82       0.9919     0.645076
HNSW-M24-ef100-s100                      6566.20      3197.27      9763.47       0.9942     0.446417
HNSW-M24-ef200-s50                       7893.85      2089.01      9982.87       0.9904     3.698229
HNSW-M24-ef200-s75                       7893.85      2767.42     10661.27       0.9949     1.795212
HNSW-M24-ef200-s100                      7893.85      3348.56     11242.41       0.9961     1.309012
HNSW-M24-ef300-s50                       8999.67      2111.93     11111.61       0.9900     5.048882
HNSW-M24-ef300-s75                       8999.67      2832.98     11832.65       0.9939     2.744873
HNSW-M24-ef300-s100                      8999.67      3438.82     12438.49       0.9954     1.964318
HNSW-M32-ef200-s50                      12229.71      2426.66     14656.38       0.9942     2.703204
HNSW-M32-ef200-s75                      12229.71      3224.33     15454.04       0.9972     0.916286
HNSW-M32-ef200-s100                     12229.71      3874.64     16104.36       0.9980     0.521010
HNSW-M32-ef300-s50                      13511.46      2362.89     15874.35       0.9905     4.458854
HNSW-M32-ef300-s75                      13511.46      3181.24     16692.69       0.9930     3.252330
HNSW-M32-ef300-s100                     13511.46      3828.52     17339.97       0.9941     2.697253
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for HSNW.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  5.48     24217.94     24223.42       1.0000     0.000000
HNSW-M16-ef50-s50                        2258.96      1280.80      3539.75       0.8111     0.008493
HNSW-M16-ef50-s75                        2258.96      1766.45      4025.41       0.8552     0.006526
HNSW-M16-ef50-s100                       2258.96      2211.70      4470.65       0.8778     0.005471
HNSW-M16-ef100-s50                       3447.60      1503.46      4951.06       0.9702     0.002176
HNSW-M16-ef100-s75                       3447.60      2061.05      5508.65       0.9822     0.001205
HNSW-M16-ef100-s100                      3447.60      2541.69      5989.29       0.9881     0.000840
HNSW-M16-ef200-s50                       4631.09      1672.33      6303.42       0.9738     0.011384
HNSW-M16-ef200-s75                       4631.09      2255.11      6886.19       0.9896     0.003017
HNSW-M16-ef200-s100                      4631.09      2783.34      7414.42       0.9939     0.001218
HNSW-M24-ef100-s50                       6616.99      1851.66      8468.65       0.9821     0.003487
HNSW-M24-ef100-s75                       6616.99      2521.53      9138.52       0.9880     0.002576
HNSW-M24-ef100-s100                      6616.99      3068.81      9685.80       0.9906     0.002176
HNSW-M24-ef200-s50                       8568.53      1954.93     10523.46       0.9850     0.005289
HNSW-M24-ef200-s75                       8568.53      2637.52     11206.05       0.9913     0.002450
HNSW-M24-ef200-s100                      8568.53      3218.78     11787.31       0.9933     0.001699
HNSW-M24-ef300-s50                       9037.24      1860.21     10897.45       0.8540     0.020845
HNSW-M24-ef300-s75                       9037.24      2534.44     11571.68       0.8734     0.014365
HNSW-M24-ef300-s100                      9037.24      3134.97     12172.20       0.8824     0.011370
HNSW-M32-ef200-s50                      12207.60      2194.64     14402.24       0.9847     0.006483
HNSW-M32-ef200-s75                      12207.60      2938.73     15146.33       0.9883     0.004823
HNSW-M32-ef200-s100                     12207.60      3559.97     15767.57       0.9898     0.004211
HNSW-M32-ef300-s50                      13576.83      2335.45     15912.28       0.9837     0.008675
HNSW-M32-ef300-s75                      13576.83      3145.23     16722.06       0.9923     0.003159
HNSW-M32-ef300-s100                     13576.83      3858.22     17435.05       0.9946     0.001750
----------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index (powering for example some of the FAISS indices) is very
powerful. Quick index build, quite fast querying times. The number of lists
(especially with this synthetic data) does not need to be particularly high and
you reach quite quickly better speeds over an exhaustive search. Larger
number of lists or points to search do not really make sense (at least not
in this data).

**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.19     23189.02     23192.21       1.0000     0.000000
IVF-nl10-np1                               86.75      8158.93      8245.68       1.0000     0.017658
IVF-nl20-np1                              146.44      4228.44      4374.88       0.9271     0.886739
IVF-nl20-np2                              146.44      8012.64      8159.08       1.0000     0.000000
IVF-nl20-np3                              146.44     11994.35     12140.79       1.0000     0.000000
IVF-nl25-np1                              206.04      4230.13      4436.17       0.9223     1.338138
IVF-nl25-np2                              206.04      7037.68      7243.72       0.9851     0.213944
IVF-nl25-np3                              206.04      9692.11      9898.15       1.0000     0.000000
IVF-nl50-np2                              418.33      3198.72      3617.05       0.9114     1.079659
IVF-nl50-np5                              418.33      6115.85      6534.17       0.9978     0.013559
IVF-nl50-np7                              418.33      8238.44      8656.77       1.0000     0.000000
IVF-nl100-np5                             919.40      3819.22      4738.62       0.9682     0.462880
IVF-nl100-np10                            919.40      6700.05      7619.45       0.9988     0.013839
IVF-nl100-np15                            919.40      9391.20     10310.59       1.0000     0.000000
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.27     24016.62     24020.89       1.0000     0.000000
IVF-nl10-np1                              201.59      7940.34      8141.93       0.9977     0.000096
IVF-nl20-np1                              366.02      3982.04      4348.06       0.9486     0.000332
IVF-nl20-np2                              366.02      7349.22      7715.24       1.0000     0.000000
IVF-nl20-np3                              366.02     10891.27     11257.29       1.0000     0.000000
IVF-nl25-np1                              531.12      3173.55      3704.67       0.9003     0.000801
IVF-nl25-np2                              531.12      5849.01      6380.12       1.0000     0.000000
IVF-nl25-np3                              531.12      8328.66      8859.78       1.0000     0.000000
IVF-nl50-np2                             1051.83      3545.53      4597.36       0.9191     0.000787
IVF-nl50-np5                             1051.83      7226.60      8278.43       0.9969     0.000026
IVF-nl50-np7                             1051.83      9387.84     10439.67       1.0000     0.000000
IVF-nl100-np5                            2520.65      4020.73      6541.37       0.9652     0.000272
IVF-nl100-np10                           2520.65      6767.83      9288.48       0.9986     0.000011
IVF-nl100-np15                           2520.65      9818.66     12339.30       1.0000     0.000000
----------------------------------------------------------------------------------------------------
```

### LSH

Locality sensitive hashing is also provided in this crate. Under the hood it
uses locality-sensitive hash functions that compared to normal hash functions
encourage collisions of similar elements. The two key parameters are the
number of bits you wish to use (the more, the faster the querying at cost of 
recall) and the number of HashMaps to use in the index. The table below gives an 
idea on the influence of the parameters. You also have the option to limit the 
number of candidates to explore during querying at cost of Recall. 

**Euclidean:**

Below are the results for the Euclidean distance measure for LSH.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.31     24061.78     24065.10       1.0000     0.000000
LSH-nt10-bits8:auto                        58.27      6300.16      6358.44       0.9545     0.565262
LSH-nt10-bits8:5k_cand                     58.27      3412.97      3471.25       0.8293     1.906462
LSH-nt20-bits8:auto                       107.57     11007.58     11115.14       0.9959     0.045436
LSH-nt20-bits8:5k_cand                    107.57      3624.01      3731.57       0.8327     1.834968
LSH-nt25-bits8:auto                       137.20     13112.01     13249.21       0.9985     0.017100
LSH-nt25-bits8:5k_cand                    137.20      3617.10      3754.30       0.8327     1.834821
LSH-nt10-bits10:auto                       73.29      4379.62      4452.91       0.9021     1.364292
LSH-nt10-bits10:5k_cand                    73.29      2855.61      2928.90       0.8225     1.998704
LSH-nt20-bits10:auto                      143.20      6558.35      6701.54       0.9799     0.247680
LSH-nt20-bits10:5k_cand                   143.20      3172.71      3315.91       0.8592     1.305625
LSH-nt25-bits10:auto                      176.73      7593.00      7769.72       0.9906     0.112573
LSH-nt25-bits10:5k_cand                   176.73      3216.13      3392.86       0.8619     1.260038
LSH-nt10-bits12:auto                       89.53      3057.12      3146.65       0.8246     2.685095
LSH-nt10-bits12:5k_cand                    89.53      2327.54      2417.07       0.7753     3.002882
LSH-nt20-bits12:auto                      165.21      4956.30      5121.51       0.9485     0.691861
LSH-nt20-bits12:5k_cand                   165.21      2908.29      3073.50       0.8620     1.322305
LSH-nt25-bits12:auto                      213.41      5526.94      5740.36       0.9684     0.410049
LSH-nt25-bits12:5k_cand                   213.41      2965.78      3179.19       0.8725     1.128326
LSH-nt10-bits16:auto                      129.97      1557.16      1687.13       0.6389     7.281324
LSH-nt10-bits16:5k_cand                   129.97      1412.30      1542.27       0.6247     7.356875
LSH-nt20-bits16:auto                      234.96      2690.88      2925.84       0.8146     3.092738
LSH-nt20-bits16:5k_cand                   234.96      2084.46      2319.42       0.7779     3.282065
LSH-nt25-bits16:auto                      287.19      3101.99      3389.19       0.8564     2.284807
LSH-nt25-bits16:5k_cand                   287.19      2238.54      2525.74       0.8135     2.508790
LSH-nt50-bits16:auto                      538.00      5219.63      5757.63       0.9553     0.635930
LSH-nt50-bits16:5k_cand                   538.00      2884.96      3422.96       0.8933     0.988096
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for LSH.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.33     23936.22     23940.55       1.0000     0.000000
LSH-nt10-bits8:auto                        36.74      9462.73      9499.47       0.9802     0.000169
LSH-nt10-bits8:5k_cand                     36.74      3849.80      3886.54       0.7734     0.002304
LSH-nt20-bits8:auto                        59.00     16024.67     16083.67       0.9991     0.000006
LSH-nt20-bits8:5k_cand                     59.00      3806.81      3865.81       0.7736     0.002301
LSH-nt25-bits8:auto                        70.12     18718.44     18788.56       0.9998     0.000002
LSH-nt25-bits8:5k_cand                     70.12      3807.30      3877.42       0.7736     0.002301
LSH-nt10-bits10:auto                       40.54      5827.58      5868.11       0.9509     0.000458
LSH-nt10-bits10:5k_cand                    40.54      3223.52      3264.05       0.8396     0.001245
LSH-nt20-bits10:auto                       72.61      8829.34      8901.95       0.9945     0.000044
LSH-nt20-bits10:5k_cand                    72.61      3357.28      3429.89       0.8514     0.001095
LSH-nt25-bits10:auto                       88.86     10199.22     10288.08       0.9980     0.000016
LSH-nt25-bits10:5k_cand                    88.86      3343.39      3432.25       0.8516     0.001093
LSH-nt10-bits12:auto                       53.11      3892.84      3945.95       0.8984     0.001000
LSH-nt10-bits12:5k_cand                    53.11      2643.00      2696.10       0.8288     0.001381
LSH-nt20-bits12:auto                       88.42      5963.10      6051.52       0.9791     0.000178
LSH-nt20-bits12:5k_cand                    88.42      2957.35      3045.77       0.8733     0.000804
LSH-nt25-bits12:auto                      102.86      7139.77      7242.63       0.9893     0.000092
LSH-nt25-bits12:5k_cand                   102.86      3015.21      3118.07       0.8766     0.000766
LSH-nt10-bits16:auto                       64.03      1971.82      2035.85       0.7258     0.003498
LSH-nt10-bits16:5k_cand                    64.03      1645.77      1709.80       0.6999     0.003607
LSH-nt20-bits16:auto                      118.34      3437.88      3556.23       0.8971     0.001087
LSH-nt20-bits16:5k_cand                   118.34      2274.35      2392.69       0.8394     0.001332
LSH-nt25-bits16:auto                      144.52      3942.61      4087.13       0.9271     0.000739
LSH-nt25-bits16:5k_cand                   144.52      2525.51      2670.03       0.8621     0.001022
LSH-nt50-bits16:auto                      258.76      5993.08      6251.84       0.9832     0.000150
LSH-nt50-bits16:5k_cand                   258.76      2727.81      2986.58       0.9037     0.000514
----------------------------------------------------------------------------------------------------
```

### NNDescent

The NNDescent implementation in this crate, heavily inspired by the amazing
[PyNNDescent](https://github.com/lmcinnes/pynndescent), shows a very good
compromise between index building and fast querying. It's a great arounder
that reaches easily performance of ≥0.98 Recalls@k neighbours. You can even
heavily short cut the initialisation of the index with only 12 trees (instead
of 32) and get in 4 seconds to a recall ≥0.9 (compared to 48 seconds for 
exhaustive search)!

**Euclidean:**

Below are the results for the Euclidean distance measure for NNDescent
implementation in this `crate`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.24     24396.42     24399.67       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                1857.41       951.95      2809.36       0.9593     0.267876
NNDescent-nt24-s:auto-dp0                2667.36       951.79      3619.15       0.9844     0.089375
NNDescent-nt:auto-s75-dp0                2566.26      1297.77      3864.03       0.9917     0.046869
NNDescent-nt:auto-s100-dp0               2566.26      1694.54      4260.79       0.9945     0.031303
NNDescent-nt:auto-s:auto-dp0             2566.26       904.26      3470.51       0.9854     0.082733
NNDescent-nt:auto-s:auto-dp0.25          2593.10       913.23      3506.34       0.9854     0.082733
NNDescent-nt:auto-s:auto-dp0.5           2724.97      1009.27      3734.24       0.9854     0.082733
NNDescent-nt:auto-s:auto-dp1             2704.98       922.44      3627.41       0.9854     0.082733
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for NNDescent
implementation in this `crate`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.81     24013.04     24017.85       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                6140.94       900.66      7041.60       0.9995     0.000002
NNDescent-nt24-s:auto-dp0                5872.24       865.43      6737.67       0.9996     0.000001
NNDescent-nt:auto-s75-dp0                5836.05      1286.77      7122.82       0.9998     0.000001
NNDescent-nt:auto-s100-dp0               5836.05      1674.89      7510.93       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0             5836.05       906.87      6742.91       0.9996     0.000001
NNDescent-nt:auto-s:auto-dp0.25          6299.27       800.83      7100.10       0.9886     0.000037
NNDescent-nt:auto-s:auto-dp0.5           6113.42       772.29      6885.71       0.9783     0.000076
NNDescent-nt:auto-s:auto-dp1             5963.72       705.95      6669.67       0.9571     0.000166
----------------------------------------------------------------------------------------------------
```

## Quantised indices

The crate also provides some quantised approximate nearest neighbour searches, 
designed for very large data sets where memory and time both start becoming 
incredibly constraining. At the moment due to the focus on single cell, 
the only quantisation is SQ8 which transforms a given vector into `i8` and does
symmetric distance calculations (query also transformed to `i8` to leverage
fast integer computations on modern CPUs).

### IVF (with scalar quantisation)

**Euclidean:**

Below are the results for the Euclidean distance measure for IVF with SQ8
quantisation. To note, the mean distance error is not calculated, as the index
does not store the original vectors anymore to reduce memory fingerprint. 

```
===================================================================================
Benchmark: 150k cells, 32D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive-IP                          3.45     24088.29     24091.74       1.0000
IVF-SQ8-nl10-np1                     140.35      1311.36      1451.71       0.8547
IVF-SQ8-nl20-np1                     153.80       721.88       875.69       0.8086
IVF-SQ8-nl20-np2                     153.80      1230.09      1383.89       0.8548
IVF-SQ8-nl20-np3                     153.80      1818.21      1972.01       0.8548
IVF-SQ8-nl25-np1                     252.32       729.15       981.47       0.8023
IVF-SQ8-nl25-np2                     252.32      1156.68      1409.00       0.8467
IVF-SQ8-nl25-np3                     252.32      1519.53      1771.85       0.8548
IVF-SQ8-nl50-np2                     411.27       583.69       994.96       0.8030
IVF-SQ8-nl50-np5                     411.27      1018.45      1429.72       0.8538
IVF-SQ8-nl50-np7                     411.27      1317.96      1729.23       0.8548
IVF-SQ8-nl100-np5                    995.34       660.64      1655.98       0.8374
IVF-SQ8-nl100-np10                   995.34      1105.49      2100.83       0.8543
IVF-SQ8-nl100-np15                   995.34      1554.03      2549.38       0.8548
-----------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF with SQ8
quantisation. To note, the mean distance error is not calculated, as the index
does not store the original vectors anymore to reduce memory fingerprint. 

```
===================================================================================
Benchmark: 150k cells, 32D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive-IP                          4.45     24035.16     24039.61       1.0000
IVF-SQ8-nl10-np1                     205.73      1224.28      1430.01       0.8594
IVF-SQ8-nl20-np1                     373.75       659.22      1032.97       0.8294
IVF-SQ8-nl20-np2                     373.75      1125.76      1499.51       0.8623
IVF-SQ8-nl20-np3                     373.75      1598.88      1972.63       0.8623
IVF-SQ8-nl25-np1                     555.26       585.04      1140.30       0.8067
IVF-SQ8-nl25-np2                     555.26      1004.46      1559.72       0.8623
IVF-SQ8-nl25-np3                     555.26      1396.99      1952.25       0.8623
IVF-SQ8-nl50-np2                    1051.30       627.63      1678.93       0.8130
IVF-SQ8-nl50-np5                    1051.30      1165.45      2216.75       0.8608
IVF-SQ8-nl50-np7                    1051.30      1492.41      2543.71       0.8623
IVF-SQ8-nl100-np5                   2538.72       723.46      3262.18       0.8447
IVF-SQ8-nl100-np10                  2538.72      1154.77      3693.49       0.8614
IVF-SQ8-nl100-np15                  2538.72      1624.15      4162.86       0.8623
-----------------------------------------------------------------------------------

```

#### More cells and dimensions

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