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

- **Quantised indices**:
  - *IVF-SQ8* (with scalar quantisation)
  - *IVF-PQ* (with product quantisation)
  - *IVF-OPQ* (with optimised product quantisation)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ann-search-rs = "*" # always get the latest version
```

## Roadmap

- Longer term, I am considering GPU-acceleration (yet to be figured out how,
likely via the [Burn framework](https://burn.dev)).
- Option to save indices on-disk and do on-disk querying.

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
function, see `./examples/commons/mod.rs`.

### Annoy

Approximate nearest neighbours Oh Yeah. A tree-based method for vector searches.
Fast index building and good query speed.

**Key parameters:**

- *Number of trees (nt)*: The number of trees to generate in the forest
- *Search budget (s)*: The search budget per tree. If set to auto it uses
  `k * n_trees * 20`; versions with a `10x` or `5x` (i.e., less) are also shown.

**Euclidean:**

Below are the results for the Euclidean distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.37     23523.50     23526.87       1.0000     0.000000
Annoy-nt5-s:auto                           72.77       809.88       882.65       0.6408    81.683927
Annoy-nt5-s:10x                            72.77       586.92       659.69       0.5175    81.587548
Annoy-nt5-s:5x                             72.77       362.61       435.38       0.3713    81.392453
Annoy-nt10-s:auto                          94.91      1520.99      1615.90       0.8479    81.814313
Annoy-nt10-s:10x                           94.91      1071.05      1165.96       0.7378    81.763178
Annoy-nt10-s:5x                            94.91       665.56       760.47       0.5603    81.632177
Annoy-nt15-s:auto                         151.49      2186.96      2338.45       0.9338    81.853076
Annoy-nt15-s:10x                          151.49      1539.40      1690.89       0.8542    81.824079
Annoy-nt15-s:5x                           151.49       969.16      1120.65       0.6898    81.734461
Annoy-nt25-s:auto                         238.07      3383.97      3622.03       0.9854    81.872143
Annoy-nt25-s:10x                          238.07      2447.85      2685.91       0.9512    81.862444
Annoy-nt25-s:5x                           238.07      1557.78      1795.85       0.8394    81.818440
Annoy-nt50-s:auto                         439.72      5871.01      6310.73       0.9994    81.876491
Annoy-nt50-s:10x                          439.72      4437.93      4877.65       0.9959    81.875733
Annoy-nt50-s:5x                           439.72      2940.64      3380.35       0.9647    81.867209
Annoy-nt75-s:auto                         649.73      8241.52      8891.24       1.0000    81.876624
Annoy-nt75-s:10x                          649.73      6361.06      7010.78       0.9995    81.876551
Annoy-nt75-s:5x                           649.73      4338.25      4987.98       0.9909    81.874592
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
Exhaustive                                  5.13     24048.95     24054.07       1.0000     0.000000
Annoy-nt5-s:auto                           73.75       312.45       386.19       0.3459     0.007802
Annoy-nt5-s:10x                            73.75       320.43       394.18       0.3459     0.007802
Annoy-nt5-s:5x                             73.75       308.15       381.90       0.3378     0.008007
Annoy-nt10-s:auto                          97.42       579.70       677.13       0.5300     0.003925
Annoy-nt10-s:10x                           97.42       588.16       685.58       0.5300     0.003925
Annoy-nt10-s:5x                            97.42       576.55       673.98       0.5241     0.004002
Annoy-nt15-s:auto                         148.80       795.59       944.40       0.6573     0.002336
Annoy-nt15-s:10x                          148.80       794.16       942.96       0.6573     0.002336
Annoy-nt15-s:5x                           148.80       786.41       935.21       0.6535     0.002372
Annoy-nt25-s:auto                         230.48      1388.85      1619.34       0.8131     0.000991
Annoy-nt25-s:10x                          230.48      1402.02      1632.51       0.8131     0.000991
Annoy-nt25-s:5x                           230.48      1372.65      1603.13       0.8115     0.001001
Annoy-nt50-s:auto                         426.71      2528.20      2954.91       0.9540     0.000177
Annoy-nt50-s:10x                          426.71      2530.38      2957.09       0.9540     0.000177
Annoy-nt50-s:5x                           426.71      2518.79      2945.50       0.9538     0.000178
Annoy-nt75-s:auto                         650.46      3904.17      4554.63       0.9871     0.000041
Annoy-nt75-s:10x                          650.46      3919.21      4569.67       0.9871     0.000041
Annoy-nt75-s:5x                           650.46      3930.34      4580.80       0.9871     0.000041
----------------------------------------------------------------------------------------------------
```

### HNSW

Hierarchical navigatable small worlds. A graph-based index that needs more time
to build the index. However, fast query speed.

**Key parameters:**

- *M (m)*: The number of connections between layers
- *EF construction (ef)*: The budget to generate good connections during 
  construction of the index.
- *EF search (s)*: The budget for the search queries. 

**Euclidean:**

Below are the results for the Euclidean distance measure for HSNW.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.61     23390.29     23394.90       1.0000     0.000000
HNSW-M16-ef50-s50                        2104.11      1317.04      3421.15       0.9252     7.516656
HNSW-M16-ef50-s75                        2104.11      1817.25      3921.36       0.9567     4.398063
HNSW-M16-ef50-s100                       2104.11      2264.65      4368.76       0.9708     3.054929
HNSW-M16-ef100-s50                       3429.25      1430.55      4859.79       0.9618     2.821499
HNSW-M16-ef100-s75                       3429.25      1927.23      5356.47       0.9735     1.483684
HNSW-M16-ef100-s100                      3429.25      2387.60      5816.85       0.9787     1.027704
HNSW-M16-ef200-s50                       4127.09      1499.91      5627.01       0.9152    11.290405
HNSW-M16-ef200-s75                       4127.09      2035.84      6162.94       0.9235     7.658499
HNSW-M16-ef200-s100                      4127.09      2500.90      6628.00       0.9267     6.201307
HNSW-M24-ef100-s50                       6291.58      1701.78      7993.36       0.9730     5.491844
HNSW-M24-ef100-s75                       6291.58      2305.94      8597.52       0.9776     4.603578
HNSW-M24-ef100-s100                      6291.58      2803.48      9095.06       0.9796     4.151941
HNSW-M24-ef200-s50                       7286.09      1775.13      9061.23       0.9564     7.359320
HNSW-M24-ef200-s75                       7286.09      2418.02      9704.11       0.9638     3.299870
HNSW-M24-ef200-s100                      7286.09      2924.28     10210.37       0.9661     2.041412
HNSW-M24-ef300-s50                       8351.74      1823.95     10175.68       0.9714    16.680926
HNSW-M24-ef300-s75                       8351.74      2488.24     10839.97       0.9846     9.201803
HNSW-M24-ef300-s100                      8351.74      3017.11     11368.85       0.9890     6.563623
HNSW-M32-ef200-s50                      11285.16      2169.19     13454.35       0.9846     8.716828
HNSW-M32-ef200-s75                      11285.16      2914.92     14200.08       0.9904     5.236196
HNSW-M32-ef200-s100                     11285.16      3524.02     14809.18       0.9929     3.754602
HNSW-M32-ef300-s50                      12322.42      2137.78     14460.20       0.9558     7.227940
HNSW-M32-ef300-s75                      12322.42      2849.39     15171.82       0.9641     3.714656
HNSW-M32-ef300-s100                     12322.42      3483.47     15805.89       0.9679     2.224845
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
Exhaustive                                  4.61     24075.74     24080.34       1.0000     0.000000
HNSW-M16-ef50-s50                        2142.93      1281.87      3424.80       0.8308     0.009739
HNSW-M16-ef50-s75                        2142.93      1763.03      3905.96       0.8606     0.007662
HNSW-M16-ef50-s100                       2142.93      2172.93      4315.86       0.8780     0.006563
HNSW-M16-ef100-s50                       3291.85      1427.58      4719.43       0.9288     0.005761
HNSW-M16-ef100-s75                       3291.85      1972.21      5264.07       0.9511     0.004270
HNSW-M16-ef100-s100                      3291.85      2427.58      5719.44       0.9640     0.003515
HNSW-M16-ef200-s50                       4279.89      1499.46      5779.34       0.9355     0.011694
HNSW-M16-ef200-s75                       4279.89      2085.75      6365.63       0.9525     0.003893
HNSW-M16-ef200-s100                      4279.89      2547.11      6827.00       0.9566     0.002241
HNSW-M24-ef100-s50                       6214.15      1807.51      8021.66       0.9392     0.009069
HNSW-M24-ef100-s75                       6214.15      2395.13      8609.28       0.9477     0.006694
HNSW-M24-ef100-s100                      6214.15      2950.05      9164.19       0.9519     0.005477
HNSW-M24-ef200-s50                       7431.74      1809.10      9240.84       0.9139     0.016804
HNSW-M24-ef200-s75                       7431.74      2445.18      9876.92       0.9245     0.011393
HNSW-M24-ef200-s100                      7431.74      2960.93     10392.67       0.9278     0.009835
HNSW-M24-ef300-s50                       8482.58      1897.40     10379.98       0.8821     0.058018
HNSW-M24-ef300-s75                       8482.58      2597.45     11080.03       0.9009     0.046460
HNSW-M24-ef300-s100                      8482.58      3169.16     11651.74       0.9108     0.040646
HNSW-M32-ef200-s50                      11454.41      2099.81     13554.23       0.9428     0.032554
HNSW-M32-ef200-s75                      11454.41      2829.83     14284.24       0.9517     0.026652
HNSW-M32-ef200-s100                     11454.41      3437.36     14891.78       0.9561     0.023815
HNSW-M32-ef300-s50                      12495.14      2134.02     14629.17       0.8914     0.058028
HNSW-M32-ef300-s75                      12495.14      2880.84     15375.98       0.9040     0.049528
HNSW-M32-ef300-s100                     12495.14      3512.76     16007.91       0.9108     0.045337
----------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index. Uses Voronoi cells to sub-partition the original data.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Usually scanning 15 to 20% of the number of lists yields good results.


**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.42     24779.72     24783.15       1.0000     0.000000
IVF-nl10-np1                               91.88      8195.27      8287.16       0.9981     0.090493
IVF-nl20-np1                              155.00      4011.87      4166.86       0.9765     0.396602
IVF-nl20-np2                              155.00      7450.47      7605.47       1.0000     0.004042
IVF-nl20-np3                              155.00     11209.87     11364.86       1.0000     0.000014
IVF-nl25-np1                              219.24      3438.27      3657.51       0.9647     0.620844
IVF-nl25-np2                              219.24      5882.46      6101.71       1.0000     0.004386
IVF-nl25-np3                              219.24      8471.56      8690.80       1.0000     0.000000
IVF-nl50-np2                              403.96      3143.57      3547.52       0.9333     0.870256
IVF-nl50-np5                              403.96      6791.70      7195.66       1.0000     0.000000
IVF-nl50-np7                              403.96      9393.38      9797.34       1.0000     0.000000
IVF-nl100-np5                             923.45      3969.09      4892.53       0.9713     0.378366
IVF-nl100-np10                            923.45      6935.36      7858.81       0.9996     0.004021
IVF-nl100-np15                            923.45     10002.61     10926.06       1.0000     0.000000
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
Exhaustive                                  5.87     24029.04     24034.92       1.0000     0.000000
IVF-nl10-np1                              370.40      7446.42      7816.82       0.9993     0.000106
IVF-nl20-np1                              365.63      4147.90      4513.53       0.9923     0.000167
IVF-nl20-np2                              365.63      7435.59      7801.22       0.9999     0.000010
IVF-nl20-np3                              365.63     11004.87     11370.50       1.0000     0.000001
IVF-nl25-np1                              556.05      3313.87      3869.93       0.9697     0.000209
IVF-nl25-np2                              556.05      5901.96      6458.01       1.0000     0.000000
IVF-nl25-np3                              556.05      8655.64      9211.69       1.0000     0.000000
IVF-nl50-np2                             1035.42      3443.88      4479.30       0.9283     0.000624
IVF-nl50-np5                             1035.42      7524.63      8560.05       1.0000     0.000000
IVF-nl50-np7                             1035.42      9915.22     10950.64       1.0000     0.000000
IVF-nl100-np5                            2545.14      4053.75      6598.89       0.9713     0.000264
IVF-nl100-np10                           2545.14      7222.58      9767.72       0.9990     0.000008
IVF-nl100-np15                           2545.14     10410.93     12956.07       1.0000     0.000000
----------------------------------------------------------------------------------------------------
```

### LSH

Locality sensitive hashing.

**Key parameters:**

- *Number of tables (nt)*: The number of independent hash tables to generate. 
  More tables improve recall at the cost of query time and memory.
- *Number of bits (nb)*: The bit resolution of the hash functions. Higher values 
  create finer partitions but may reduce collision rates.
- *Max candidates (s)*: The search budget limiting the number of candidates 
  examined. Set to 'auto' for full search or a fixed value (e.g., 5k) for faster 
  queries with lower recall.

**Euclidean:**

Below are the results for the Euclidean distance measure for LSH.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  5.27     23235.57     23240.84       1.0000     0.000000
LSH-nt10-nb8:-s:auto                       57.44      5969.75      6027.19       0.9638     0.480160
LSH-nt10-nb8:-s:5k                         57.44      3310.59      3368.04       0.8537     1.571043
LSH-nt20-nb8:-s:auto                      111.39     10233.56     10344.96       0.9967     0.038897
LSH-nt20-nb8:-s:5k                        111.39      3525.71      3637.11       0.8595     1.469591
LSH-nt25-nb8:-s:auto                      136.33     12506.81     12643.14       0.9987     0.014704
LSH-nt25-nb8:-s:5k                        136.33      3547.63      3683.96       0.8596     1.469395
LSH-nt10-nb10:-s:auto                      74.99      3955.41      4030.41       0.9114     1.283957
LSH-nt10-nb10:-s:5k                        74.99      2675.10      2750.10       0.8493     1.675435
LSH-nt20-nb10:-s:auto                     135.91      5980.99      6116.90       0.9818     0.229392
LSH-nt20-nb10:-s:5k                       135.91      3008.77      3144.68       0.8902     0.944406
LSH-nt25-nb10:-s:auto                     169.14      6754.45      6923.59       0.9914     0.104942
LSH-nt25-nb10:-s:5k                       169.14      3045.08      3214.22       0.8932     0.897108
LSH-nt10-nb12:-s:auto                      93.55      2990.19      3083.74       0.8431     2.506719
LSH-nt10-nb12:-s:5k                        93.55      2226.47      2320.02       0.8021     2.687833
LSH-nt20-nb12:-s:auto                     165.11      4643.25      4808.36       0.9547     0.632133
LSH-nt20-nb12:-s:5k                       165.11      2822.00      2987.11       0.8929     0.977663
LSH-nt25-nb12:-s:auto                     208.64      5117.29      5325.93       0.9720     0.373990
LSH-nt25-nb12:-s:5k                       208.64      2906.96      3115.60       0.9044     0.780383
LSH-nt10-nb16:-s:auto                     109.25      1861.30      1970.54       0.6891     6.654130
LSH-nt10-nb16:-s:5k                       109.25      1544.76      1654.00       0.6680     6.721914
LSH-nt20-nb16:-s:auto                     218.61      2961.96      3180.58       0.8391     2.829489
LSH-nt20-nb16:-s:5k                       218.61      2056.63      2275.24       0.8022     2.961812
LSH-nt25-nb16:-s:auto                     286.95      3371.30      3658.25       0.8751     2.077903
LSH-nt25-nb16:-s:5k                       286.95      2224.71      2511.66       0.8346     2.227713
LSH-nt50-nb16:-s:auto                     516.32      5179.43      5695.75       0.9597     0.585007
LSH-nt50-nb16:-s:5k                       516.32      2875.00      3391.33       0.9107     0.793054
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
Exhaustive                                  4.95     24363.62     24368.57       1.0000     0.000000
LSH-nt10-nb8:-s:auto                       34.98      8698.38      8733.36       0.9807     0.000164
LSH-nt10-nb8:-s:5k                         34.98      3804.06      3839.04       0.8236     0.001536
LSH-nt20-nb8:-s:auto                       60.64     15206.81     15267.45       0.9992     0.000006
LSH-nt20-nb8:-s:5k                         60.64      3781.17      3841.81       0.8239     0.001533
LSH-nt25-nb8:-s:auto                       69.13     18176.21     18245.35       0.9998     0.000002
LSH-nt25-nb8:-s:5k                         69.13      3784.18      3853.31       0.8239     0.001533
LSH-nt10-nb10:-s:auto                      41.19      5189.09      5230.28       0.9543     0.000426
LSH-nt10-nb10:-s:5k                        41.19      3121.45      3162.64       0.8669     0.000950
LSH-nt20-nb10:-s:auto                      69.90      7964.61      8034.51       0.9948     0.000042
LSH-nt20-nb10:-s:5k                        69.90      3273.96      3343.86       0.8797     0.000791
LSH-nt25-nb10:-s:auto                      85.88      9086.07      9171.95       0.9981     0.000016
LSH-nt25-nb10:-s:5k                        85.88      3266.55      3352.44       0.8799     0.000789
LSH-nt10-nb12:-s:auto                      50.17      3567.99      3618.16       0.9078     0.000924
LSH-nt10-nb12:-s:5k                        50.17      2540.27      2590.44       0.8592     0.001114
LSH-nt20-nb12:-s:auto                      87.08      5408.18      5495.27       0.9813     0.000162
LSH-nt20-nb12:-s:5k                        87.08      2889.75      2976.83       0.9078     0.000521
LSH-nt25-nb12:-s:auto                     103.60      5994.96      6098.56       0.9904     0.000084
LSH-nt25-nb12:-s:5k                       103.60      2921.76      3025.36       0.9116     0.000480
LSH-nt10-nb16:-s:auto                      63.23      2087.50      2150.72       0.7512     0.003265
LSH-nt10-nb16:-s:5k                        63.23      1657.20      1720.43       0.7274     0.003324
LSH-nt20-nb16:-s:auto                     113.07      3414.59      3527.66       0.9067     0.001003
LSH-nt20-nb16:-s:5k                       113.07      2249.06      2362.13       0.8673     0.001112
LSH-nt25-nb16:-s:auto                     143.23      3872.65      4015.89       0.9344     0.000677
LSH-nt25-nb16:-s:5k                       143.23      2382.90      2526.13       0.8916     0.000801
LSH-nt50-nb16:-s:auto                     260.29      5643.84      5904.12       0.9852     0.000139
LSH-nt50-nb16:-s:5k                       260.29      2765.60      3025.89       0.9332     0.000314
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

**Key parameters:**

- *Number of trees (nt)*: Number of trees to use for the initialisation. If set
  to auto, it defaults to 32.
- *Search budget (s)*: The search budget for the exploration of the graph during
  querying. Here it defaults to `k * 2` (with min 60, maximum 200).
- *Diversify probability (dp)*: This is based on the original papers leveraging
  NNDescent and it is supposed to remove redundant edges from the graph to
  increase query speed.

**Euclidean:**

Below are the results for the Euclidean distance measure for NNDescent
implementation in this `crate`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.99     23157.09     23162.08       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                1756.92       910.33      2667.25       0.9687     0.189218
NNDescent-nt24-s:auto-dp0                2485.42       922.47      3407.89       0.9885     0.061129
NNDescent-nt:auto-s75-dp0                2526.79      1248.13      3774.92       0.9939     0.031297
NNDescent-nt:auto-s100-dp0               2526.79      1628.16      4154.94       0.9960     0.020544
NNDescent-nt:auto-s:auto-dp0             2526.79       931.00      3457.79       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.25          2568.93       881.91      3450.84       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.5           2602.86       892.60      3495.46       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp1             2596.52       889.45      3485.97       0.9893     0.056484
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
Exhaustive                                  4.50     24142.07     24146.57       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                5837.83       841.81      6679.64       0.9997     0.000001
NNDescent-nt24-s:auto-dp0                5586.89       843.60      6430.50       0.9998     0.000001
NNDescent-nt:auto-s75-dp0                5429.10      1225.21      6654.31       0.9999     0.000000
NNDescent-nt:auto-s100-dp0               5429.10      1593.40      7022.50       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0             5429.10       837.02      6266.12       0.9997     0.000001
NNDescent-nt:auto-s:auto-dp0.25          5486.43       820.87      6307.30       0.9899     0.000030
NNDescent-nt:auto-s:auto-dp0.5           5662.70       754.05      6416.75       0.9806     0.000062
NNDescent-nt:auto-s:auto-dp1             5528.17       717.38      6245.55       0.9613     0.000135
----------------------------------------------------------------------------------------------------
```

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

### IVF-SQ8

This index uses scalar quantisation to 8-bits. It projects every dimensions
onto an `i8`. This also causes a reduction of the memory finger print. In the
case of 96 dimensions in f32 per vector, we go from *96 x 32 bits = 384 bytes*
to *96 x 8 bits = 96 bytes per vector*, a **4x reduction in memory per vector** 
(with overhead of the codebook).

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to search for. Usually 
  scanning 15 to 20% of the number of lists.

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
Exhaustive                             3.37     23513.19     23516.56       1.0000
IVF-SQ8-nl10-np1                     132.87      1327.39      1460.26       0.8553
IVF-SQ8-nl20-np1                     179.74       608.70       788.44       0.8415
IVF-SQ8-nl20-np2                     179.74      1071.75      1251.50       0.8567
IVF-SQ8-nl20-np3                     179.74      1561.34      1741.08       0.8567
IVF-SQ8-nl25-np1                     217.46       540.73       758.19       0.8337
IVF-SQ8-nl25-np2                     217.46       876.57      1094.03       0.8567
IVF-SQ8-nl25-np3                     217.46      1210.26      1427.72       0.8567
IVF-SQ8-nl50-np2                     387.17       516.36       903.53       0.8167
IVF-SQ8-nl50-np5                     387.17       985.62      1372.79       0.8567
IVF-SQ8-nl50-np7                     387.17      1327.84      1715.01       0.8567
IVF-SQ8-nl100-np5                    907.31       660.42      1567.73       0.8412
IVF-SQ8-nl100-np10                   907.31      1042.99      1950.29       0.8565
IVF-SQ8-nl100-np15                   907.31      1484.96      2392.27       0.8567
-----------------------------------------------------------------------------------
```

**With more dimensions**

In this case, we increase the dimensions from 32 to 96.

```
===================================================================================
Benchmark: 150k cells, 96D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            12.41    106495.61    106508.02       1.0000
IVF-SQ8-nl10-np1                     513.90      4518.84      5032.74       0.8342
IVF-SQ8-nl20-np1                     621.55      2481.98      3103.53       0.8224
IVF-SQ8-nl20-np2                     621.55      6290.28      6911.83       0.8345
IVF-SQ8-nl20-np3                     621.55      9660.60     10282.15       0.8345
IVF-SQ8-nl25-np1                     845.18      2010.54      2855.72       0.7701
IVF-SQ8-nl25-np2                     845.18      4213.22      5058.40       0.8289
IVF-SQ8-nl25-np3                     845.18      6491.24      7336.42       0.8345
IVF-SQ8-nl50-np2                    1653.27      1604.25      3257.52       0.7744
IVF-SQ8-nl50-np5                    1653.27      4110.53      5763.80       0.8345
IVF-SQ8-nl50-np7                    1653.27      5745.57      7398.85       0.8345
IVF-SQ8-nl100-np5                   4074.50      2349.44      6423.94       0.7955
IVF-SQ8-nl100-np10                  4074.50      4152.39      8226.88       0.8293
IVF-SQ8-nl100-np15                  4074.50      5884.22      9958.72       0.8345
-----------------------------------------------------------------------------------
```

**Data set with stronger correlation structure**

To compare against the next two indices. This data is designed to be better 
suited for high dimensionality.

```
===================================================================================
Benchmark: 150k cells, 128D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            16.90    170023.89    170040.79       1.0000
IVF-SQ8-nl10-np1                     529.20      6242.15      6771.35       0.6422
IVF-SQ8-nl20-np1                     938.95      3001.32      3940.27       0.6422
IVF-SQ8-nl20-np2                     938.95      6546.20      7485.15       0.6422
IVF-SQ8-nl20-np3                     938.95      9745.26     10684.21       0.6422
IVF-SQ8-nl25-np1                    1360.90      2550.12      3911.02       0.6353
IVF-SQ8-nl25-np2                    1360.90      5250.34      6611.24       0.6422
IVF-SQ8-nl25-np3                    1360.90      7731.06      9091.97       0.6422
IVF-SQ8-nl50-np2                    2633.82      2482.75      5116.57       0.6320
IVF-SQ8-nl50-np5                    2633.82      5230.11      7863.93       0.6422
IVF-SQ8-nl50-np7                    2633.82      6964.30      9598.12       0.6422
IVF-SQ8-nl100-np5                   6272.42      2670.09      8942.51       0.6401
IVF-SQ8-nl100-np10                  6272.42      4929.05     11201.47       0.6422
IVF-SQ8-nl100-np15                  6272.42      7393.44     13665.87       0.6422
-----------------------------------------------------------------------------------
```

### IVF-PQ

This index uses product quantisation. To note, the quantisation is quite harsh 
and hence, reduces the Recall quite substantially. Each vector gets reduced to
from *192 x 32 bits (192 x f32) = 768 bytes* to for 
*m = 32 (32 sub vectors) to 32 x u8 = 32 bytes*, a 
**24x reduction in memory usage** (of course with overhead from the cook book). 
However, it can still be useful in situation where good enough works and you 
have VERY large scale data.</br>
This version is run on a special version of the synthetic data that is less
affected by the curse of dimensionality! Also, there are some correlated 
features in there that can be theoretically better exploited by optimised
product quantisation, see below.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to search for. Usually 
  scanning 15 to 20% of the number of lists.
- *Number of subvectors (m)*: In how many subvectors to divide the given main
  vector. The initial dimensionality needs to be divisable by m.

**Euclidean:**

With 128 dimensions.

```
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            15.48    171113.43    171128.91       1.0000
IVF-PQ-nl10-m16-np1                 6966.22      3150.53     10116.75       0.3630
IVF-PQ-nl10-m32-np1                10098.03      6682.09     16780.12       0.4879
IVF-PQ-nl20-m16-np1                 7345.96      1821.47      9167.43       0.4615
IVF-PQ-nl20-m16-np2                 7345.96      4171.80     11517.76       0.4615
IVF-PQ-nl20-m16-np3                 7345.96      5930.64     13276.60       0.4615
IVF-PQ-nl20-m32-np1                10539.48      3688.99     14228.47       0.6120
IVF-PQ-nl20-m32-np2                10539.48      8843.03     19382.51       0.6120
IVF-PQ-nl20-m32-np3                10539.48     12790.68     23330.16       0.6120
IVF-PQ-nl25-m16-np1                 8148.12      1635.92      9784.04       0.4833
IVF-PQ-nl25-m16-np2                 8148.12      3320.57     11468.69       0.4883
IVF-PQ-nl25-m16-np3                 8148.12      4925.77     13073.89       0.4883
IVF-PQ-nl25-m32-np1                12439.03      3275.81     15714.85       0.6343
IVF-PQ-nl25-m32-np2                12439.03      6957.59     19396.63       0.6420
IVF-PQ-nl25-m32-np3                12439.03     10302.13     22741.16       0.6420
IVF-PQ-nl50-m16-np2                 9118.11      1950.65     11068.75       0.5410
IVF-PQ-nl50-m16-np5                 9118.11      3919.13     13037.23       0.5483
IVF-PQ-nl50-m16-np7                 9118.11      5126.94     14245.04       0.5483
IVF-PQ-nl50-m32-np2                12408.44      3624.94     16033.38       0.7050
IVF-PQ-nl50-m32-np5                12408.44      7324.25     19732.69       0.7171
IVF-PQ-nl50-m32-np7                12408.44      9581.92     21990.36       0.7171
IVF-PQ-nl100-m16-np5               12900.83      2759.08     15659.91       0.5502
IVF-PQ-nl100-m16-np10              12900.83      5110.85     18011.68       0.5517
IVF-PQ-nl100-m16-np15              12900.83      7603.05     20503.88       0.5517
IVF-PQ-nl100-m32-np5               16289.65      4846.47     21136.12       0.7172
IVF-PQ-nl100-m32-np10              16289.65      9066.28     25355.93       0.7199
IVF-PQ-nl100-m32-np15              16289.65     13639.33     29928.98       0.7199
-----------------------------------------------------------------------------------
```

One can appreciate that PQ in this case can yield higher Recalls than just
SQ8 going from ca. 0.64 to 0.72. However, index building and query time are both
higher. However, you also reduce the memory finger print quite substantially.

**With more dimensions**

With 192 dimensions. In this case, also m = 48, i.e., dividing the origina 
vectors into 48 subvectors was tested.

```
===================================================================================
Benchmark: 150k cells, 192D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            23.71    288695.42    288719.13       1.0000
IVF-PQ-nl10-m16-np1                11646.72      4534.24     16180.96       0.2940
IVF-PQ-nl10-m32-np1                14445.56      9817.59     24263.15       0.4281
IVF-PQ-nl10-m48-np1                16358.32     15100.48     31458.80       0.5182
IVF-PQ-nl20-m16-np1                11274.09      1939.92     13214.01       0.3779
IVF-PQ-nl20-m16-np2                11274.09      3672.10     14946.19       0.3779
IVF-PQ-nl20-m16-np3                11274.09      5913.59     17187.68       0.3779
IVF-PQ-nl20-m32-np1                14883.37      3676.42     18559.78       0.5370
IVF-PQ-nl20-m32-np2                14883.37      7342.15     22225.52       0.5370
IVF-PQ-nl20-m32-np3                14883.37     11666.08     26549.45       0.5370
IVF-PQ-nl20-m48-np1                17140.87      5551.50     22692.37       0.6408
IVF-PQ-nl20-m48-np2                17140.87     11575.37     28716.24       0.6408
IVF-PQ-nl20-m48-np3                17140.87     18278.17     35419.04       0.6408
IVF-PQ-nl25-m16-np1                12136.07      1683.36     13819.43       0.4202
IVF-PQ-nl25-m16-np2                12136.07      3609.25     15745.32       0.4227
IVF-PQ-nl25-m16-np3                12136.07      4948.15     17084.23       0.4227
IVF-PQ-nl25-m32-np1                15753.81      3203.84     18957.65       0.5932
IVF-PQ-nl25-m32-np2                15753.81      7110.32     22864.13       0.5976
IVF-PQ-nl25-m32-np3                15753.81     10330.47     26084.28       0.5976
IVF-PQ-nl25-m48-np1                17872.94      4887.58     22760.52       0.6984
IVF-PQ-nl25-m48-np2                17872.94     11087.74     28960.67       0.7044
IVF-PQ-nl25-m48-np3                17872.94     15145.60     33018.53       0.7044
IVF-PQ-nl50-m16-np2                13973.30      2003.18     15976.48       0.4354
IVF-PQ-nl50-m16-np5                13973.30      4479.03     18452.33       0.4444
IVF-PQ-nl50-m16-np7                13973.30      6029.90     20003.20       0.4444
IVF-PQ-nl50-m32-np2                17723.22      3539.87     21263.09       0.6070
IVF-PQ-nl50-m32-np5                17723.22      8053.70     25776.92       0.6233
IVF-PQ-nl50-m32-np7                17723.22     11389.42     29112.64       0.6233
IVF-PQ-nl50-m48-np2                20224.95      5510.25     25735.20       0.7083
IVF-PQ-nl50-m48-np5                20224.95     12060.18     32285.13       0.7303
IVF-PQ-nl50-m48-np7                20224.95     16491.01     36715.96       0.7303
IVF-PQ-nl100-m16-np5               19952.35      3022.39     22974.75       0.4443
IVF-PQ-nl100-m16-np10              19952.35      5575.88     25528.23       0.4481
IVF-PQ-nl100-m16-np15              19952.35      8297.43     28249.78       0.4481
IVF-PQ-nl100-m32-np5               22743.64      5391.92     28135.56       0.6182
IVF-PQ-nl100-m32-np10              22743.64     10073.69     32817.33       0.6257
IVF-PQ-nl100-m32-np15              22743.64     15251.43     37995.07       0.6257
IVF-PQ-nl100-m48-np5               26600.53      7496.64     34097.17       0.7216
IVF-PQ-nl100-m48-np10              26600.53     14039.71     40640.23       0.7320
IVF-PQ-nl100-m48-np15              26600.53     22019.12     48619.65       0.7321
-----------------------------------------------------------------------------------
```

### IVF-OPQ

This index uses optimised product quantisation - this substantially increases
the build time. Similar to IVF-PQ, the quantisation is quite harsh and hence, 
reduces the recall quite substantially compared to exhaustive search. Each 
vector gets reduced to from *192 x 32 bits (192 x f32) = 768 bytes* to for 
*m = 32 (32 sub vectors) to 32 x u8 = 32 bytes*, a 
**24x reduction in memory usage** (of course with overhead from the cook book). 
However, it can still be useful in situation where good enough works and you 
have VERY large scale data. The theoretical benefits at least in this
synthetic data do not translate very well. IVF-PQ is usually more than enough, 
outside of cases in which a specific correlation structure can be exploited
by the optimised PQ. In doubt, use the IVF-PQ index.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to search for. Usually 
  scanning 15 to 20% of the number of lists.
- *Number of subvectors (m)*: In how many subvectors to divide the given main
  vector. The initial dimensionality needs to be divisable by m.

**Euclidean:**

With 128 dimensions.

```
===================================================================================
Benchmark: 150k cells, 128D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            15.76    177863.96    177879.72       1.0000
IVF-OPQ-nl10-m16-np1               21317.79      3154.91     24472.70       0.3633
IVF-OPQ-nl10-m32-np1               30600.10      6740.79     37340.89       0.5043
IVF-OPQ-nl20-m16-np1               22766.92      2022.24     24789.16       0.4617
IVF-OPQ-nl20-m16-np2               22766.92      4363.23     27130.15       0.4617
IVF-OPQ-nl20-m16-np3               22766.92      6055.66     28822.58       0.4617
IVF-OPQ-nl20-m32-np1               30677.18      3853.00     34530.19       0.6218
IVF-OPQ-nl20-m32-np2               30677.18      9125.19     39802.37       0.6218
IVF-OPQ-nl20-m32-np3               30677.18     12918.39     43595.57       0.6218
IVF-OPQ-nl25-m16-np1               22034.21      1844.14     23878.34       0.4833
IVF-OPQ-nl25-m16-np2               22034.21      3787.78     25821.99       0.4882
IVF-OPQ-nl25-m16-np3               22034.21      5185.69     27219.90       0.4882
IVF-OPQ-nl25-m32-np1               30806.12      3333.21     34139.33       0.6424
IVF-OPQ-nl25-m32-np2               30806.12      7045.58     37851.70       0.6503
IVF-OPQ-nl25-m32-np3               30806.12     10349.99     41156.11       0.6503
IVF-OPQ-nl50-m16-np2               22326.73      2202.78     24529.51       0.5410
IVF-OPQ-nl50-m16-np5               22326.73      4827.33     27154.06       0.5483
IVF-OPQ-nl50-m16-np7               22326.73      6174.45     28501.17       0.5483
IVF-OPQ-nl50-m32-np2               30681.98      3776.73     34458.71       0.7100
IVF-OPQ-nl50-m32-np5               30681.98      8549.89     39231.87       0.7221
IVF-OPQ-nl50-m32-np7               30681.98     10622.56     41304.53       0.7221
IVF-OPQ-nl100-m16-np5              27011.31      3575.80     30587.11       0.5502
IVF-OPQ-nl100-m16-np10             27011.31      6730.96     33742.27       0.5518
IVF-OPQ-nl100-m16-np15             27011.31     10129.60     37140.91       0.5518
IVF-OPQ-nl100-m32-np5              38038.66      5366.80     43405.46       0.7227
IVF-OPQ-nl100-m32-np10             38038.66     10211.27     48249.93       0.7255
IVF-OPQ-nl100-m32-np15             38038.66     15222.34     53261.00       0.7255
-----------------------------------------------------------------------------------
```

**With more dimensions**

With 192 dimensions.

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