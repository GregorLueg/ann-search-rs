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

- **Quantised indices**:
  - *IVF-SQ8* (with scalar quantisation)
  - *IVF-PQ* (with product quantisation)
  - *IVF-OPQ* (with optimised product quantisation)

- **GPU-accelerated indices**:
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
with 64 GB of unified memory.

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

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  5.02     23963.35     23968.37       1.0000     0.000000
Annoy-nt5-s:auto                           75.00       306.31       381.31       0.3459     0.007802
Annoy-nt5-s:10x                            75.00       318.52       393.52       0.3459     0.007802
Annoy-nt5-s:5x                             75.00       307.78       382.78       0.3378     0.008007
Annoy-nt10-s:auto                          98.01       574.53       672.54       0.5300     0.003925
Annoy-nt10-s:10x                           98.01       588.22       686.22       0.5300     0.003925
Annoy-nt10-s:5x                            98.01       581.35       679.36       0.5241     0.004002
Annoy-nt15-s:auto                         156.68       843.23       999.91       0.6573     0.002336
Annoy-nt15-s:10x                          156.68       845.93      1002.61       0.6573     0.002336
Annoy-nt15-s:5x                           156.68       844.36      1001.04       0.6535     0.002372
Annoy-nt25-s:auto                         243.94      1298.36      1542.29       0.8131     0.000991
Annoy-nt25-s:10x                          243.94      1294.02      1537.95       0.8131     0.000991
Annoy-nt25-s:5x                           243.94      1279.33      1523.26       0.8115     0.001001
Annoy-nt50-s:auto                         474.03      2666.33      3140.36       0.9540     0.000177
Annoy-nt50-s:10x                          474.03      2663.56      3137.59       0.9540     0.000177
Annoy-nt50-s:5x                           474.03      2679.21      3153.24       0.9538     0.000178
Annoy-nt75-s:auto                         710.69      4250.26      4960.95       0.9871     0.000041
Annoy-nt75-s:10x                          710.69      4506.63      5217.32       0.9871     0.000041
Annoy-nt75-s:5x                           710.69      4385.25      5095.94       0.9871     0.000041
Annoy-nt100-s:auto                        948.39      5345.15      6293.55       0.9960     0.000011
Annoy-nt100-s:10x                         948.39      5489.02      6437.41       0.9960     0.000011
Annoy-nt100-s:5x                          948.39      6014.70      6963.09       0.9960     0.000011
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
Exhaustive                                  3.43     25397.50     25400.93       1.0000     0.000000
HNSW-M16-ef50-s50                        2432.58      1516.70      3949.28       0.8879     4.623140
HNSW-M16-ef50-s75                        2432.58      2045.14      4477.73       0.9120     3.478027
HNSW-M16-ef50-s100                       2432.58      2521.75      4954.33       0.9244     2.700429
HNSW-M16-ef100-s50                       3746.53      1563.14      5309.67       0.9765     1.656335
HNSW-M16-ef100-s75                       3746.53      2041.77      5788.30       0.9863     0.914656
HNSW-M16-ef100-s100                      3746.53      2562.24      6308.77       0.9908     0.656868
HNSW-M16-ef200-s50                       4775.80      1568.20      6344.00       0.9656     4.918914
HNSW-M16-ef200-s75                       4775.80      2283.27      7059.07       0.9726     2.126443
HNSW-M16-ef200-s100                      4775.80      2672.91      7448.72       0.9748     1.470300
HNSW-M24-ef100-s50                       6282.72      1820.96      8103.68       0.9400     7.885647
HNSW-M24-ef100-s75                       6282.72      2428.13      8710.85       0.9479     5.724750
HNSW-M24-ef100-s100                      6282.72      2987.34      9270.06       0.9523     4.500737
HNSW-M24-ef200-s50                       7860.27      2036.46      9896.72       0.9795     6.384675
HNSW-M24-ef200-s75                       7860.27      2770.16     10630.42       0.9872     2.047901
HNSW-M24-ef200-s100                      7860.27      3347.92     11208.19       0.9884     1.540868
HNSW-M24-ef300-s50                       9479.17      2203.89     11683.06       0.9909     4.437885
HNSW-M24-ef300-s75                       9479.17      3059.72     12538.90       0.9939     2.657133
HNSW-M24-ef300-s100                      9479.17      3683.63     13162.80       0.9949     2.165261
HNSW-M32-ef200-s50                      12526.94      2176.31     14703.25       0.9906     5.062128
HNSW-M32-ef200-s75                      12526.94      3336.55     15863.49       0.9931     3.659440
HNSW-M32-ef200-s100                     12526.94      3672.57     16199.52       0.9940     3.158058
HNSW-M32-ef300-s50                      13970.91      2402.67     16373.58       0.9690     6.915551
HNSW-M32-ef300-s75                      13970.91      3492.69     17463.60       0.9749     3.424761
HNSW-M32-ef300-s100                     13970.91      3721.96     17692.87       0.9773     2.026151
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
Exhaustive                                  4.57     25403.83     25408.40       1.0000     0.000000
HNSW-M16-ef50-s50                        2571.50      1447.74      4019.24       0.8785     0.019600
HNSW-M16-ef50-s75                        2571.50      2161.70      4733.19       0.9228     0.013238
HNSW-M16-ef50-s100                       2571.50      2772.29      5343.78       0.9402     0.010435
HNSW-M16-ef100-s50                       3786.68      1627.38      5414.06       0.9421     0.015589
HNSW-M16-ef100-s75                       3786.68      2408.17      6194.85       0.9610     0.009599
HNSW-M16-ef100-s100                      3786.68      2905.29      6691.97       0.9701     0.006626
HNSW-M16-ef200-s50                       4931.96      2053.27      6985.23       0.9565     0.012105
HNSW-M16-ef200-s75                       4931.96      2338.34      7270.29       0.9692     0.005377
HNSW-M16-ef200-s100                      4931.96      3337.83      8269.78       0.9736     0.003202
HNSW-M24-ef100-s50                       6523.06      1932.47      8455.53       0.9317     0.025393
HNSW-M24-ef100-s75                       6523.06      2702.79      9225.84       0.9389     0.022835
HNSW-M24-ef100-s100                      6523.06      3247.83      9770.89       0.9427     0.021139
HNSW-M24-ef200-s50                       7704.75      2149.33      9854.08       0.9211     0.013118
HNSW-M24-ef200-s75                       7704.75      2539.71     10244.46       0.9356     0.005597
HNSW-M24-ef200-s100                      7704.75      3058.88     10763.64       0.9389     0.003964
HNSW-M24-ef300-s50                       9605.09      2108.23     11713.32       0.9540     0.023076
HNSW-M24-ef300-s75                       9605.09      2866.73     12471.81       0.9753     0.010799
HNSW-M24-ef300-s100                      9605.09      3362.17     12967.26       0.9813     0.007274
HNSW-M32-ef200-s50                      12424.73      2541.91     14966.64       0.9084     0.037275
HNSW-M32-ef200-s75                      12424.73      2956.55     15381.28       0.9198     0.030290
HNSW-M32-ef200-s100                     12424.73      3574.42     15999.15       0.9249     0.027301
HNSW-M32-ef300-s50                      13840.60      2282.51     16123.10       0.9660     0.016257
HNSW-M32-ef300-s75                      13840.60      2997.39     16837.99       0.9720     0.013021
HNSW-M32-ef300-s100                     13840.60      3584.61     17425.21       0.9745     0.011981
----------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index. Uses Voronoi cells to sub-partition the original data.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.


**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.41     24485.65     24489.06       1.0000     0.000000
IVF-nl273-np13                           1332.21      3223.10      4555.31       0.9954     0.036065
IVF-nl273-np16                           1332.21      4070.12      5402.33       0.9998     0.001561
IVF-nl273-np23                           1332.21      5655.86      6988.07       1.0000     0.000000
IVF-nl387-np19                           1885.38      3394.14      5279.52       0.9962     0.020320
IVF-nl387-np27                           1885.38      5047.88      6933.26       1.0000     0.000000
IVF-nl547-np23                           2715.44      3037.62      5753.06       0.9905     0.043293
IVF-nl547-np27                           2715.44      3440.96      6156.40       0.9971     0.009402
IVF-nl547-np33                           2715.44      4231.32      6946.76       0.9997     0.000616
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
Exhaustive                                  4.67     25495.86     25500.53       1.0000     0.000000
IVF-nl273-np13                           4028.34      3813.64      7841.98       0.9957     0.000021
IVF-nl273-np16                           4028.34      4055.35      8083.69       0.9998     0.000001
IVF-nl273-np23                           4028.34      5689.40      9717.73       1.0000     0.000000
IVF-nl387-np19                           5812.22      3484.11      9296.33       0.9965     0.000012
IVF-nl387-np27                           5812.22      4857.49     10669.71       1.0000     0.000000
IVF-nl547-np23                           7955.42      3074.57     11029.98       0.9914     0.000027
IVF-nl547-np27                           7955.42      3648.66     11604.08       0.9974     0.000006
IVF-nl547-np33                           7955.42      4332.03     12287.44       0.9997     0.000000
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
Exhaustive                                  3.51     25277.12     25280.62       1.0000     0.000000
LSH-nt10-nb8-s:auto                        68.72      6286.11      6354.83       0.9638     0.480160
LSH-nt10-nb8-s:5k                          68.72      3461.24      3529.96       0.8537     1.571043
LSH-nt20-nb8-s:auto                       125.52     10695.94     10821.46       0.9967     0.038897
LSH-nt20-nb8-s:5k                         125.52      3618.50      3744.02       0.8595     1.469591
LSH-nt25-nb8-s:auto                       148.28     13496.94     13645.22       0.9987     0.014704
LSH-nt25-nb8-s:5k                         148.28      3641.49      3789.77       0.8596     1.469395
LSH-nt10-nb10-s:auto                       82.04      3977.43      4059.46       0.9114     1.283957
LSH-nt10-nb10-s:5k                         82.04      2786.54      2868.58       0.8493     1.675435
LSH-nt20-nb10-s:auto                      149.43      6301.92      6451.35       0.9818     0.229392
LSH-nt20-nb10-s:5k                        149.43      3149.04      3298.47       0.8902     0.944406
LSH-nt25-nb10-s:auto                      171.04      7827.87      7998.90       0.9914     0.104942
LSH-nt25-nb10-s:5k                        171.04      3170.87      3341.91       0.8932     0.897108
LSH-nt10-nb12-s:auto                       93.51      3165.84      3259.35       0.8431     2.506719
LSH-nt10-nb12-s:5k                         93.51      2397.94      2491.45       0.8021     2.687833
LSH-nt20-nb12-s:auto                      174.68      4968.07      5142.75       0.9547     0.632133
LSH-nt20-nb12-s:5k                        174.68      2873.25      3047.93       0.8929     0.977663
LSH-nt25-nb12-s:auto                      218.89      5482.12      5701.01       0.9720     0.373990
LSH-nt25-nb12-s:5k                        218.89      3127.12      3346.01       0.9044     0.780383
LSH-nt10-nb16-s:auto                      142.56      2076.96      2219.52       0.6891     6.654130
LSH-nt10-nb16-s:5k                        142.56      1735.79      1878.35       0.6680     6.721914
LSH-nt20-nb16-s:auto                      273.09      3131.83      3404.92       0.8391     2.829489
LSH-nt20-nb16-s:5k                        273.09      2411.04      2684.13       0.8022     2.961812
LSH-nt25-nb16-s:auto                      298.95      3610.89      3909.84       0.8751     2.077903
LSH-nt25-nb16-s:5k                        298.95      2199.53      2498.48       0.8346     2.227713
LSH-nt50-nb16-s:auto                      610.82      5648.65      6259.48       0.9597     0.585007
LSH-nt50-nb16-s:5k                        610.82      2942.43      3553.25       0.9107     0.793054
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
Exhaustive                                  4.71     25246.81     25251.51       1.0000     0.000000
LSH-nt10-nb8-s:auto                        33.95      9194.68      9228.63       0.9807     0.000164
LSH-nt10-nb8-s:5k                          33.95      3786.69      3820.64       0.8236     0.001536
LSH-nt20-nb8-s:auto                       240.83     16340.34     16581.17       0.9992     0.000006
LSH-nt20-nb8-s:5k                         240.83      4036.03      4276.86       0.8239     0.001533
LSH-nt25-nb8-s:auto                        71.34     19574.58     19645.92       0.9998     0.000002
LSH-nt25-nb8-s:5k                          71.34      4018.83      4090.17       0.8239     0.001533
LSH-nt10-nb10-s:auto                       43.02      5901.33      5944.35       0.9543     0.000426
LSH-nt10-nb10-s:5k                         43.02      3280.28      3323.30       0.8669     0.000950
LSH-nt20-nb10-s:auto                       75.90      8585.23      8661.13       0.9948     0.000042
LSH-nt20-nb10-s:5k                         75.90      3665.60      3741.50       0.8797     0.000791
LSH-nt25-nb10-s:auto                       88.88      9786.34      9875.21       0.9981     0.000016
LSH-nt25-nb10-s:5k                         88.88      3682.90      3771.77       0.8799     0.000789
LSH-nt10-nb12-s:auto                       54.52      3750.15      3804.68       0.9078     0.000924
LSH-nt10-nb12-s:5k                         54.52      2570.94      2625.46       0.8592     0.001114
LSH-nt20-nb12-s:auto                       95.20      6119.94      6215.14       0.9813     0.000162
LSH-nt20-nb12-s:5k                         95.20      2846.66      2941.86       0.9078     0.000521
LSH-nt25-nb12-s:auto                      111.64      6480.08      6591.72       0.9904     0.000084
LSH-nt25-nb12-s:5k                        111.64      2909.10      3020.74       0.9116     0.000480
LSH-nt10-nb16-s:auto                       74.11      2180.71      2254.82       0.7512     0.003265
LSH-nt10-nb16-s:5k                         74.11      1827.45      1901.56       0.7274     0.003324
LSH-nt20-nb16-s:auto                      121.68      3432.00      3553.68       0.9067     0.001003
LSH-nt20-nb16-s:5k                        121.68      2243.54      2365.22       0.8673     0.001112
LSH-nt25-nb16-s:auto                      149.79      4062.47      4212.26       0.9344     0.000677
LSH-nt25-nb16-s:5k                        149.79      2494.87      2644.66       0.8916     0.000801
LSH-nt50-nb16-s:auto                      287.78      5938.67      6226.45       0.9852     0.000139
LSH-nt50-nb16-s:5k                        287.78      3010.85      3298.63       0.9332     0.000314
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
Exhaustive                                  3.87     23929.25     23933.11       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                2021.42      1226.24      3247.65       0.9687     0.189218
NNDescent-nt24-s:auto-dp0                3396.03      1228.06      4624.09       0.9885     0.061129
NNDescent-nt:auto-s75-dp0                3050.66      1737.59      4788.25       0.9939     0.031297
NNDescent-nt:auto-s100-dp0               3050.66      1816.99      4867.65       0.9960     0.020544
NNDescent-nt:auto-s:auto-dp0             3050.66       969.59      4020.25       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.25          2820.79      1163.20      3983.99       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.5           3532.68       982.95      4515.63       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp1             2849.11      1125.53      3974.64       0.9893     0.056484
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
Exhaustive                                  4.63     28409.27     28413.90       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                7270.95       999.62      8270.57       0.9997     0.000001
NNDescent-nt24-s:auto-dp0                6613.81       897.66      7511.47       0.9998     0.000001
NNDescent-nt:auto-s75-dp0                6025.44      1579.33      7604.77       0.9999     0.000000
NNDescent-nt:auto-s100-dp0               6025.44      1840.56      7866.00       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0             6025.44      1073.47      7098.92       0.9997     0.000001
NNDescent-nt:auto-s:auto-dp0.25          6849.90      1072.01      7921.91       0.9899     0.000030
NNDescent-nt:auto-s:auto-dp0.5           6818.67       861.79      7680.46       0.9806     0.000062
NNDescent-nt:auto-s:auto-dp1             6338.33       763.17      7101.50       0.9613     0.000135
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
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

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
Exhaustive                             3.57     24960.24     24963.80       1.0000
IVF-SQ8-nl273-np13                  1572.69       555.18      2127.87       0.8548
IVF-SQ8-nl273-np16                  1572.69       699.63      2272.32       0.8566
IVF-SQ8-nl273-np23                  1572.69       879.88      2452.57       0.8567
IVF-SQ8-nl387-np19                  1935.02       593.86      2528.89       0.8552
IVF-SQ8-nl387-np27                  1935.02       809.99      2745.01       0.8567
IVF-SQ8-nl547-np23                  2752.78       689.26      3442.04       0.8529
IVF-SQ8-nl547-np27                  2752.78       749.43      3502.21       0.8556
IVF-SQ8-nl547-np33                  2752.78       856.76      3609.54       0.8566
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
Exhaustive                            11.07    116822.76    116833.84       1.0000
IVF-SQ8-nl273-np13                  6270.91      2197.15      8468.06       0.8224
IVF-SQ8-nl273-np16                  6270.91      2605.84      8876.75       0.8291
IVF-SQ8-nl273-np23                  6270.91      3616.88      9887.79       0.8345
IVF-SQ8-nl387-np19                  8964.76      2403.18     11367.94       0.8263
IVF-SQ8-nl387-np27                  8964.76      3102.59     12067.35       0.8340
IVF-SQ8-nl547-np23                 13144.41      2574.42     15718.83       0.8184
IVF-SQ8-nl547-np27                 13144.41      2728.36     15872.77       0.8277
IVF-SQ8-nl547-np33                 13144.41      3358.42     16502.83       0.8335
-----------------------------------------------------------------------------------
```

**Data set with stronger correlation structure and more dimensions**

To compare against the next two indices. This data is designed to be better 
suited for high dimensionality. One can appreciate, that with higher 
dimensionality the Recall starts suffering for this
index.

```
===================================================================================
Benchmark: 150k cells, 128D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            14.95    185329.68    185344.63       1.0000
IVF-SQ8-nl273-np13                 10181.62      3100.52     13282.14       0.6413
IVF-SQ8-nl273-np16                 10181.62      3510.16     13691.78       0.6421
IVF-SQ8-nl273-np23                 10181.62      5295.38     15477.00       0.6422
IVF-SQ8-nl387-np19                 14813.81      3271.58     18085.39       0.6419
IVF-SQ8-nl387-np27                 14813.81      4584.85     19398.67       0.6422
IVF-SQ8-nl547-np23                 20138.92      2907.62     23046.54       0.6415
IVF-SQ8-nl547-np27                 20138.92      3355.53     23494.45       0.6420
IVF-SQ8-nl547-np33                 20138.92      3960.62     24099.54       0.6422
-----------------------------------------------------------------------------------
```

Let's check how the two other indices deal with this.

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
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.
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
Exhaustive                            16.01    180361.27    180377.28       1.0000
IVF-PQ-nl273-m16-np13              16781.22      4450.38     21231.60       0.5571
IVF-PQ-nl273-m16-np16              16781.22      5175.86     21957.08       0.5575
IVF-PQ-nl273-m16-np23              16781.22      7891.47     24672.69       0.5575
IVF-PQ-nl273-m32-np13              20016.47      7353.13     27369.60       0.7225
IVF-PQ-nl273-m32-np16              20016.47      8984.63     29001.10       0.7235
IVF-PQ-nl273-m32-np23              20016.47     12319.57     32336.04       0.7237
IVF-PQ-nl387-m16-np19              20712.47      5509.88     26222.35       0.5590
IVF-PQ-nl387-m16-np27              20712.47      7915.02     28627.49       0.5593
IVF-PQ-nl387-m32-np19              24812.90      9081.07     33893.97       0.7250
IVF-PQ-nl387-m32-np27              24812.90     13135.28     37948.18       0.7255
IVF-PQ-nl547-m16-np23              27162.68      6891.43     34054.11       0.5619
IVF-PQ-nl547-m16-np27              27162.68      7580.48     34743.15       0.5622
IVF-PQ-nl547-m16-np33              27162.68      8951.89     36114.57       0.5623
IVF-PQ-nl547-m32-np23              30516.85     10388.61     40905.47       0.7270
IVF-PQ-nl547-m32-np27              30516.85     12168.79     42685.64       0.7277
IVF-PQ-nl547-m32-np33              30516.85     14645.70     45162.56       0.7280
-----------------------------------------------------------------------------------
```

One can appreciate that PQ in this case can yield higher Recalls than just
SQ8 going from ca. 0.64 to 0.72. However, index building and query time are both
higher. However, you also reduce the memory finger print quite substantially.

**With more dimensions**

With 192 dimensions. In this case, also m = 48, i.e., dividing the original 
vectors into 48 subvectors was tested.

```
===================================================================================
Benchmark: 150k cells, 192D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            23.37    308023.77    308047.14       1.0000
IVF-PQ-nl273-m16-np13              25713.83      5433.23     31147.06       0.4529
IVF-PQ-nl273-m16-np16              25713.83      7264.61     32978.44       0.4536
IVF-PQ-nl273-m16-np23              25713.83      9574.54     35288.37       0.4537
IVF-PQ-nl273-m32-np13              31793.39     10472.38     42265.77       0.6285
IVF-PQ-nl273-m32-np16              31793.39     12255.97     44049.36       0.6301
IVF-PQ-nl273-m32-np23              31793.39     17548.37     49341.77       0.6305
IVF-PQ-nl273-m48-np13              32853.65     11373.11     44226.76       0.7318
IVF-PQ-nl273-m48-np16              32853.65     14238.54     47092.19       0.7343
IVF-PQ-nl273-m48-np23              32853.65     20256.78     53110.43       0.7349
IVF-PQ-nl387-m16-np19              34085.74      7788.76     41874.50       0.4550
IVF-PQ-nl387-m16-np27              34085.74     10108.84     44194.58       0.4557
IVF-PQ-nl387-m32-np19              38396.24     13675.44     52071.68       0.6303
IVF-PQ-nl387-m32-np27              38396.24     17223.62     55619.86       0.6317
IVF-PQ-nl387-m48-np19              41160.11     17149.93     58310.05       0.7347
IVF-PQ-nl387-m48-np27              41160.11     21030.83     62190.94       0.7368
IVF-PQ-nl547-m16-np23              44462.83      8506.78     52969.61       0.4570
IVF-PQ-nl547-m16-np27              44462.83      9607.35     54070.17       0.4577
IVF-PQ-nl547-m16-np33              44462.83     11342.21     55805.04       0.4581
IVF-PQ-nl547-m32-np23              46745.07     13209.96     59955.03       0.6305
IVF-PQ-nl547-m32-np27              46745.07     16297.98     63043.05       0.6321
IVF-PQ-nl547-m32-np33              46745.07     20421.76     67166.82       0.6330
IVF-PQ-nl547-m48-np23              48715.88     15664.02     64379.90       0.7331
IVF-PQ-nl547-m48-np27              48715.88     18333.34     67049.22       0.7355
IVF-PQ-nl547-m48-np33              48715.88     23769.50     72485.38       0.7368
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
by the optimised PQ. If in doubt, use the IVF-PQ index.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.
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
Exhaustive                            14.60    181852.46    181867.06       1.0000
IVF-OPQ-nl273-m16-np13             29770.87      4444.53     34215.40       0.5572
IVF-OPQ-nl273-m16-np16             29770.87      5332.95     35103.82       0.5576
IVF-OPQ-nl273-m16-np23             29770.87      7733.49     37504.36       0.5576
IVF-OPQ-nl273-m32-np13             40200.77      7213.76     47414.53       0.7253
IVF-OPQ-nl273-m32-np16             40200.77      9369.07     49569.84       0.7263
IVF-OPQ-nl273-m32-np23             40200.77     12113.81     52314.58       0.7264
IVF-OPQ-nl387-m16-np19             35565.86      6149.77     41715.63       0.5595
IVF-OPQ-nl387-m16-np27             35565.86      8551.83     44117.68       0.5597
IVF-OPQ-nl387-m32-np19             44867.67     12729.91     57597.58       0.7275
IVF-OPQ-nl387-m32-np27             44867.67     14298.33     59166.00       0.7280
IVF-OPQ-nl547-m16-np23             42874.72      8035.92     50910.64       0.5616
IVF-OPQ-nl547-m16-np27             42874.72      8576.86     51451.59       0.5619
IVF-OPQ-nl547-m16-np33             42874.72      9583.35     52458.07       0.5619
IVF-OPQ-nl547-m32-np23             49953.24     10573.52     60526.75       0.7286
IVF-OPQ-nl547-m32-np27             49953.24     11856.71     61809.95       0.7294
IVF-OPQ-nl547-m32-np33             49953.24     15153.34     65106.57       0.7296
-----------------------------------------------------------------------------------
```

**With more dimensions**

With 192 dimensions.

```
===================================================================================
Benchmark: 150k cells, 192D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive                            23.94    300669.42    300693.36       1.0000
IVF-OPQ-nl273-m16-np13             48538.75      5168.65     53707.39       0.4567
IVF-OPQ-nl273-m16-np16             48538.75      6243.53     54782.27       0.4575
IVF-OPQ-nl273-m16-np23             48538.75      8749.50     57288.24       0.4576
IVF-OPQ-nl273-m32-np13             56733.12      8602.57     65335.69       0.6295
IVF-OPQ-nl273-m32-np16             56733.12     10411.57     67144.69       0.6312
IVF-OPQ-nl273-m32-np23             56733.12     14800.91     71534.03       0.6316
IVF-OPQ-nl273-m48-np13             68469.74     10826.48     79296.23       0.7334
IVF-OPQ-nl273-m48-np16             68469.74     13173.62     81643.36       0.7360
IVF-OPQ-nl273-m48-np23             68469.74     18713.54     87183.29       0.7365
IVF-OPQ-nl387-m16-np19             58510.78      7467.02     65977.80       0.4579
IVF-OPQ-nl387-m16-np27             58510.78      9919.46     68430.24       0.4585
IVF-OPQ-nl387-m32-np19             74880.54     11942.58     86823.12       0.6319
IVF-OPQ-nl387-m32-np27             74880.54     16289.07     91169.61       0.6334
IVF-OPQ-nl387-m48-np19             73287.48     13815.35     87102.83       0.7356
IVF-OPQ-nl387-m48-np27             73287.48     19191.27     92478.75       0.7377
IVF-OPQ-nl547-m16-np23             66317.33      8294.84     74612.17       0.4603
IVF-OPQ-nl547-m16-np27             66317.33      9636.94     75954.27       0.4610
IVF-OPQ-nl547-m16-np33             66317.33     11443.52     77760.85       0.4614
IVF-OPQ-nl547-m32-np23             78328.35     13391.88     91720.23       0.6308
IVF-OPQ-nl547-m32-np27             78328.35     15424.57     93752.92       0.6324
IVF-OPQ-nl547-m32-np33             78328.35     18499.00     96827.35       0.6333
IVF-OPQ-nl547-m48-np23             84417.78     15267.71     99685.49       0.7349
IVF-OPQ-nl547-m48-np27             84417.78     17701.44    102119.22       0.7372
IVF-OPQ-nl547-m48-np33             84417.78     21675.29    106093.07       0.7386
-----------------------------------------------------------------------------------
```

## GPU

Two indices are also implemented in GPU-accelerated versions. The exhaustive
search and the IVF index. Under the hood, this uses 
[cubecl](https://github.com/tracel-ai/cubecl) with wgpu backend (system agnostic, 
for details please check [here](https://burn.dev/books/cubecl/getting-started/installation.html)). 
Let's first look at the indices compared against exhaustive (CPU). You can
of course provide other backends.

### Comparison against CPU exhaustive

The GPU acceleration is notable already in the exhaustive index. The IVF-GPU
reaches very fast speeds here.

**Euclidean:**

```
====================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                              3.41     23351.89     23355.30       1.0000     0.000000
GPU-Exhaustive                              4.69     14489.47     14494.16       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   1335.93      3047.53      4383.46       0.9954     0.036068
IVF-GPU-kNN-nl273-np16                   1335.93      3421.05      4756.98       0.9998     0.001564
IVF-GPU-kNN-nl273-np23                   1335.93      4549.04      5884.97       1.0000     0.000003
IVF-GPU-kNN-nl273-np27                   1335.93      5251.19      6587.12       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                   1874.16      3398.09      5272.26       0.9962     0.020323
IVF-GPU-kNN-nl387-np27                   1874.16      4177.51      6051.67       1.0000     0.000003
IVF-GPU-kNN-nl387-np38                   1874.16      5570.25      7444.42       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                   2666.30      3514.38      6180.69       0.9905     0.043296
IVF-GPU-kNN-nl547-np27                   2666.30      3644.23      6310.54       0.9971     0.009404
IVF-GPU-kNN-nl547-np33                   2666.30      4212.87      6879.17       0.9997     0.000619
IVF-GPU-kNN-nl547-np54                   2666.30      5952.71      8619.01       1.0000     0.000003
----------------------------------------------------------------------------------------------------
```

**Cosine:**

```
====================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                              5.38     23923.51     23928.89       1.0000     0.000000
GPU-Exhaustive                              4.59     14661.81     14666.40       1.0000     0.000000
IVF-GPU-kNN-nl273-np13                   3911.37      3105.11      7016.47       0.9956     0.000021
IVF-GPU-kNN-nl273-np16                   3911.37      3485.04      7396.40       0.9998     0.000001
IVF-GPU-kNN-nl273-np23                   3911.37      4644.56      8555.93       1.0000     0.000000
IVF-GPU-kNN-nl273-np27                   3911.37      5304.09      9215.46       1.0000     0.000000
IVF-GPU-kNN-nl387-np19                   5494.76      3276.61      8771.36       0.9965     0.000012
IVF-GPU-kNN-nl387-np27                   5494.76      4256.59      9751.34       1.0000     0.000000
IVF-GPU-kNN-nl387-np38                   5494.76      5410.46     10905.22       1.0000     0.000000
IVF-GPU-kNN-nl547-np23                   7799.18      3458.07     11257.26       0.9913     0.000026
IVF-GPU-kNN-nl547-np27                   7799.18      3670.37     11469.55       0.9973     0.000006
IVF-GPU-kNN-nl547-np33                   7799.18      4144.41     11943.59       0.9997     0.000000
IVF-GPU-kNN-nl547-np54                   7799.18      5778.40     13577.59       1.0000     0.000000
----------------------------------------------------------------------------------------------------
```

### Comparison against IVF CPU

In this case, the IVF CPU implementation is being compared against the GPU 
version. GPU acceleration shines with larger data sets, hence, the number of
samples was increased to 500_000 for these benchmarks.

**IVF CPU:**

```
====================================================================================================
Benchmark: 500k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                 11.83    258458.26    258470.09       1.0000     0.000000
IVF-nl500-np22                           8518.95     47041.19     55560.14       0.9938     0.017566
IVF-nl500-np25                           8518.95     52879.84     61398.78       0.9978     0.004195
IVF-nl500-np31                           8518.95     65425.15     73944.10       0.9998     0.000204
IVF-nl707-np26                          11304.72     37316.38     48621.10       0.9861     0.055255
IVF-nl707-np35                          11304.72     48435.29     59740.01       0.9981     0.005750
IVF-nl707-np37                          11304.72     50956.66     62261.38       0.9989     0.002849
IVF-nl1000-np31                         15980.61     31357.65     47338.26       0.9787     0.093676
IVF-nl1000-np44                         15980.61     43634.73     59615.34       0.9972     0.008673
IVF-nl1000-np50                         15980.61     50370.27     66350.89       0.9992     0.001971
----------------------------------------------------------------------------------------------------
```

**IVF GPU:**

```
====================================================================================================
Benchmark: 500k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             11.21    261010.65    261021.87       1.0000     0.000000
GPU-Exhaustive                             20.81    175500.07    175520.88       1.0000     0.000002
IVF-GPU-kNN-nl500-np22                   7979.49     24180.63     32160.12       0.9938     0.017569
IVF-GPU-kNN-nl500-np25                   7979.49     26730.53     34710.02       0.9978     0.004182
IVF-GPU-kNN-nl500-np31                   7979.49     30677.90     38657.39       0.9998     0.000206
IVF-GPU-kNN-nl500-np50                   7979.49     47844.94     55824.44       1.0000     0.000002
IVF-GPU-kNN-nl707-np26                  11900.13     22641.16     34541.29       0.9861     0.055257
IVF-GPU-kNN-nl707-np35                  11900.13     27722.34     39622.47       0.9981     0.005752
IVF-GPU-kNN-nl707-np37                  11900.13     29293.64     41193.77       0.9989     0.002852
IVF-GPU-kNN-nl707-np70                  11900.13     51075.16     62975.29       1.0000     0.000002
IVF-GPU-kNN-nl1000-np31                 16655.31     20986.76     37642.07       0.9787     0.093678
IVF-GPU-kNN-nl1000-np44                 16655.31     26519.50     43174.81       0.9972     0.008676
IVF-GPU-kNN-nl1000-np50                 16655.31     30444.66     47099.97       0.9992     0.001973
IVF-GPU-kNN-nl1000-np100                16655.31     52896.22     69551.53       1.0000     0.000002
```

### Higher dimensionality

Similar to the quantised indices, this shows 

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