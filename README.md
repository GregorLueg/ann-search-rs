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

## Performance and recommendations

To identify good basic thresholds, there are a set of different gridsearch
scripts available. These can be run via

```bash
# how you would run the annoy benchmark
cargo run --example gridsearch_annoy --release
```

For every index, 250k cells with 24 dimensions distance and 20 distinct clusters 
in the synthetic data has been run. The results for the different  indices are
show below. For details on the synthetic data function, see `/src/synthetic.rs`.

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
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.11     48407.86     48411.97       1.0000     0.000000
Annoy-nt5:auto                       114.88      1598.35      1713.23       0.6684    21.132964
Annoy-nt5:10x                        114.88       955.11      1069.99       0.5058    21.044210
Annoy-nt5:5x                         114.88       578.60       693.48       0.3638    20.917090
Annoy-nt10:auto                      165.92      3080.97      3246.89       0.8727    21.205646
Annoy-nt10:10x                       165.92      1772.27      1938.19       0.7244    21.158445
Annoy-nt10:5x                        165.92      1024.13      1190.06       0.5514    21.075058
Annoy-nt15:auto                      240.21      4492.72      4732.93       0.9484    21.224211
Annoy-nt15:10x                       240.21      2534.21      2774.41       0.8424    21.198283
Annoy-nt15:5x                        240.21      1457.48      1697.69       0.6800    21.141419
Annoy-nt25:auto                      355.14      6642.40      6997.54       0.9907    21.232370
Annoy-nt25:10x                       355.14      3901.51      4256.65       0.9454    21.223984
Annoy-nt25:5x                        355.14      2406.41      2761.54       0.8325    21.196221
Annoy-nt50:auto                      683.14     13599.26     14282.39       0.9998    21.233759
Annoy-nt50:10x                       683.14      8703.74      9386.88       0.9950    21.233124
Annoy-nt50:5x                        683.14      5304.67      5987.81       0.9619    21.227610
Annoy-nt75:auto                     1059.01     18598.91     19657.91       1.0000    21.233786
Annoy-nt75:10x                      1059.01     12338.75     13397.76       0.9994    21.233721
Annoy-nt75:5x                       1059.01      8098.23      9157.23       0.9900    21.232444
-----------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.70     49962.10     49967.79       1.0000     0.000000
Annoy-nt5:auto                       109.93       630.61       740.55       0.3816     0.006956
Annoy-nt5:10x                        109.93       630.72       740.65       0.3813     0.006967
Annoy-nt5:5x                         109.93       570.61       680.54       0.3561     0.007672
Annoy-nt10:auto                      152.38      1113.70      1266.08       0.5727     0.003337
Annoy-nt10:10x                       152.38      1142.48      1294.86       0.5726     0.003339
Annoy-nt10:5x                        152.38      1029.75      1182.13       0.5482     0.003699
Annoy-nt15:auto                      233.36      1527.71      1761.07       0.7004     0.001889
Annoy-nt15:10x                       233.36      1549.41      1782.76       0.7003     0.001890
Annoy-nt15:5x                        233.36      1458.31      1691.67       0.6798     0.002110
Annoy-nt25:auto                      366.51      2514.32      2880.83       0.8483     0.000735
Annoy-nt25:10x                       366.51      2510.49      2877.00       0.8483     0.000736
Annoy-nt25:5x                        366.51      2425.16      2791.67       0.8351     0.000835
Annoy-nt50:auto                      676.90      5049.30      5726.19       0.9683     0.000108
Annoy-nt50:10x                       676.90      5148.86      5825.76       0.9683     0.000108
Annoy-nt50:5x                        676.90      4890.24      5567.13       0.9645     0.000127
Annoy-nt75:auto                     1012.87      7517.70      8530.57       0.9924     0.000021
Annoy-nt75:10x                      1012.87      7716.54      8729.41       0.9924     0.000021
Annoy-nt75:5x                       1012.87      7610.39      8623.26       0.9913     0.000025
-----------------------------------------------------------------------------------------------
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
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.37     47732.48     47736.85       1.0000     0.000000
HNSW-M16-ef50-s50                   3463.79      2758.91      6222.70       0.9360     0.332254
HNSW-M16-ef50-s75                   3463.79      3737.88      7201.66       0.9584     0.202089
HNSW-M16-ef50-s100                  3463.79      4563.33      8027.11       0.9691     0.148079
HNSW-M16-ef100-s50                  5632.73      2799.97      8432.70       0.9742     0.722915
HNSW-M16-ef100-s75                  5632.73      3891.27      9523.99       0.9852     0.367588
HNSW-M16-ef100-s100                 5632.73      4809.55     10442.27       0.9903     0.190140
HNSW-M16-ef200-s50                  7217.15      2945.24     10162.39       0.9370     7.428632
HNSW-M16-ef200-s75                  7217.15      4067.76     11284.91       0.9464     6.144968
HNSW-M16-ef200-s100                 7217.15      4981.07     12198.22       0.9511     5.521321
HNSW-M24-ef100-s50                  9447.68      3580.37     13028.05       0.9916     0.364111
HNSW-M24-ef100-s75                  9447.68      4800.85     14248.53       0.9958     0.113064
HNSW-M24-ef100-s100                 9447.68      5909.70     15357.37       0.9972     0.066088
HNSW-M24-ef200-s50                 12092.23      3906.15     15998.38       0.9667     5.022693
HNSW-M24-ef200-s75                 12092.23      5382.22     17474.44       0.9763     3.509496
HNSW-M24-ef200-s100                12092.23      6703.40     18795.63       0.9808     2.840705
HNSW-M24-ef300-s50                 14493.20      4054.99     18548.19       0.9898     2.004376
HNSW-M24-ef300-s75                 14493.20      5509.26     20002.46       0.9968     0.498183
HNSW-M24-ef300-s100                14493.20      7643.57     22136.77       0.9985     0.179004
HNSW-M32-ef200-s50                 20188.11      4868.94     25057.04       0.9932     0.784239
HNSW-M32-ef200-s75                 20188.11      5722.35     25910.45       0.9966     0.329492
HNSW-M32-ef200-s100                20188.11      6671.82     26859.92       0.9979     0.183148
HNSW-M32-ef300-s50                 19730.27      4206.83     23937.09       0.9768     6.425630
HNSW-M32-ef300-s75                 19730.27      5720.62     25450.88       0.9924     1.937420
HNSW-M32-ef300-s100                19730.27      6958.42     26688.69       0.9967     0.720552
-----------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for HSNW.

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.42     52076.03     52081.45       1.0000     0.000000
HNSW-M16-ef50-s50                   3439.37      2581.94      6021.31       0.8991     0.003204
HNSW-M16-ef50-s75                   3439.37      3445.54      6884.90       0.9303     0.002194
HNSW-M16-ef50-s100                  3439.37      4614.69      8054.05       0.9522     0.001371
HNSW-M16-ef100-s50                  5399.88      2855.16      8255.04       0.9769     0.000731
HNSW-M16-ef100-s75                  5399.88      3767.81      9167.69       0.9840     0.000365
HNSW-M16-ef100-s100                 5399.88      4629.67     10029.55       0.9870     0.000256
HNSW-M16-ef200-s50                  7992.32      2962.06     10954.39       0.9873     0.003984
HNSW-M16-ef200-s75                  7992.32      4033.47     12025.80       0.9945     0.001314
HNSW-M16-ef200-s100                 7992.32      4971.52     12963.85       0.9966     0.000668
HNSW-M24-ef100-s50                  9488.36      3479.70     12968.07       0.9691     0.001490
HNSW-M24-ef100-s75                  9488.36      4678.12     14166.49       0.9830     0.000712
HNSW-M24-ef100-s100                 9488.36      5753.96     15242.33       0.9891     0.000466
HNSW-M24-ef200-s50                 11623.08      3432.57     15055.64       0.9884     0.003003
HNSW-M24-ef200-s75                 11623.08      4655.47     16278.54       0.9939     0.001000
HNSW-M24-ef200-s100                11623.08      5722.85     17345.92       0.9955     0.000611
HNSW-M24-ef300-s50                 13787.98      3611.82     17399.80       0.9756     0.010737
HNSW-M24-ef300-s75                 13787.98      4798.03     18586.01       0.9860     0.005767
HNSW-M24-ef300-s100                13787.98      6011.51     19799.49       0.9897     0.004028
HNSW-M32-ef200-s50                 17642.82      4110.90     21753.72       0.9888     0.006818
HNSW-M32-ef200-s75                 17642.82      5538.18     23181.00       0.9946     0.002555
HNSW-M32-ef200-s100                17642.82      6708.08     24350.90       0.9960     0.001624
HNSW-M32-ef300-s50                 19777.36      4157.87     23935.23       0.9855     0.008224
HNSW-M32-ef300-s75                 19777.36      5562.48     25339.84       0.9955     0.002090
HNSW-M32-ef300-s100                19777.36      6757.88     26535.24       0.9979     0.000729
-----------------------------------------------------------------------------------------------
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
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.25     48067.39     48071.65       1.0000     0.000000
IVF-nl10-np1                         107.82     18886.55     18994.37       0.9997     0.016862
IVF-nl20-np1                         176.83     11571.22     11748.05       0.9606     0.199568
IVF-nl20-np2                         176.83     21753.79     21930.62       1.0000     0.000056
IVF-nl20-np3                         176.83     32442.97     32619.80       1.0000     0.000045
IVF-nl25-np1                         272.79      8921.22      9194.00       0.9453     0.346001
IVF-nl25-np2                         272.79     15905.97     16178.76       0.9956     0.029999
IVF-nl25-np3                         272.79     21834.62     22107.41       1.0000     0.000105
IVF-nl50-np2                         467.66      8434.93      8902.58       0.9307     0.370176
IVF-nl50-np5                         467.66     17537.65     18005.31       0.9982     0.010276
IVF-nl50-np7                         467.66     23519.25     23986.91       1.0000     0.000044
IVF-nl100-np5                       1125.33     10286.87     11412.20       0.9733     0.132915
IVF-nl100-np10                      1125.33     18740.76     19866.09       0.9994     0.003194
IVF-nl100-np15                      1125.33     27450.31     28575.64       1.0000     0.000007
-----------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF.

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.79     48560.73     48566.52       1.0000     0.000000
IVF-nl10-np1                         240.14     20406.02     20646.16       0.9940     0.000319
IVF-nl20-np1                         425.35     10569.43     10994.78       0.9750     0.000217
IVF-nl20-np2                         425.35     19665.65     20091.00       1.0000     0.000003
IVF-nl20-np3                         425.35     29417.70     29843.05       1.0000     0.000000
IVF-nl25-np1                         599.06      9867.79     10466.85       0.9422     0.000715
IVF-nl25-np2                         599.06     16919.43     17518.49       0.9930     0.000114
IVF-nl25-np3                         599.06     23942.39     24541.45       0.9979     0.000030
IVF-nl50-np2                        1205.31      9188.56     10393.87       0.9146     0.000797
IVF-nl50-np5                        1205.31     20376.21     21581.52       0.9991     0.000005
IVF-nl50-np7                        1205.31     28592.98     29798.30       1.0000     0.000000
IVF-nl100-np5                       2847.38     10123.24     12970.62       0.9750     0.000263
IVF-nl100-np10                      2847.38     18625.97     21473.35       0.9986     0.000014
IVF-nl100-np15                      2847.38     28059.77     30907.15       1.0000     0.000000
-----------------------------------------------------------------------------------------------
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
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             3.87     48708.23     48712.11       1.0000     0.000000
LSH-nt10-bits8:auto                   95.24     22779.13     22874.37       0.9708     0.154976
LSH-nt10-bits8:5k_cand                95.24      8483.09      8578.33       0.7479     1.071855
LSH-nt20-bits8:auto                  153.63     36309.78     36463.41       0.9965     0.017869
LSH-nt20-bits8:5k_cand               153.63      8945.27      9098.90       0.7495     1.056109
LSH-nt25-bits8:auto                  176.27     40446.77     40623.04       0.9985     0.007605
LSH-nt25-bits8:5k_cand               176.27      8314.34      8490.60       0.7495     1.055960
LSH-nt10-bits10:auto                  95.56     13651.71     13747.27       0.9419     0.321709
LSH-nt10-bits10:5k_cand               95.56      6698.84      6794.40       0.7979     0.794104
LSH-nt20-bits10:auto                 166.76     19307.37     19474.13       0.9868     0.071297
LSH-nt20-bits10:5k_cand              166.76      7244.06      7410.82       0.8115     0.667058
LSH-nt25-bits10:auto                 220.01     21989.03     22209.04       0.9931     0.036092
LSH-nt25-bits10:5k_cand              220.01      6930.49      7150.50       0.8130     0.653911
LSH-nt10-bits12:auto                 112.85      9336.12      9448.97       0.8807     0.765442
LSH-nt10-bits12:5k_cand              112.85      5649.20      5762.05       0.7965     0.968725
LSH-nt20-bits12:auto                 203.59     13728.19     13931.78       0.9637     0.215157
LSH-nt20-bits12:5k_cand              203.59      6213.47      6417.06       0.8458     0.521710
LSH-nt25-bits12:auto                 265.38     15275.76     15541.14       0.9768     0.136619
LSH-nt25-bits12:5k_cand              265.38      6305.97      6571.36       0.8522     0.464855
LSH-nt10-bits16:auto                 162.06      5239.83      5401.89       0.7397     2.226454
LSH-nt10-bits16:5k_cand              162.06      3579.47      3741.52       0.6888     2.301114
LSH-nt20-bits16:auto                 276.20      8202.28      8478.48       0.8722     0.962165
LSH-nt20-bits16:5k_cand              276.20      4409.69      4685.89       0.7931     1.097979
LSH-nt25-bits16:auto                 363.23      9613.84      9977.07       0.9058     0.710223
LSH-nt25-bits16:5k_cand              363.23      4634.24      4997.48       0.8172     0.869836
LSH-nt50-bits16:auto                 678.97     14703.57     15382.54       0.9653     0.245996
LSH-nt50-bits16:5k_cand              678.97      5197.62      5876.59       0.8602     0.447244
-----------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for LSH.

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.35     50237.57     50242.92       1.0000     0.000000
LSH-nt10-bits8:auto                   37.57     27497.59     27535.16       0.9892     0.000102
LSH-nt10-bits8:5k_cand                37.57      9170.96      9208.53       0.7612     0.002292
LSH-nt20-bits8:auto                   64.35     43829.13     43893.47       0.9993     0.000007
LSH-nt20-bits8:5k_cand                64.35      9295.49      9359.84       0.7612     0.002291
LSH-nt25-bits8:auto                   75.26     51081.88     51157.14       0.9998     0.000002
LSH-nt25-bits8:5k_cand                75.26      9327.17      9402.43       0.7612     0.002291
LSH-nt10-bits10:auto                  40.41     16174.04     16214.45       0.9682     0.000329
LSH-nt10-bits10:5k_cand               40.41      7547.83      7588.24       0.8240     0.001301
LSH-nt20-bits10:auto                  73.84     24774.18     24848.01       0.9963     0.000040
LSH-nt20-bits10:5k_cand               73.84      7744.89      7818.72       0.8277     0.001232
LSH-nt25-bits10:auto                  91.24     28689.08     28780.32       0.9983     0.000018
LSH-nt25-bits10:5k_cand               91.24      7656.23      7747.47       0.8278     0.001230
LSH-nt10-bits12:auto                  53.21     11160.31     11213.52       0.9297     0.000776
LSH-nt10-bits12:5k_cand               53.21      6079.85      6133.06       0.8125     0.001401
LSH-nt20-bits12:auto                  88.58     16696.03     16784.61       0.9866     0.000150
LSH-nt20-bits12:5k_cand               88.58      6314.81      6403.39       0.8354     0.001015
LSH-nt25-bits12:auto                 112.35     18868.03     18980.38       0.9931     0.000078
LSH-nt25-bits12:5k_cand              112.35      6287.82      6400.17       0.8373     0.000981
LSH-nt10-bits16:auto                  66.11      6124.92      6191.03       0.8019     0.002861
LSH-nt10-bits16:5k_cand               66.11      4117.67      4183.78       0.7349     0.003099
LSH-nt20-bits16:auto                 133.22      9654.24      9787.47       0.9260     0.000961
LSH-nt20-bits16:5k_cand              133.22      4750.93      4884.15       0.8289     0.001334
LSH-nt25-bits16:auto                 149.00     11346.46     11495.46       0.9513     0.000632
LSH-nt25-bits16:5k_cand              149.00      5089.68      5238.68       0.8462     0.001047
LSH-nt50-bits16:auto                 273.39     17352.20     17625.59       0.9887     0.000146
LSH-nt50-bits16:5k_cand              273.39      5472.02      5745.41       0.8716     0.000629
-----------------------------------------------------------------------------------------------
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
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.07     46454.34     46458.41       1.0000     0.000000
NNDescent-nt12-s:auto-dp0           2756.21      1229.22      3985.43       0.9353     0.141336
NNDescent-nt24-s:auto-dp0           4078.37      1153.80      5232.18       0.9775     0.041659
NNDescent-nt:auto-s50-dp0           4134.42      1649.71      5784.13       0.9914     0.015461
NNDescent-nt:auto-s100-dp0          4134.42      2979.43      7113.85       0.9969     0.005887
NNDescent-nt:auto-s:auto-dp0        4134.42      1156.48      5290.90       0.9820     0.032240
NNDescent-nt:auto-s:auto-dp5        4342.65      1129.49      5472.14       0.9820     0.032240
NNDescent-nt:auto-s:auto-dp1        4287.96      1181.62      5469.58       0.9820     0.032240
-----------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for NNDescent
implementation in this `crate`.

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.98     49187.83     49193.80       1.0000     0.000000
NNDescent-nt12-s:auto-dp0           9849.49       990.76     10840.25       0.9995     0.000002
NNDescent-nt24-s:auto-dp0           9642.54       988.41     10630.95       0.9996     0.000001
NNDescent-nt:auto-s50-dp0           9987.02      1740.59     11727.61       0.9997     0.000001
NNDescent-nt:auto-s100-dp0          9987.02      3315.32     13302.34       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0        9987.02      1059.23     11046.25       0.9996     0.000001
NNDescent-nt:auto-s:auto-dp5       10846.29       855.71     11702.00       0.9677     0.000109
NNDescent-nt:auto-s:auto-dp1        9997.64       799.59     10797.23       0.9336     0.000254
-----------------------------------------------------------------------------------------------
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
Benchmark: 250k cells, 24D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive-IP                          4.63     47453.27     47457.90       1.0000
IVF-SQ8-nl10-np1                     114.19      4721.65      4835.83       0.8536
IVF-SQ8-nl20-np1                     181.33      2461.08      2642.41       0.8289
IVF-SQ8-nl20-np2                     181.33      4458.71      4640.03       0.8539
IVF-SQ8-nl20-np3                     181.33      7420.09      7601.42       0.8539
IVF-SQ8-nl25-np1                     254.08      2202.96      2457.04       0.8184
IVF-SQ8-nl25-np2                     254.08      4053.62      4307.70       0.8515
IVF-SQ8-nl25-np3                     254.08      5897.09      6151.17       0.8539
IVF-SQ8-nl50-np2                     465.16      1876.05      2341.21       0.8128
IVF-SQ8-nl50-np5                     465.16      3756.93      4222.09       0.8530
IVF-SQ8-nl50-np7                     465.16      5017.77      5482.93       0.8539
IVF-SQ8-nl100-np5                   1072.23      2319.90      3392.13       0.8403
IVF-SQ8-nl100-np10                  1072.23      4021.30      5093.54       0.8536
IVF-SQ8-nl100-np15                  1072.23      5743.33      6815.56       0.8539
-----------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF with SQ8
quantisation. To note, the mean distance error is not calculated, as the index
does not store the original vectors anymore to reduce memory fingerprint. 

```
===================================================================================
Benchmark: 250k cells, 24D
===================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k
-----------------------------------------------------------------------------------
Exhaustive-IP                          5.60     49426.93     49432.53       1.0000
IVF-SQ8-nl10-np1                     287.72      4917.78      5205.50       0.8766
IVF-SQ8-nl20-np1                     434.91      2712.73      3147.64       0.8647
IVF-SQ8-nl20-np2                     434.91      4891.31      5326.21       0.8820
IVF-SQ8-nl20-np3                     434.91      7354.64      7789.55       0.8820
IVF-SQ8-nl25-np1                     615.12      2669.36      3284.49       0.8413
IVF-SQ8-nl25-np2                     615.12      4384.72      4999.84       0.8775
IVF-SQ8-nl25-np3                     615.12      6083.93      6699.05       0.8809
IVF-SQ8-nl50-np2                    1176.95      2458.37      3635.32       0.8280
IVF-SQ8-nl50-np5                    1176.95      5530.90      6707.85       0.8816
IVF-SQ8-nl50-np7                    1176.95      7548.35      8725.30       0.8820
IVF-SQ8-nl100-np5                   2851.03      2908.22      5759.25       0.8679
IVF-SQ8-nl100-np10                  2851.03      5383.15      8234.18       0.8813
IVF-SQ8-nl100-np15                  2851.03      8345.92     11196.95       0.8820
-----------------------------------------------------------------------------------
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