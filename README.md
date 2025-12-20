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
  - [**IVF (Inverted File index)**]
    

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

To identify good basic thresholds, there are a set of different gridsearch
scripts available. These can be run via

```bash
cargo run --example gridsearch_annoy --release
```

For example for Annoy. 
For every index, 250k cells with 16 dimensions, Euclidean distance and 20 
distinct clusters in the synthetic data has been run. The results for the 
different indices are show below. For details on the synthetic data function,
see `/src/synthetic.rs`.

### Annoy

50 to 75 trees are already sufficient to achieve very high Recalls@k, while
being substantially faster than an exhaustive search. The search_budget is 
set as default to `k * n_trees * 20` which is quite a large one. This is 
reflected in `:auto`. Smaller multipliers with `5x` and `10x` are also shown.
Overall, index generation is very fast (highly parallelisable), but the search
budget needs to be quite decent to get good Recall@k (especially with few 
trees). The more trees you use the more you can reduce the search budget per 
given tree. It depends on your use case. Good allrounder. 

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             5.47     48889.21     48894.68       1.0000     0.000000
Annoy-nt5:auto                       131.42      1870.45      2001.87       0.6666    18.479546
Annoy-nt5:10x                        131.42      1049.30      1180.72       0.5044    18.397135
Annoy-nt5:5x                         131.42       678.13       809.55       0.3631    18.278673
Annoy-nt10:auto                      151.87      3261.84      3413.71       0.8701    18.546778
Annoy-nt10:10x                       151.87      1963.74      2115.61       0.7219    18.503052
Annoy-nt10:5x                        151.87      1174.95      1326.82       0.5498    18.425594
Annoy-nt15:auto                      245.74      4462.78      4708.51       0.9464    18.563919
Annoy-nt15:10x                       245.74      2640.62      2886.36       0.8393    18.539768
Annoy-nt15:5x                        245.74      1591.63      1837.36       0.6777    18.486980
Annoy-nt25:auto                      366.04      7269.50      7635.54       0.9897    18.571518
Annoy-nt25:10x                       366.04      4326.31      4692.35       0.9430    18.563623
Annoy-nt25:5x                        366.04      2793.13      3159.17       0.8293    18.537578
Annoy-nt50:auto                      720.93     12517.90     13238.83       0.9997    18.572859
Annoy-nt50:10x                       720.93      8236.70      8957.63       0.9945    18.572242
Annoy-nt50:5x                        720.93      5211.30      5932.23       0.9601    18.566994
Annoy-nt75:auto                     1100.16     18775.37     19875.53       1.0000    18.572887
Annoy-nt75:10x                      1100.16     12251.45     13351.61       0.9993    18.572818
Annoy-nt75:5x                       1100.16      8089.08      9189.24       0.9893    18.571595
-----------------------------------------------------------------------------------------------
```

### HNSW

HNSW has a trade off between `m` (connections between layers) and the 
`ef_construction` (the budget to generate good connections during construction).
One can appreciate, that higher `m` warrants bigger construction budgets. 
Overall, index generation takes a bit longer, but the query speed is (very) 
high with great recalls (if you took the time to generate the index).

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.25     48958.06     48962.31       1.0000     0.000000
HNSW-M16-ef50-s50                   3425.57      2511.09      5936.66       0.8428     1.972127
HNSW-M16-ef50-s75                   3425.57      3416.58      6842.16       0.8862     1.262422
HNSW-M16-ef50-s100                  3425.57      4313.91      7739.49       0.9122     0.756509
HNSW-M16-ef100-s50                  5101.27      2625.06      7726.33       0.9717     0.662756
HNSW-M16-ef100-s75                  5101.27      3610.14      8711.41       0.9837     0.340288
HNSW-M16-ef100-s100                 5101.27      4489.27      9590.54       0.9885     0.231304
HNSW-M16-ef200-s50                  7186.49      2926.20     10112.68       0.9895     0.670834
HNSW-M16-ef200-s75                  7186.49      3932.64     11119.13       0.9947     0.276285
HNSW-M16-ef200-s100                 7186.49      4817.12     12003.61       0.9964     0.166569
HNSW-M24-ef100-s50                 10123.92      3428.64     13552.56       0.9897     0.394810
HNSW-M24-ef100-s75                 10123.92      4658.60     14782.52       0.9941     0.232457
HNSW-M24-ef100-s100                10123.92      5766.82     15890.74       0.9961     0.156487
HNSW-M24-ef200-s50                 12274.43      3412.67     15687.10       0.9921     0.954220
HNSW-M24-ef200-s75                 12274.43      4504.35     16778.78       0.9961     0.314994
HNSW-M24-ef200-s100                12274.43      5576.27     17850.70       0.9973     0.162504
HNSW-M24-ef300-s50                 14440.63      3658.80     18099.43       0.9764     5.282342
HNSW-M24-ef300-s75                 14440.63      4857.09     19297.71       0.9920     1.404960
HNSW-M24-ef300-s100                14440.63      6087.41     20528.04       0.9958     0.550391
HNSW-M32-ef200-s50                 17663.37      4256.09     21919.46       0.9917     1.500449
HNSW-M32-ef200-s75                 17663.37      5657.45     23320.82       0.9969     0.429996
HNSW-M32-ef200-s100                17663.37      6749.17     24412.54       0.9979     0.250094
HNSW-M32-ef300-s50                 19569.94      4074.35     23644.29       0.9953     0.812227
HNSW-M32-ef300-s75                 19569.94      5512.40     25082.34       0.9977     0.346202
HNSW-M32-ef300-s100                19569.94      6804.02     26373.96       0.9984     0.224769
-----------------------------------------------------------------------------------------------
```

### IVF

Inverted file index (powering for example some of the FAISS indices) is very
powerful. Quick index build, quite fast querying times. The number of lists
(especially with this synthetic data) does not need to be particularly high and
you reach quite quickly 5x better speeds over an exhaustive search. Larger
number of lists or points to search do not really make sense (at least not
in this data).

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.40     50378.56     50382.96       1.0000     0.000000
IVF-nl125-np6                       1528.05     11537.41     13065.46       0.9674     0.166553
IVF-nl125-np12                      1528.05     19755.48     21283.54       0.9976     0.010936
IVF-nl125-np18                      1528.05     28816.68     30344.74       1.0000     0.000001
IVF-nl125-np25                      1528.05     39247.82     40775.87       1.0000     0.000000
IVF-nl250-np12                      1556.02      9036.22     10592.24       0.9855     0.021989
IVF-nl250-np25                      1556.02     17516.19     19072.21       1.0000     0.000000
IVF-nl250-np37                      1556.02     25639.35     27195.37       1.0000     0.000000
IVF-nl250-np50                      1556.02     35206.95     36762.97       1.0000     0.000000
IVF-nl500-np25                      2983.81      9243.05     12226.86       0.9962     0.004162
IVF-nl500-np50                      2983.81     17357.36     20341.17       1.0000     0.000001
IVF-nl500-np75                      2983.81     25665.48     28649.30       1.0000     0.000000
IVF-nl500-np100                     2983.81     35359.74     38343.55       1.0000     0.000000
IVF-nl750-np37                      4407.84      8713.03     13120.87       0.9980     0.002129
IVF-nl750-np75                      4407.84     17639.77     22047.61       1.0000     0.000000
IVF-nl750-np112                     4407.84     25861.45     30269.29       1.0000     0.000000
IVF-nl750-np150                     4407.84     36205.97     40613.81       1.0000     0.000000
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

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             3.97     51177.20     51181.17       1.0000     0.000000
LSH-nt10-bits8:auto                   74.78     23653.72     23728.50       0.9991     0.000912
LSH-nt10-bits8:1k_cand                74.78      6883.59      6958.37       0.6464     0.727822
LSH-nt10-bits8:5k_cand                74.78     10323.67     10398.45       0.8211     0.280555
LSH-nt20-bits8:auto                  134.60     40225.76     40360.36       1.0000     0.000003
LSH-nt20-bits8:1k_cand               134.60      7359.67      7494.27       0.6464     0.727822
LSH-nt20-bits8:5k_cand               134.60     10527.92     10662.52       0.8211     0.280553
LSH-nt50-bits8:auto                  323.20     72161.10     72484.30       1.0000     0.000000
LSH-nt50-bits8:1k_cand               323.20      7968.71      8291.91       0.6464     0.727822
LSH-nt50-bits8:5k_cand               323.20     11724.32     12047.52       0.8211     0.280553
LSH-nt10-bits10:auto                 100.83     19523.89     19624.72       0.9944     0.005796
LSH-nt10-bits10:1k_cand              100.83      4624.02      4724.85       0.5795     0.845285
LSH-nt10-bits10:5k_cand              100.83      9614.71      9715.54       0.8302     0.235886
LSH-nt20-bits10:auto                 190.78     26238.96     26429.74       0.9999     0.000047
LSH-nt20-bits10:1k_cand              190.78      4646.93      4837.71       0.5795     0.845285
LSH-nt20-bits10:5k_cand              190.78      9157.03      9347.80       0.8303     0.235670
LSH-nt50-bits10:auto                 438.78     46161.91     46600.70       1.0000     0.000000
LSH-nt50-bits10:1k_cand              438.78      5007.46      5446.24       0.5795     0.845285
LSH-nt50-bits10:5k_cand              438.78      8452.51      8891.29       0.8303     0.235670
LSH-nt10-bits12:auto                 113.00     16184.20     16297.20       0.9868     0.013963
LSH-nt10-bits12:1k_cand              113.00      3978.72      4091.72       0.5747     0.812246
LSH-nt10-bits12:5k_cand              113.00      8840.11      8953.11       0.8390     0.207945
LSH-nt20-bits12:auto                 301.81     24404.15     24705.96       0.9997     0.000292
LSH-nt20-bits12:1k_cand              301.81      4609.76      4911.57       0.5747     0.812238
LSH-nt20-bits12:5k_cand              301.81      9199.11      9500.92       0.8395     0.207061
LSH-nt50-bits12:auto                 572.11     35503.09     36075.20       1.0000     0.000000
LSH-nt50-bits12:1k_cand              572.11      4046.00      4618.11       0.5747     0.812238
LSH-nt50-bits12:5k_cand              572.11      7744.16      8316.27       0.8395     0.207060
LSH-nt10-bits16:auto                 144.68      9583.90      9728.58       0.9363     0.076193
LSH-nt10-bits16:1k_cand              144.68      3247.20      3391.88       0.5577     0.809636
LSH-nt10-bits16:5k_cand              144.68      6195.91      6340.59       0.8350     0.207271
LSH-nt20-bits16:auto                 299.07     15939.52     16238.59       0.9933     0.006674
LSH-nt20-bits16:1k_cand              299.07      3121.49      3420.56       0.5579     0.808608
LSH-nt20-bits16:5k_cand              299.07      6368.59      6667.66       0.8486     0.184229
LSH-nt50-bits16:auto                 665.09     22580.94     23246.03       1.0000     0.000038
LSH-nt50-bits16:1k_cand              665.09      3134.13      3799.22       0.5579     0.808608
LSH-nt50-bits16:5k_cand              665.09      6791.05      7456.13       0.8489     0.183742
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

```
===============================================================================================
Benchmark: 250k cells, 24D
===============================================================================================
Method                           Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
-----------------------------------------------------------------------------------------------
Exhaustive                             4.74     47357.37     47362.12       1.0000     0.000000
NNDescent-nt12-s:auto-dp0           2741.71      1225.99      3967.70       0.9384     0.053351
NNDescent-nt24-s:auto-dp0           3881.33      1152.48      5033.81       0.9792     0.015214
NNDescent-nt:auto-s50-dp0           4140.49      1622.34      5762.83       0.9921     0.005539
NNDescent-nt:auto-s100-dp0          4140.49      2973.66      7114.15       0.9972     0.001995
NNDescent-nt:auto-s:auto-dp0        4140.49      1127.51      5268.00       0.9834     0.011777
NNDescent-nt:auto-s:auto-dp5        4301.94      1109.89      5411.83       0.9834     0.011777
NNDescent-nt:auto-s:auto-dp1        4320.66      1154.39      5475.05       0.9834     0.011777
-----------------------------------------------------------------------------------------------
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