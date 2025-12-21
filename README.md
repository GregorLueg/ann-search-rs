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
  - [**FANNG (Fast Approximate Nearest Neighbour Graphs)**](https://openaccess.thecvf.com/content_cvpr_2016/papers/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.pdf).
  A version with some modifications in terms of starting node generation and
  some parallel operations in the index generation for speed purposes. This
  version is still being actively developed and optimised.
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
Exhaustive                             3.88     48661.51     48665.39       1.0000     0.000000
Annoy-nt5:auto                       115.34      1652.27      1767.61       0.6834     7.135329
Annoy-nt5:10x                        115.34      1062.38      1177.72       0.5137     7.075608
Annoy-nt5:5x                         115.34       584.78       700.12       0.3686     6.991708
Annoy-nt10:auto                      145.37      3183.88      3329.25       0.8832     7.179861
Annoy-nt10:10x                       145.37      1807.85      1953.22       0.7337     7.149232
Annoy-nt10:5x                        145.37      1108.62      1253.99       0.5584     7.094616
Annoy-nt15:auto                      240.72      4445.10      4685.82       0.9548     7.190838
Annoy-nt15:10x                       240.72      2544.13      2784.85       0.8505     7.174453
Annoy-nt15:5x                        240.72      1534.44      1775.16       0.6877     7.137555
Annoy-nt25:auto                      365.13      6666.41      7031.54       0.9925     7.195367
Annoy-nt25:10x                       365.13      3864.00      4229.13       0.9501     7.190379
Annoy-nt25:5x                        365.13      2421.96      2787.09       0.8392     7.172716
Annoy-nt50:auto                      658.58     12028.31     12686.88       0.9999     7.196043
Annoy-nt50:10x                       658.58      8284.36      8942.93       0.9958     7.195712
Annoy-nt50:5x                        658.58      5082.28      5740.85       0.9653     7.192433
Annoy-nt75:auto                     1000.28     17103.93     18104.22       1.0000     7.196051
Annoy-nt75:10x                      1000.28     11361.13     12361.41       0.9995     7.196022
Annoy-nt75:5x                       1000.28      7527.20      8527.48       0.9913     7.195303
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
Exhaustive                             4.56     46382.07     46386.64       1.0000     0.000000
HNSW-M16-ef50-s50                   3481.02      4048.75      7529.77       0.8253     1.587356
HNSW-M16-ef50-s75                   3481.02      7269.76     10750.78       0.8661     0.984694
HNSW-M16-ef50-s100                  3481.02      9991.19     13472.21       0.9006     0.566272
HNSW-M16-ef100-s50                  6590.19      4145.56     10735.75       0.9386     1.504171
HNSW-M16-ef100-s75                  6590.19      7416.22     14006.40       0.9593     0.567458
HNSW-M16-ef100-s100                 6590.19     10135.90     16726.09       0.9714     0.352226
HNSW-M16-ef200-s50                 12758.50      4517.95     17276.45       0.9854     1.675700
HNSW-M16-ef200-s75                 12758.50      8065.47     20823.97       0.9938     0.507523
HNSW-M16-ef200-s100                12758.50     10934.03     23692.53       0.9958     0.335814
HNSW-M24-ef100-s50                 11007.48      5131.76     16139.24       0.9915     0.160333
HNSW-M24-ef100-s75                 11007.48      8979.96     19987.44       0.9947     0.086197
HNSW-M24-ef100-s100                11007.48     12072.39     23079.87       0.9960     0.076060
HNSW-M24-ef200-s50                 19117.35      5639.14     24756.49       0.9842     3.025679
HNSW-M24-ef200-s75                 19117.35      9708.49     28825.84       0.9930     1.174860
HNSW-M24-ef200-s100                19117.35     12777.75     31895.10       0.9956     0.690686
HNSW-M24-ef300-s50                 31128.95      6231.13     37360.08       0.9695     0.885382
HNSW-M24-ef300-s75                 31128.95     10710.40     41839.35       0.9711     0.688625
HNSW-M24-ef300-s100                31128.95     14875.50     46004.45       0.9739     0.616345
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
Exhaustive                             3.85     56436.27     56440.12       1.0000     0.000000
IVF-nl125-np6                       1682.87     10244.25     11927.11       0.9896     0.009815
IVF-nl125-np12                      1682.87     19759.30     21442.17       1.0000     0.000000
IVF-nl125-np18                      1682.87     25528.27     27211.14       1.0000     0.000000
IVF-nl125-np25                      1682.87     34394.04     36076.91       1.0000     0.000000
IVF-nl250-np12                      1451.32      8812.91     10264.22       0.9925     0.006330
IVF-nl250-np25                      1451.32     15809.24     17260.56       1.0000     0.000000
IVF-nl250-np37                      1451.32     23880.74     25332.06       1.0000     0.000000
IVF-nl250-np50                      1451.32     33096.72     34548.04       1.0000     0.000000
IVF-nl500-np25                      2859.37      8697.25     11556.62       0.9990     0.000708
IVF-nl500-np50                      2859.37     15863.26     18722.63       1.0000     0.000000
IVF-nl500-np75                      2859.37     23784.22     26643.60       1.0000     0.000000
IVF-nl500-np100                     2859.37     31738.51     34597.88       1.0000     0.000000
IVF-nl750-np37                      4159.80      8355.12     12514.92       0.9996     0.000256
IVF-nl750-np75                      4159.80     15681.71     19841.51       1.0000     0.000000
IVF-nl750-np112                     4159.80     25669.76     29829.56       1.0000     0.000000
IVF-nl750-np150                     4159.80     32923.25     37083.05       1.0000     0.000000
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