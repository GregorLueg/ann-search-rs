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
  Additionally, need to figure out better ways to do the kernel magic as the
  CPU to GPU transfers are quite costly and costing performance.

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
with 64 GB of unified memory, see one example below:

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

Detailed benchmarks on the standard benchmarks can be found [here](/docs/benchmarks_general.md)

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

The benchmarks can be found [here]().

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
reaches very fast speeds here, but not much faster actually than the IVF-CPU
version. Due to the current implementation, there is quite a bit of overhead
in copying data from CPU to GPU during individual kernel launches. The 
advantages of the GPU versions are stronger when querying more samples at
higher dimensionality, see next section.

**Euclidean:**

```
====================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                              3.48     24372.91     24376.38       1.0000     0.000000
GPU-Exhaustive                              3.27     15536.64     15539.92       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   1470.27      3163.37      4633.64       0.9954     0.036068
IVF-GPU-kNN-nl273-np16                   1470.27      3668.76      5139.03       0.9998     0.001564
IVF-GPU-kNN-nl273-np23                   1470.27      4714.30      6184.57       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                   2216.07      3473.84      5689.91       0.9962     0.020323
IVF-GPU-kNN-nl387-np27                   2216.07      4536.40      6752.46       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                   2926.25      3489.63      6415.87       0.9905     0.043297
IVF-GPU-kNN-nl547-np27                   2926.25      3636.37      6562.61       0.9971     0.009404
IVF-GPU-kNN-nl547-np33                   2926.25      4077.09      7003.34       0.9997     0.000619
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
version. GPU acceleration shines with larger data sets and larger dimensions, 
hence, the number of samples was increased to 250_000 and dimensions to 48 for 
these benchmarks.

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

With even higher dimensionality, we can observe the advantage of the GPU-accelerated
versions. 

```
====================================================================================================
Benchmark: 150k cells, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             15.25    194745.13    194760.37       1.0000     0.000000
GPU-Exhaustive                             17.07     26365.08     26382.14       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   9825.64      4803.13     14628.77       0.9957     0.018317
IVF-GPU-kNN-nl273-np16                   9825.64      5401.57     15227.21       0.9993     0.002087
IVF-GPU-kNN-nl273-np23                   9825.64      7152.16     16977.80       1.0000     0.000003
IVF-GPU-kNN-nl273-np27                   9825.64      7927.21     17752.85       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                  14274.27      5790.95     20065.22       0.9981     0.009822
IVF-GPU-kNN-nl387-np27                  14274.27      7187.35     21461.62       1.0000     0.000003
IVF-GPU-kNN-nl387-np38                  14274.27      9105.42     23379.69       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                  19754.76      5949.04     25703.80       0.9961     0.019664
IVF-GPU-kNN-nl547-np27                  19754.76      6208.95     25963.71       0.9989     0.004980
IVF-GPU-kNN-nl547-np33                  19754.76      7175.23     26929.99       0.9999     0.000131
IVF-GPU-kNN-nl547-np54                  19754.76     10194.08     29948.84       1.0000     0.000003
----------------------------------------------------------------------------------------------------
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