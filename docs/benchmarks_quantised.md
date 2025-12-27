## Quantised indices benchmarks and parameter gridsearch

Some of these indices were run on synthetic data with better structure in higher
dimensions to avoid the curse of dimensionality, via for example this command.

```bash
cargo run --example gridsearch_ivf_sq8 --release -- --distance euclidean --dim 128 --data correlated
```

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

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.