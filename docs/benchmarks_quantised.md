## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger print.
</br>
In the benchmarks below, some of these indices were run on synthetic data with 
better structure in higher dimensions to avoid the curse of dimensionality, via 
for example this command:

```bash
cargo run --example gridsearch_ivf_sq8 --release --features quantiser -- --distance euclidean --dim 128 --data correlated
```

If you wish to run all of the benchmarks, below, you can just run:

```bash
bash examples/run_benchmarks.sh --quantised
```

### IVF-BF16

The BF16 quantisation reduces the floats to `bf16` which keeps the range of 
`f32`, but loses precision in the digits from ~3 onwards. The actual distance
calculations in the index happen in `f32`; however, due to lossy compression
to `bf16` there is some Recall loss. This is compensated with drastically
reduced memory fingerprint (nearly halved for f32) and increased query speed.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

**Euclidean:**

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.11      2359.09      2362.20       1.0000     0.000000        18.31
Exhaustive (self)                                     3.11     23332.19     23335.30       1.0000     0.000000        18.31
IVF-BF16-nl273-np13 (query)                        1343.66       229.81      1573.47       0.9771     0.096213        10.34
IVF-BF16-nl273-np16 (query)                        1343.66       275.15      1618.81       0.9840     0.070989        10.34
IVF-BF16-nl273-np23 (query)                        1343.66       379.34      1723.00       0.9867     0.065388        10.34
IVF-BF16-nl273 (self)                              1343.66      4329.59      5673.25       0.9830     0.094689        10.34
IVF-BF16-nl387-np19 (query)                        1842.80       233.43      2076.23       0.9804     0.093392        10.35
IVF-BF16-nl387-np27 (query)                        1842.80       319.17      2161.96       0.9865     0.065987        10.35
IVF-BF16-nl387 (self)                              1842.80      3599.21      5442.01       0.9828     0.095265        10.35
IVF-BF16-nl547-np23 (query)                        2602.57       204.71      2807.28       0.9767     0.102772        10.37
IVF-BF16-nl547-np27 (query)                        2602.57       236.60      2839.17       0.9833     0.078352        10.37
IVF-BF16-nl547-np33 (query)                        2602.57       280.14      2882.71       0.9865     0.066298        10.37
IVF-BF16-nl547 (self)                              2602.57      3154.54      5757.12       0.9827     0.095506        10.37
---------------------------------------------------------------------------------------------------------------------------
```

Direct comparison with IVF at nl-273 configuration

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
IVF-nl273-np13 (query)                             1456.09       349.83      1805.92       0.9889     0.034523        19.49
IVF-nl273-np16 (query)                             1456.09       406.19      1862.28       0.9967     0.006608        19.49
IVF-nl273-np23 (query)                             1456.09       666.53      2122.62       1.0000     0.000000        19.49
IVF-nl273 (self)                                   1456.09      6145.20      7601.29       1.0000     0.000000        19.49
---------------------------------------------------------------------------------------------------------------------------
```

Query speed is improved compared to IVF (run on `f32`), the size in memory is
nearly halved; however, the Recall@k does suffer a bit.

**Cosine**

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.40      2423.44      2427.84       1.0000     0.000000        18.88
Exhaustive (self)                                     4.40     23963.03     23967.43       1.0000     0.000000        18.88
IVF-BF16-nl273-np13 (query)                        1358.62       234.47      1593.09       0.7400     0.000825        10.62
IVF-BF16-nl273-np16 (query)                        1358.62       281.52      1640.14       0.7428     0.000825        10.62
IVF-BF16-nl273-np23 (query)                        1358.62       399.44      1758.06       0.7437     0.000827        10.62
IVF-BF16-nl273 (self)                              1358.62      4376.11      5734.73       0.7414     0.001477        10.62
IVF-BF16-nl387-np19 (query)                        1911.42       242.13      2153.55       0.7412     0.000828        10.64
IVF-BF16-nl387-np27 (query)                        1911.42       335.04      2246.46       0.7437     0.000827        10.64
IVF-BF16-nl387 (self)                              1911.42      3657.19      5568.61       0.7413     0.001477        10.64
IVF-BF16-nl547-np23 (query)                        2690.04       211.46      2901.50       0.7399     0.000826        10.66
IVF-BF16-nl547-np27 (query)                        2690.04       240.14      2930.19       0.7425     0.000826        10.66
IVF-BF16-nl547-np33 (query)                        2690.04       285.24      2975.28       0.7436     0.000827        10.66
IVF-BF16-nl547 (self)                              2690.04      3157.21      5847.25       0.7413     0.001478        10.66
---------------------------------------------------------------------------------------------------------------------------
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
does not store the original vectors anymore to reduce memory fingerprint. While
distances can be reported in the `i8` space, they are not very comparable to the
other indices anymore. They are useful for ranking, less for direct comparison
against non-quantised indices.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.15      2323.63      2326.78       1.0000     0.000000        18.31
Exhaustive (self)                                     3.15     23257.97     23261.13       1.0000     0.000000        18.31
IVF-SQ8-nl273-np13 (query)                         1317.10        60.17      1377.27       0.7906          NaN         5.76
IVF-SQ8-nl273-np16 (query)                         1317.10        65.88      1382.98       0.7936          NaN         5.76
IVF-SQ8-nl273-np23 (query)                         1317.10        98.18      1415.27       0.7947          NaN         5.76
IVF-SQ8-nl273 (self)                               1317.10       897.31      2214.41       0.7953          NaN         5.76
IVF-SQ8-nl387-np19 (query)                         1847.52        59.21      1906.73       0.7921          NaN         5.77
IVF-SQ8-nl387-np27 (query)                         1847.52        77.29      1924.81       0.7947          NaN         5.77
IVF-SQ8-nl387 (self)                               1847.52       767.79      2615.31       0.7953          NaN         5.77
IVF-SQ8-nl547-np23 (query)                         2596.54        56.71      2653.24       0.7907          NaN         5.79
IVF-SQ8-nl547-np27 (query)                         2596.54        61.54      2658.08       0.7933          NaN         5.79
IVF-SQ8-nl547-np33 (query)                         2596.54        72.19      2668.73       0.7946          NaN         5.79
IVF-SQ8-nl547 (self)                               2596.54       716.23      3312.77       0.7952          NaN         5.79
---------------------------------------------------------------------------------------------------------------------------
```

We can observe quite a loss of Recall (reducing to ca 0.8); but we gain incredible
query speed. A given query can go from ***~2 seconds to 60 to 80 ms***, a nearly
30x improvement in query speed.

**With more dimensions**

In this case, we increase the dimensions from 32 to 96.

```
===========================================================================================================================
Benchmark: 150k cells, 96D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   10.38     10496.40     10506.78       1.0000     0.000000        54.93
Exhaustive (self)                                    10.38    106818.01    106828.39       1.0000     0.000000        54.93
IVF-SQ8-nl273-np13 (query)                         6155.53       198.79      6354.32       0.8174          NaN        14.98
IVF-SQ8-nl273-np16 (query)                         6155.53       253.81      6409.35       0.8276          NaN        14.98
IVF-SQ8-nl273-np23 (query)                         6155.53       329.67      6485.20       0.8318          NaN        14.98
IVF-SQ8-nl273 (self)                               6155.53      3346.18      9501.71       0.8325          NaN        14.98
IVF-SQ8-nl387-np19 (query)                         8980.64       209.83      9190.47       0.8224          NaN        15.02
IVF-SQ8-nl387-np27 (query)                         8980.64       331.50      9312.13       0.8318          NaN        15.02
IVF-SQ8-nl387 (self)                               8980.64      2779.45     11760.09       0.8325          NaN        15.02
IVF-SQ8-nl547-np23 (query)                        12433.70       220.39     12654.09       0.8169          NaN        15.08
IVF-SQ8-nl547-np27 (query)                        12433.70       244.53     12678.23       0.8258          NaN        15.08
IVF-SQ8-nl547-np33 (query)                        12433.70       265.32     12699.02       0.8307          NaN        15.08
IVF-SQ8-nl547 (self)                              12433.70      2835.38     15269.08       0.8312          NaN        15.08
---------------------------------------------------------------------------------------------------------------------------
```

The improvements in (querying) speed are even more substantial here (40-fold
increase in speed) and we can appreciate a slight improvement in Recall. 

**Data set with stronger correlation structure and more dimensions**

To compare against the next two indices. This data is designed to be better 
suited for high dimensionality with correlation structure between dimensions.
The SQ8 quantisation starts failing here. The next two indices can deal much
better with data that has strong correlations between dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.10     17780.50     17795.60       1.0000     0.000000        73.24
Exhaustive (self)                                    15.10    178991.04    179006.14       1.0000     0.000000        73.24
IVF-SQ8-nl273-np13 (query)                         9936.22       313.35     10249.57       0.6108          NaN        19.59
IVF-SQ8-nl273-np16 (query)                         9936.22       348.11     10284.33       0.6116          NaN        19.59
IVF-SQ8-nl273-np23 (query)                         9936.22       466.00     10402.22       0.6117          NaN        19.59
IVF-SQ8-nl273 (self)                               9936.22      4749.27     14685.49       0.6128          NaN        19.59
IVF-SQ8-nl387-np19 (query)                        14266.19       330.13     14596.32       0.6111          NaN        19.65
IVF-SQ8-nl387-np27 (query)                        14266.19       437.82     14704.01       0.6117          NaN        19.65
IVF-SQ8-nl387 (self)                              14266.19      4278.07     18544.26       0.6128          NaN        19.65
IVF-SQ8-nl547-np23 (query)                        20159.47       424.42     20583.90       0.6107          NaN        19.73
IVF-SQ8-nl547-np27 (query)                        20159.47       391.48     20550.95       0.6113          NaN        19.73
IVF-SQ8-nl547-np33 (query)                        20159.47       444.56     20604.03       0.6117          NaN        19.73
IVF-SQ8-nl547 (self)                              20159.47      4050.60     24210.07       0.6128          NaN        19.73
---------------------------------------------------------------------------------------------------------------------------
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

The self queries here run on the compressed indices stored in the structure
itself. We can appreciate that the lossy compression affects the recall here. If
you wish to get great kNN graphs from these indices, you need to re-supply the
non-compressed data (at cost of memory!). Again, similar to `IVF-SQ8` the 
distances are difficult to interpret/compare against original vectors due to 
the heavy quantisation, thus, are not reported.

**Euclidean:**

With 128 dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.02     18288.27     18303.29       1.0000     0.000000        73.24
Exhaustive (self)                                    15.02    174754.84    174769.86       1.0000     0.000000        73.24
IVF-PQ-nl273-m16-np13 (query)                     17232.81       453.24     17686.06       0.5578          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                     17232.81       543.61     17776.42       0.5584          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                     17232.81       748.60     17981.42       0.5585          NaN         3.69
IVF-PQ-nl273-m16 (self)                           17232.81      7809.20     25042.01       0.4607          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                     19319.67       700.87     20020.53       0.7273          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                     19319.67       838.40     20158.07       0.7287          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                     19319.67      1202.35     20522.02       0.7291          NaN         5.98
IVF-PQ-nl273-m32 (self)                           19319.67     12045.04     31364.71       0.6608          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                     20417.56       541.78     20959.34       0.5580          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                     20417.56       750.22     21167.79       0.5585          NaN         3.75
IVF-PQ-nl387-m16 (self)                           20417.56      7585.76     28003.32       0.4624          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                     23508.80       891.40     24400.20       0.7302          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                     23508.80      1234.14     24742.94       0.7312          NaN         6.04
IVF-PQ-nl387-m32 (self)                           23508.80     12621.52     36130.32       0.6633          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                     26843.28       645.61     27488.89       0.5610          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                     26843.28       789.26     27632.54       0.5614          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                     26843.28       949.42     27792.70       0.5617          NaN         3.83
IVF-PQ-nl547-m16 (self)                           26843.28      9089.45     35932.72       0.4652          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                     29472.25       988.81     30461.06       0.7310          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                     29472.25      1289.50     30761.75       0.7322          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                     29472.25      1525.52     30997.77       0.7328          NaN         6.12
IVF-PQ-nl547-m32 (self)                           29472.25     13438.71     42910.97       0.6656          NaN         6.12
---------------------------------------------------------------------------------------------------------------------------
```

One can appreciate that PQ in this case can yield higher Recalls than just
SQ8 going from ca. 0.62 to 0.72. However, index building and query time are both
higher but at massive reduction of the memory finger print.

**With more dimensions**

With 192 dimensions. In this case, also m = 48, i.e., dividing the original 
vectors into 48 subvectors was tested.

```
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.02     18288.27     18303.29       1.0000     0.000000        73.24
Exhaustive (self)                                    15.02    174754.84    174769.86       1.0000     0.000000        73.24
IVF-PQ-nl273-m16-np13 (query)                     17232.81       453.24     17686.06       0.5578          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                     17232.81       543.61     17776.42       0.5584          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                     17232.81       748.60     17981.42       0.5585          NaN         3.69
IVF-PQ-nl273-m16 (self)                           17232.81      7809.20     25042.01       0.4607          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                     19319.67       700.87     20020.53       0.7273          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                     19319.67       838.40     20158.07       0.7287          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                     19319.67      1202.35     20522.02       0.7291          NaN         5.98
IVF-PQ-nl273-m32 (self)                           19319.67     12045.04     31364.71       0.6608          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                     20417.56       541.78     20959.34       0.5580          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                     20417.56       750.22     21167.79       0.5585          NaN         3.75
IVF-PQ-nl387-m16 (self)                           20417.56      7585.76     28003.32       0.4624          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                     23508.80       891.40     24400.20       0.7302          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                     23508.80      1234.14     24742.94       0.7312          NaN         6.04
IVF-PQ-nl387-m32 (self)                           23508.80     12621.52     36130.32       0.6633          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                     26843.28       645.61     27488.89       0.5610          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                     26843.28       789.26     27632.54       0.5614          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                     26843.28       949.42     27792.70       0.5617          NaN         3.83
IVF-PQ-nl547-m16 (self)                           26843.28      9089.45     35932.72       0.4652          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                     29472.25       988.81     30461.06       0.7310          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                     29472.25      1289.50     30761.75       0.7322          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                     29472.25      1525.52     30997.77       0.7328          NaN         6.12
IVF-PQ-nl547-m32 (self)                           29472.25     13438.71     42910.97       0.6656          NaN         6.12
---------------------------------------------------------------------------------------------------------------------------
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

Similar to IVF-OP, the self kNN generation is run on the compressed indices,
with the same loss of Recall due to the severe compression. Again, similar to 
`IVF-SQ8` the distances are difficult to interpret/compare against original 
vectors due to the heavy quantisation (plus rotation), thus, are not reported.

**Euclidean:**

With 128 dimensions.

```
============================================================================================
Benchmark: 150k cells, 128D
============================================================================================
Method                                     Build (ms)   Query (ms)   Total (ms)     Recall@k
--------------------------------------------------------------------------------------------
Exhaustive (query)                              14.53     17349.43     17363.96       1.0000
Exhaustive (self)                               14.53    175437.33    175451.86       1.0000
IVF-OPQ-nl273-m16-np13 (query)               29891.15       436.46     30327.61       0.5563
IVF-OPQ-nl273-m16-np16 (query)               29891.15       520.67     30411.82       0.5569
IVF-OPQ-nl273-m16-np23 (query)               29891.15       721.99     30613.14       0.5571
IVF-OPQ-nl273-m16 (self)                     29891.15      7471.33     37362.48       0.4608
IVF-OPQ-nl273-m32-np13 (query)               36851.12       690.14     37541.26       0.7290
IVF-OPQ-nl273-m32-np16 (query)               36851.12       834.66     37685.78       0.7305
IVF-OPQ-nl273-m32-np23 (query)               36851.12      1174.46     38025.58       0.7308
IVF-OPQ-nl273-m32 (self)                     36851.12     12003.31     48854.43       0.6653
IVF-OPQ-nl387-m16-np19 (query)               33631.55       540.88     34172.43       0.5581
IVF-OPQ-nl387-m16-np27 (query)               33631.55       755.04     34386.60       0.5586
IVF-OPQ-nl387-m16 (self)                     33631.55      7792.71     41424.26       0.4628
IVF-OPQ-nl387-m32-np19 (query)               41130.64       896.34     42026.98       0.7322
IVF-OPQ-nl387-m32-np27 (query)               41130.64      1237.63     42368.27       0.7332
IVF-OPQ-nl387-m32 (self)                     41130.64     12807.23     53937.87       0.6669
IVF-OPQ-nl547-m16-np23 (query)               39396.87       611.54     40008.41       0.5615
IVF-OPQ-nl547-m16-np27 (query)               39396.87       714.49     40111.36       0.5620
IVF-OPQ-nl547-m16-np33 (query)               39396.87       862.31     40259.18       0.5622
IVF-OPQ-nl547-m16 (self)                     39396.87      8836.52     48233.39       0.4657
IVF-OPQ-nl547-m32-np23 (query)               46883.00       980.48     47863.49       0.7328
IVF-OPQ-nl547-m32-np27 (query)               46883.00      1136.66     48019.66       0.7340
IVF-OPQ-nl547-m32-np33 (query)               46883.00      1360.99     48244.00       0.7345
IVF-OPQ-nl547-m32 (self)                     46883.00     14238.80     61121.80       0.6692
--------------------------------------------------------------------------------------------
```

**With more dimensions**

With 192 dimensions.

```
============================================================================================
Benchmark: 150k cells, 192D
============================================================================================
Method                                     Build (ms)   Query (ms)   Total (ms)     Recall@k
--------------------------------------------------------------------------------------------
Exhaustive (query)                              22.41     28448.17     28470.58       1.0000
Exhaustive (self)                               22.41    283657.88    283680.30       1.0000
IVF-OPQ-nl273-m16-np13 (query)               50293.57       533.11     50826.67       0.4613
IVF-OPQ-nl273-m16-np16 (query)               50293.57       643.56     50937.12       0.4618
IVF-OPQ-nl273-m16-np23 (query)               50293.57       908.28     51201.84       0.4619
IVF-OPQ-nl273-m16 (self)                     50293.57      9534.01     59827.58       0.3651
IVF-OPQ-nl273-m32-np13 (query)               58646.60       883.23     59529.83       0.6347
IVF-OPQ-nl273-m32-np16 (query)               58646.60      1053.27     59699.87       0.6359
IVF-OPQ-nl273-m32-np23 (query)               58646.60      1481.17     60127.77       0.6361
IVF-OPQ-nl273-m32 (self)                     58646.60     15211.51     73858.11       0.5544
IVF-OPQ-nl273-m48-np13 (query)               63756.75      1033.02     64789.76       0.7352
IVF-OPQ-nl273-m48-np16 (query)               63756.75      1250.81     65007.56       0.7370
IVF-OPQ-nl273-m48-np23 (query)               63756.75      1755.07     65511.81       0.7372
IVF-OPQ-nl273-m48 (self)                     63756.75     18089.40     81846.15       0.6757
IVF-OPQ-nl387-m16-np19 (query)               56530.62       711.53     57242.16       0.4650
IVF-OPQ-nl387-m16-np27 (query)               56530.62       968.79     57499.42       0.4652
IVF-OPQ-nl387-m16 (self)                     56530.62     10452.18     66982.81       0.3679
IVF-OPQ-nl387-m32-np19 (query)               66098.35      1163.20     67261.55       0.6376
IVF-OPQ-nl387-m32-np27 (query)               66098.35      1585.60     67683.94       0.6383
IVF-OPQ-nl387-m32 (self)                     66098.35     16320.14     82418.49       0.5574
IVF-OPQ-nl387-m48-np19 (query)               70394.53      1317.40     71711.93       0.7372
IVF-OPQ-nl387-m48-np27 (query)               70394.53      1836.00     72230.53       0.7383
IVF-OPQ-nl387-m48 (self)                     70394.53     18938.40     89332.93       0.6774
IVF-OPQ-nl547-m16-np23 (query)               65962.12       819.98     66782.10       0.4662
IVF-OPQ-nl547-m16-np27 (query)               65962.12       964.90     66927.02       0.4665
IVF-OPQ-nl547-m16-np33 (query)               65962.12      1158.44     67120.56       0.4666
IVF-OPQ-nl547-m16 (self)                     65962.12     12014.58     77976.69       0.3698
IVF-OPQ-nl547-m32-np23 (query)               75614.05      1281.38     76895.43       0.6377
IVF-OPQ-nl547-m32-np27 (query)               75614.05      1496.12     77110.17       0.6385
IVF-OPQ-nl547-m32-np33 (query)               75614.05      1822.78     77436.82       0.6388
IVF-OPQ-nl547-m32 (self)                     75614.05     18387.25     94001.29       0.5579
IVF-OPQ-nl547-m48-np23 (query)               79678.03      1482.07     81160.09       0.7378
IVF-OPQ-nl547-m48-np27 (query)               79678.03      1724.69     81402.72       0.7391
IVF-OPQ-nl547-m48-np33 (query)               79678.03      2105.26     81783.28       0.7395
IVF-OPQ-nl547-m48 (self)                     79678.03     21495.91    101173.94       0.6786
--------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.