## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger print.
</br>
In the benchmarks below, some of these indices were run on synthetic data with 
better structure in higher dimensions to avoid the curse of dimensionality, via 
for example this command:

```bash
cargo run --example gridsearch_ivf_sq8 --release --features quantised -- --distance euclidean --dim 128 --data correlated
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
Benchmark: 150k cells, 192D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   22.77     28768.85     28791.63       1.0000     0.000000       109.86
Exhaustive (self)                                    22.77    304666.13    304688.90       1.0000     0.000000       109.86
IVF-PQ-nl273-m16-np13 (query)                     24989.29       510.14     25499.42       0.4551          NaN         3.82
IVF-PQ-nl273-m16-np16 (query)                     24989.29       605.66     25594.95       0.4554          NaN         3.82
IVF-PQ-nl273-m16-np23 (query)                     24989.29       855.85     25845.13       0.4555          NaN         3.82
IVF-PQ-nl273-m16 (self)                           24989.29      8529.15     33518.44       0.3547          NaN         3.82
IVF-PQ-nl273-m32-np13 (query)                     30712.07      1053.19     31765.26       0.6326          NaN         6.11
IVF-PQ-nl273-m32-np16 (query)                     30712.07      1135.41     31847.48       0.6338          NaN         6.11
IVF-PQ-nl273-m32-np23 (query)                     30712.07      1735.02     32447.09       0.6340          NaN         6.11
IVF-PQ-nl273-m32 (self)                           30712.07     16236.94     46949.01       0.5492          NaN         6.11
IVF-PQ-nl273-m48-np13 (query)                     32343.71      1095.30     33439.00       0.7318          NaN         8.40
IVF-PQ-nl273-m48-np16 (query)                     32343.71      1290.47     33634.18       0.7335          NaN         8.40
IVF-PQ-nl273-m48-np23 (query)                     32343.71      1778.10     34121.81       0.7338          NaN         8.40
IVF-PQ-nl273-m48 (self)                           32343.71     18077.79     50421.50       0.6707          NaN         8.40
IVF-PQ-nl387-m16-np19 (query)                     34313.26       742.89     35056.15       0.4585          NaN         3.91
IVF-PQ-nl387-m16-np27 (query)                     34313.26       970.34     35283.60       0.4587          NaN         3.91
IVF-PQ-nl387-m16 (self)                           34313.26      9888.05     44201.31       0.3581          NaN         3.91
IVF-PQ-nl387-m32-np19 (query)                     36978.29      1152.81     38131.10       0.6336          NaN         6.20
IVF-PQ-nl387-m32-np27 (query)                     36978.29      1681.90     38660.18       0.6341          NaN         6.20
IVF-PQ-nl387-m32 (self)                           36978.29     16139.78     53118.07       0.5521          NaN         6.20
IVF-PQ-nl387-m48-np19 (query)                     38754.55      1530.66     40285.20       0.7333          NaN         8.49
IVF-PQ-nl387-m48-np27 (query)                     38754.55      2230.67     40985.21       0.7343          NaN         8.49
IVF-PQ-nl387-m48 (self)                           38754.55     19156.14     57910.69       0.6719          NaN         8.49
IVF-PQ-nl547-m16-np23 (query)                     43206.86       971.77     44178.62       0.4606          NaN         4.03
IVF-PQ-nl547-m16-np27 (query)                     43206.86      1133.87     44340.73       0.4610          NaN         4.03
IVF-PQ-nl547-m16-np33 (query)                     43206.86      1365.84     44572.70       0.4611          NaN         4.03
IVF-PQ-nl547-m16 (self)                           43206.86     13609.93     56816.79       0.3601          NaN         4.03
IVF-PQ-nl547-m32-np23 (query)                     51061.84      1310.41     52372.26       0.6351          NaN         6.32
IVF-PQ-nl547-m32-np27 (query)                     51061.84      1605.42     52667.26       0.6359          NaN         6.32
IVF-PQ-nl547-m32-np33 (query)                     51061.84      1786.68     52848.53       0.6362          NaN         6.32
IVF-PQ-nl547-m32 (self)                           51061.84     18043.41     69105.26       0.5530          NaN         6.32
IVF-PQ-nl547-m48-np23 (query)                     50243.09      1651.08     51894.17       0.7338          NaN         8.60
IVF-PQ-nl547-m48-np27 (query)                     50243.09      1931.16     52174.26       0.7351          NaN         8.60
IVF-PQ-nl547-m48-np33 (query)                     50243.09      2271.64     52514.73       0.7355          NaN         8.60
IVF-PQ-nl547-m48 (self)                           50243.09     21822.83     72065.92       0.6734          NaN         8.60
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
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   14.44     18310.79     18325.23       1.0000     0.000000        73.24
Exhaustive (self)                                    14.44    174999.86    175014.30       1.0000     0.000000        73.24
IVF-OPQ-nl273-m16-np13 (query)                    31730.82       513.62     32244.44       0.5560          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                    31730.82       599.52     32330.34       0.5566          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                    31730.82       847.82     32578.64       0.5568          NaN         3.76
IVF-OPQ-nl273-m16 (self)                          31730.82      7741.97     39472.79       0.4607          NaN         3.76
IVF-OPQ-nl273-m32-np13 (query)                    37127.97       705.35     37833.32       0.7288          NaN         6.05
IVF-OPQ-nl273-m32-np16 (query)                    37127.97       851.71     37979.68       0.7303          NaN         6.05
IVF-OPQ-nl273-m32-np23 (query)                    37127.97      1325.35     38453.33       0.7307          NaN         6.05
IVF-OPQ-nl273-m32 (self)                          37127.97     13479.71     50607.68       0.6652          NaN         6.05
IVF-OPQ-nl387-m16-np19 (query)                    34500.44       579.01     35079.45       0.5587          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                    34500.44       753.76     35254.20       0.5592          NaN         3.81
IVF-OPQ-nl387-m16 (self)                          34500.44      7825.79     42326.24       0.4628          NaN         3.81
IVF-OPQ-nl387-m32-np19 (query)                    41614.44       899.07     42513.50       0.7327          NaN         6.10
IVF-OPQ-nl387-m32-np27 (query)                    41614.44      1266.10     42880.54       0.7337          NaN         6.10
IVF-OPQ-nl387-m32 (self)                          41614.44     12916.84     54531.27       0.6671          NaN         6.10
IVF-OPQ-nl547-m16-np23 (query)                    41269.15       669.71     41938.86       0.5618          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                    41269.15       754.22     42023.38       0.5623          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                    41269.15       928.76     42197.92       0.5626          NaN         3.89
IVF-OPQ-nl547-m16 (self)                          41269.15      9464.63     50733.78       0.4654          NaN         3.89
IVF-OPQ-nl547-m32-np23 (query)                    50331.22      1259.92     51591.14       0.7327          NaN         6.18
IVF-OPQ-nl547-m32-np27 (query)                    50331.22      1450.12     51781.34       0.7338          NaN         6.18
IVF-OPQ-nl547-m32-np33 (query)                    50331.22      1760.89     52092.10       0.7345          NaN         6.18
IVF-OPQ-nl547-m32 (self)                          50331.22     15280.95     65612.17       0.6689          NaN         6.18
---------------------------------------------------------------------------------------------------------------------------
```

**With more dimensions**

With 192 dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 192D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   21.79     27923.13     27944.92       1.0000     0.000000       109.86
Exhaustive (self)                                    21.79    288515.89    288537.68       1.0000     0.000000       109.86
IVF-OPQ-nl273-m16-np13 (query)                    47598.83       516.93     48115.76       0.4618          NaN         3.96
IVF-OPQ-nl273-m16-np16 (query)                    47598.83       631.76     48230.59       0.4623          NaN         3.96
IVF-OPQ-nl273-m16-np23 (query)                    47598.83       874.81     48473.64       0.4624          NaN         3.96
IVF-OPQ-nl273-m16 (self)                          47598.83      9324.44     56923.27       0.3650          NaN         3.96
IVF-OPQ-nl273-m32-np13 (query)                    57200.66       856.83     58057.49       0.6343          NaN         6.25
IVF-OPQ-nl273-m32-np16 (query)                    57200.66      1036.25     58236.91       0.6355          NaN         6.25
IVF-OPQ-nl273-m32-np23 (query)                    57200.66      1461.49     58662.15       0.6357          NaN         6.25
IVF-OPQ-nl273-m32 (self)                          57200.66     15067.78     72268.44       0.5543          NaN         6.25
IVF-OPQ-nl273-m48-np13 (query)                    65944.02      1049.44     66993.46       0.7348          NaN         8.54
IVF-OPQ-nl273-m48-np16 (query)                    65944.02      1275.15     67219.17       0.7365          NaN         8.54
IVF-OPQ-nl273-m48-np23 (query)                    65944.02      1780.09     67724.11       0.7368          NaN         8.54
IVF-OPQ-nl273-m48 (self)                          65944.02     18432.73     84376.75       0.6761          NaN         8.54
IVF-OPQ-nl387-m16-np19 (query)                    56256.93       719.50     56976.43       0.4643          NaN         4.05
IVF-OPQ-nl387-m16-np27 (query)                    56256.93       973.22     57230.15       0.4645          NaN         4.05
IVF-OPQ-nl387-m16 (self)                          56256.93     10343.36     66600.29       0.3679          NaN         4.05
IVF-OPQ-nl387-m32-np19 (query)                    65727.08      1162.76     66889.84       0.6371          NaN         6.34
IVF-OPQ-nl387-m32-np27 (query)                    65727.08      1599.09     67326.17       0.6380          NaN         6.34
IVF-OPQ-nl387-m32 (self)                          65727.08     16507.21     82234.29       0.5573          NaN         6.34
IVF-OPQ-nl387-m48-np19 (query)                    69889.65      1300.99     71190.64       0.7372          NaN         8.63
IVF-OPQ-nl387-m48-np27 (query)                    69889.65      1805.67     71695.32       0.7382          NaN         8.63
IVF-OPQ-nl387-m48 (self)                          69889.65     18702.80     88592.45       0.6769          NaN         8.63
IVF-OPQ-nl547-m16-np23 (query)                    63899.68       824.86     64724.54       0.4655          NaN         4.17
IVF-OPQ-nl547-m16-np27 (query)                    63899.68       934.51     64834.19       0.4659          NaN         4.17
IVF-OPQ-nl547-m16-np33 (query)                    63899.68      1122.14     65021.82       0.4660          NaN         4.17
IVF-OPQ-nl547-m16 (self)                          63899.68     11732.55     75632.23       0.3694          NaN         4.17
IVF-OPQ-nl547-m32-np23 (query)                    73802.83      1278.52     75081.35       0.6376          NaN         6.46
IVF-OPQ-nl547-m32-np27 (query)                    73802.83      1491.19     75294.03       0.6385          NaN         6.46
IVF-OPQ-nl547-m32-np33 (query)                    73802.83      1930.58     75733.41       0.6388          NaN         6.46
IVF-OPQ-nl547-m32 (self)                          73802.83     18357.73     92160.56       0.5582          NaN         6.46
IVF-OPQ-nl547-m48-np23 (query)                    81040.84      1474.67     82515.50       0.7382          NaN         8.75
IVF-OPQ-nl547-m48-np27 (query)                    81040.84      1709.56     82750.40       0.7394          NaN         8.75
IVF-OPQ-nl547-m48-np33 (query)                    81040.84      2057.34     83098.17       0.7398          NaN         8.75
IVF-OPQ-nl547-m48 (self)                          81040.84     21391.77    102432.61       0.6788          NaN         8.75
---------------------------------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.