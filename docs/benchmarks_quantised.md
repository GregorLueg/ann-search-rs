## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger
print. If you wish to run on the examples, you can do so via:

```bash
cargo run --example gridsearch_sq8 --release --features quantised
```

If you wish to run all of the benchmarks, below, you can just run:

```bash
bash examples/run_benchmarks.sh --quantised
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided.

## Table of Contents

- [BF16 quantisation](#bf16-ivf-and-exhaustive)
- [SQ8 quantisation](#sq8-ivf-and-exhaustive)
- [Product quantisation](#product-quantisation-ivf-only)
- [Optimised product quantisation](#optimised-product-quantisation-ivf-only)

### BF16 (IVF and exhaustive)

The BF16 quantisation reduces the floats to `bf16` which keeps the range of
`f32`, but loses precision in the digits from ~3 onwards. The actual distance
calculations in the index happen in `f32`; however, due to lossy compression
to `bf16` there is some Recall loss. This is compensated with drastically
reduced memory fingerprint (nearly halved for f32). The precision loss is
higher for Cosine over Euclidean distance.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

<details>
<summary><b>BF16 quantisations - Euclidean (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.14     1_466.17     1_469.31       1.0000     0.000000        18.31
Exhaustive (self)                                          3.14    14_688.74    14_691.88       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.46     1_189.46     1_194.92       0.9867     0.065388         9.16
Exhaustive-BF16 (self)                                     5.46    15_217.56    15_223.02       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              809.67       134.15       943.82       0.9771     0.096200        10.34
IVF-BF16-nl273-np16 (query)                              809.67       159.54       969.21       0.9840     0.070925        10.34
IVF-BF16-nl273-np23 (query)                              809.67       218.74     1_028.41       0.9867     0.065388        10.34
IVF-BF16-nl273 (self)                                    809.67     2_216.19     3_025.85       0.9830     0.094689        10.34
IVF-BF16-nl387-np19 (query)                            1_143.36       139.22     1_282.58       0.9805     0.093344        10.35
IVF-BF16-nl387-np27 (query)                            1_143.36       187.51     1_330.87       0.9865     0.065976        10.35
IVF-BF16-nl387 (self)                                  1_143.36     1_905.95     3_049.31       0.9828     0.095259        10.35
IVF-BF16-nl547-np23 (query)                            1_586.90       119.64     1_706.55       0.9767     0.102772        10.37
IVF-BF16-nl547-np27 (query)                            1_586.90       140.60     1_727.51       0.9833     0.078352        10.37
IVF-BF16-nl547-np33 (query)                            1_586.90       162.02     1_748.92       0.9865     0.066297        10.37
IVF-BF16-nl547 (self)                                  1_586.90     1_648.56     3_235.47       0.9827     0.095506        10.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.35     1_519.06     1_523.40       1.0000     0.000000        18.88
Exhaustive (self)                                          4.35    15_957.53    15_961.88       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    6.32     1_325.73     1_332.05       0.9240     0.000230         9.44
Exhaustive-BF16 (self)                                     6.32    15_892.71    15_899.03       1.0000     0.000000         9.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              842.22       140.20       982.42       0.9176     0.000245        10.62
IVF-BF16-nl273-np16 (query)                              842.22       170.53     1_012.76       0.9224     0.000233        10.62
IVF-BF16-nl273-np23 (query)                              842.22       236.86     1_079.08       0.9240     0.000230        10.62
IVF-BF16-nl273 (self)                                    842.22     2_421.74     3_263.96       0.9229     0.001251        10.62
IVF-BF16-nl387-np19 (query)                            1_189.78       144.80     1_334.58       0.9198     0.000242        10.64
IVF-BF16-nl387-np27 (query)                            1_189.78       198.89     1_388.68       0.9239     0.000230        10.64
IVF-BF16-nl387 (self)                                  1_189.78     2_022.12     3_211.91       0.9228     0.001251        10.64
IVF-BF16-nl547-np23 (query)                            1_667.97       127.68     1_795.65       0.9177     0.000245        10.66
IVF-BF16-nl547-np27 (query)                            1_667.97       145.67     1_813.64       0.9219     0.000235        10.66
IVF-BF16-nl547-np33 (query)                            1_667.97       172.73     1_840.70       0.9238     0.000230        10.66
IVF-BF16-nl547 (self)                                  1_667.97     1_795.51     3_463.48       0.9228     0.001251        10.66
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.09     1_499.29     1_502.38       1.0000     0.000000        18.31
Exhaustive (self)                                          3.09    16_277.11    16_280.21       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    6.08     1_297.43     1_303.51       0.9649     0.019029         9.16
Exhaustive-BF16 (self)                                     6.08    15_647.79    15_653.87       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              816.40       125.28       941.68       0.9648     0.019044        10.34
IVF-BF16-nl273-np16 (query)                              816.40       153.51       969.92       0.9649     0.019032        10.34
IVF-BF16-nl273-np23 (query)                              816.40       218.28     1_034.68       0.9649     0.019029        10.34
IVF-BF16-nl273 (self)                                    816.40     2_197.18     3_013.59       0.9561     0.026167        10.34
IVF-BF16-nl387-np19 (query)                            1_128.00       128.67     1_256.67       0.9649     0.019036        10.35
IVF-BF16-nl387-np27 (query)                            1_128.00       176.35     1_304.36       0.9649     0.019029        10.35
IVF-BF16-nl387 (self)                                  1_128.00     1_776.20     2_904.21       0.9561     0.026167        10.35
IVF-BF16-nl547-np23 (query)                            1_592.79       115.57     1_708.36       0.9649     0.019042        10.37
IVF-BF16-nl547-np27 (query)                            1_592.79       131.72     1_724.51       0.9649     0.019035        10.37
IVF-BF16-nl547-np33 (query)                            1_592.79       155.14     1_747.93       0.9649     0.019029        10.37
IVF-BF16-nl547 (self)                                  1_592.79     1_566.60     3_159.39       0.9561     0.026169        10.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.17     1_491.84     1_495.01       1.0000     0.000000        18.31
Exhaustive (self)                                          3.17    16_004.95    16_008.13       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    4.91     1_183.87     1_188.79       0.9348     0.158338         9.16
Exhaustive-BF16 (self)                                     4.91    15_293.71    15_298.62       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              839.60       127.87       967.47       0.9348     0.158347        10.34
IVF-BF16-nl273-np16 (query)                              839.60       151.63       991.23       0.9348     0.158338        10.34
IVF-BF16-nl273-np23 (query)                              839.60       202.52     1_042.12       0.9348     0.158338        10.34
IVF-BF16-nl273 (self)                                    839.60     2_060.06     2_899.66       0.9174     0.221294        10.34
IVF-BF16-nl387-np19 (query)                            1_144.47       129.91     1_274.38       0.9348     0.158338        10.35
IVF-BF16-nl387-np27 (query)                            1_144.47       173.33     1_317.80       0.9348     0.158338        10.35
IVF-BF16-nl387 (self)                                  1_144.47     1_763.88     2_908.36       0.9174     0.221294        10.35
IVF-BF16-nl547-np23 (query)                            1_608.34       119.13     1_727.47       0.9348     0.158338        10.37
IVF-BF16-nl547-np27 (query)                            1_608.34       136.06     1_744.41       0.9348     0.158338        10.37
IVF-BF16-nl547-np33 (query)                            1_608.34       160.35     1_768.69       0.9348     0.158338        10.37
IVF-BF16-nl547 (self)                                  1_608.34     1_731.03     3_339.37       0.9174     0.221294        10.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### SQ8 (IVF and exhaustive)

This index uses scalar quantisation to 8-bits. It projects every dimensions
onto an `i8`. This also causes a reduction of the memory finger print. In the
case of 96 dimensions in f32 per vector, we go from *96 x 32 bits = 384 bytes*
to *96 x 8 bits = 96 bytes per vector*, a **4x reduction in memory per vector**
(with overhead of the codebook). Additionally, the querying becomes much faster
due to integer math.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

#### With 32 dimensions

The quantisation performs well on GaussianNoise data; however, the it loses
information for the correlated and also low rank data, indicating that complex
structure is lost during the lossy compression.

<details>
<summary><b>SQ8 quantisations - Euclidean (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.09     1_564.24     1_567.33       1.0000     0.000000        18.31
Exhaustive (self)                                          3.09    17_367.36    17_370.45       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.27       758.78       766.05       0.8011          NaN         4.58
Exhaustive-SQ8 (self)                                      7.27     7_863.08     7_870.35       0.8007          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               925.86        59.22       985.08       0.7972          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               925.86        66.88       992.74       0.8001          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               925.86        92.28     1_018.14       0.8011          NaN         5.76
IVF-SQ8-nl273 (self)                                     925.86       912.20     1_838.07       0.8007          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_227.42        64.32     1_291.74       0.7986          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_227.42        79.79     1_307.21       0.8011          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_227.42       737.41     1_964.83       0.8007          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_687.83        56.47     1_744.30       0.7968          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_687.83        73.60     1_761.43       0.7995          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_687.83        72.61     1_760.45       0.8010          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_687.83       720.97     2_408.80       0.8006          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.33     1_522.92     1_527.25       1.0000     0.000000        18.88
Exhaustive (self)                                          4.33    16_145.25    16_149.58       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    10.12       718.15       728.26       0.8501          NaN         5.15
Exhaustive-SQ8 (self)                                     10.12     7_479.45     7_489.56       0.8497          NaN         5.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                             1_003.57        62.48     1_066.04       0.8455          NaN         6.33
IVF-SQ8-nl273-np16 (query)                             1_003.57        73.81     1_077.38       0.8488          NaN         6.33
IVF-SQ8-nl273-np23 (query)                             1_003.57       102.79     1_106.36       0.8501          NaN         6.33
IVF-SQ8-nl273 (self)                                   1_003.57     1_048.72     2_052.29       0.8497          NaN         6.33
IVF-SQ8-nl387-np19 (query)                             1_249.26        65.47     1_314.74       0.8472          NaN         6.34
IVF-SQ8-nl387-np27 (query)                             1_249.26        96.27     1_345.53       0.8500          NaN         6.34
IVF-SQ8-nl387 (self)                                   1_249.26       904.26     2_153.53       0.8497          NaN         6.34
IVF-SQ8-nl547-np23 (query)                             1_712.07        65.53     1_777.60       0.8456          NaN         6.37
IVF-SQ8-nl547-np27 (query)                             1_712.07        72.26     1_784.33       0.8486          NaN         6.37
IVF-SQ8-nl547-np33 (query)                             1_712.07        83.69     1_795.75       0.8500          NaN         6.37
IVF-SQ8-nl547 (self)                                   1_712.07       787.25     2_499.32       0.8496          NaN         6.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.07     1_573.82     1_576.89       1.0000     0.000000        18.31
Exhaustive (self)                                          3.07    16_483.63    16_486.70       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     6.97       756.11       763.08       0.6828          NaN         4.58
Exhaustive-SQ8 (self)                                      6.97     8_091.68     8_098.65       0.6835          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               908.38        67.40       975.78       0.6829          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               908.38        75.59       983.97       0.6829          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               908.38        92.14     1_000.52       0.6829          NaN         5.76
IVF-SQ8-nl273 (self)                                     908.38       898.96     1_807.34       0.6835          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_360.89        60.16     1_421.05       0.6829          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_360.89        82.01     1_442.91       0.6828          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_360.89       749.15     2_110.04       0.6835          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_694.63        55.93     1_750.56       0.6830          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_694.63        62.69     1_757.32       0.6830          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_694.63        71.06     1_765.69       0.6829          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_694.63       688.57     2_383.20       0.6835          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.08     1_549.25     1_552.33       1.0000     0.000000        18.31
Exhaustive (self)                                          3.08    15_387.54    15_390.62       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.59       724.68       732.27       0.4800          NaN         4.58
Exhaustive-SQ8 (self)                                      7.59     7_880.97     7_888.56       0.4862          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               908.20        55.87       964.06       0.4802          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               908.20        73.73       981.92       0.4801          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               908.20        97.39     1_005.59       0.4801          NaN         5.76
IVF-SQ8-nl273 (self)                                     908.20       825.77     1_733.97       0.4864          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_211.19        60.81     1_272.00       0.4802          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_211.19        74.70     1_285.89       0.4801          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_211.19       722.73     1_933.92       0.4864          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_693.60        60.84     1_754.44       0.4802          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_693.60        70.58     1_764.18       0.4801          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_693.60        78.30     1_771.90       0.4801          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_693.60       689.46     2_383.06       0.4864          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 96D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.82     4_562.27     4_573.08       1.0000     0.000000        54.93
Exhaustive (self)                                         10.82    48_238.23    48_249.05       1.0000     0.000000        54.93
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    27.86     1_323.04     1_350.90       0.7793          NaN        13.73
Exhaustive-SQ8 (self)                                     27.86    13_446.15    13_474.01       0.7834          NaN        13.73
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                             2_132.55       193.59     2_326.14       0.7794          NaN        14.98
IVF-SQ8-nl273-np16 (query)                             2_132.55       226.37     2_358.92       0.7794          NaN        14.98
IVF-SQ8-nl273-np23 (query)                             2_132.55       317.06     2_449.60       0.7794          NaN        14.98
IVF-SQ8-nl273 (self)                                   2_132.55     3_071.56     5_204.10       0.7835          NaN        14.98
IVF-SQ8-nl387-np19 (query)                             2_842.84       209.01     3_051.85       0.7794          NaN        15.02
IVF-SQ8-nl387-np27 (query)                             2_842.84       266.35     3_109.19       0.7793          NaN        15.02
IVF-SQ8-nl387 (self)                                   2_842.84     2_628.29     5_471.13       0.7835          NaN        15.02
IVF-SQ8-nl547-np23 (query)                             4_091.59       211.09     4_302.68       0.7794          NaN        15.08
IVF-SQ8-nl547-np27 (query)                             4_091.59       230.51     4_322.10       0.7793          NaN        15.08
IVF-SQ8-nl547-np33 (query)                             4_091.59       269.22     4_360.82       0.7794          NaN        15.08
IVF-SQ8-nl547 (self)                                   4_091.59     2_674.10     6_765.69       0.7835          NaN        15.08
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Product quantisation (Exhaustive and IVF)

This index uses product quantisation. To note, the quantisation is quite harsh
and hence, reduces the Recall quite substantially. In the case of 192
dimensions, each vector gets reduced to
from *192 x 32 bits (192 x f32) = 768 bytes* to for
*m = 32 (32 sub vectors) to 32 x u8 = 32 bytes*, a
**24x reduction in memory usage** (of course with overhead from the cook book).
However, it can still be useful in situation where good enough works and you
have VERY large scale data and memory constraints start biting.

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
non-compressed data (at cost of memory!). Similar to `SQ8`-indices, the
distances are difficult to interpret/compare against original vectors due to
the heavy quantisation, thus, are not reported. The self queries default
to `sqrt(nlist)`.

#### Why IVF massively outperforms Exhaustive PQ

A key observation is the large outperformance of the IVF index over the
exhaustive index. This is not incidental - it is fundamental to how PQ works.

Product quantisation divides each vector into m subvectors and quantises each
to one of 256 centroids. The quality of this approximation depends critically
on the **variance** of the data being quantised: lower variance means the 256
centroids can tile the space more densely, yielding smaller quantisation error.

**IVF-PQ** first clusters the dataset, then encodes **residuals** (vector minus
cluster centroid) rather than raw vectors. Vectors within a cluster are similar,
so their residuals are small, tightly distributed around zero, and share
correlated structure. The PQ codebooks can represent these local patterns
efficiently.

**Exhaustive-PQ** must encode raw vectors directly. The codebooks must represent
the entire dataset's diversity - wildly different vectors compete for the same
256 centroids per subspace. This leads to fundamentally higher quantisation
error.

In short: IVF's clustering creates **locality**, and locality is what PQ needs
to quantise accurately. Mean-centering or rotations (OPQ) do not create this
locality - they shift or rotate the data but do not reduce its intrinsic
spread. The clustering step is not optional for high-recall PQ search.

#### With 128 dimensions

Due to being optimised for high dimensional data, we start with 128 dimensions
here. One can appreciate that `m16` causes a high Recall loss as it compresses
the data too much and the more "structure" the data has (correlated features,
low rank manifolds), the better these indices perform. The PQ quantisation
however is a bit more expensive, hence, more time is being used on index
building. Querying on the other hand is very performant.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.59     6_087.47     6_102.06       1.0000     0.000000        73.24
Exhaustive (self)                                         14.59    62_185.97    62_200.56       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_050.68     1_898.02     2_948.70       0.1328          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_050.68    18_935.04    19_985.72       0.0967          NaN         2.41
Exhaustive-PQ-m32 (query)                              1_745.70     4_348.06     6_093.77       0.2523          NaN         4.70
Exhaustive-PQ-m32 (self)                               1_745.70    43_066.56    44_812.26       0.1715          NaN         4.70
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          4_616.23       583.77     5_200.00       0.2674          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          4_616.23       787.01     5_403.24       0.2696          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          4_616.23     1_054.07     5_670.31       0.2707          NaN         3.69
IVF-PQ-nl273-m16 (self)                                4_616.23     9_309.11    13_925.34       0.1802          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                          4_664.83       895.41     5_560.24       0.5096          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                          4_664.83     1_091.34     5_756.17       0.5175          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                          4_664.83     1_535.93     6_200.77       0.5224          NaN         5.98
IVF-PQ-nl273-m32 (self)                                4_664.83    15_403.56    20_068.40       0.4290          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                          5_128.10       677.73     5_805.84       0.2696          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          5_128.10       957.75     6_085.85       0.2714          NaN         3.75
IVF-PQ-nl387-m16 (self)                                5_128.10     9_469.66    14_597.76       0.1804          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                          5_823.09     1_134.70     6_957.79       0.5165          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                          5_823.09     1_578.12     7_401.21       0.5242          NaN         6.04
IVF-PQ-nl387-m32 (self)                                5_823.09    15_898.07    21_721.16       0.4301          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                          6_659.42       752.79     7_412.21       0.2711          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          6_659.42       886.22     7_545.64       0.2725          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          6_659.42     1_060.31     7_719.73       0.2735          NaN         3.83
IVF-PQ-nl547-m16 (self)                                6_659.42    10_674.97    17_334.39       0.1820          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                          7_390.95     1_267.73     8_658.68       0.5140          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                          7_390.95     1_465.28     8_856.23       0.5201          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                          7_390.95     1_781.92     9_172.87       0.5245          NaN         6.12
IVF-PQ-nl547-m32 (self)                                7_390.95    17_839.86    25_230.81       0.4325          NaN         6.12
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.84     6_272.76     6_287.60       1.0000     0.000000        73.24
Exhaustive (self)                                         14.84    69_310.28    69_325.12       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_114.92     5_855.97     6_970.90       0.3518          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_114.92    24_743.58    25_858.51       0.2621          NaN         2.41
Exhaustive-PQ-m32 (query)                              3_995.15     5_193.57     9_188.72       0.4767          NaN         4.70
Exhaustive-PQ-m32 (self)                               3_995.15    48_893.32    52_888.47       0.3849          NaN         4.70
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_757.00       586.01     6_343.01       0.5509          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          5_757.00       691.00     6_448.00       0.5515          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          5_757.00       988.05     6_745.05       0.5516          NaN         3.69
IVF-PQ-nl273-m16 (self)                                5_757.00    10_070.76    15_827.77       0.4540          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                          4_894.75       891.25     5_786.01       0.7142          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                          4_894.75     1_099.62     5_994.37       0.7156          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                          4_894.75     1_545.88     6_440.64       0.7160          NaN         5.98
IVF-PQ-nl273-m32 (self)                                4_894.75    15_562.98    20_457.73       0.6433          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                          5_300.75       723.40     6_024.15       0.5528          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          5_300.75       955.34     6_256.09       0.5533          NaN         3.75
IVF-PQ-nl387-m16 (self)                                5_300.75     9_534.64    14_835.39       0.4566          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                          5_913.23     1_132.81     7_046.04       0.7159          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                          5_913.23     1_586.67     7_499.89       0.7170          NaN         6.04
IVF-PQ-nl387-m32 (self)                                5_913.23    16_166.44    22_079.67       0.6463          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                          6_944.84       765.59     7_710.43       0.5575          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          6_944.84       907.97     7_852.81       0.5580          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          6_944.84     1_071.32     8_016.16       0.5584          NaN         3.83
IVF-PQ-nl547-m16 (self)                                6_944.84    10_696.67    17_641.52       0.4597          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                          7_614.53     1_267.02     8_881.55       0.7161          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                          7_614.53     1_480.44     9_094.97       0.7172          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                          7_614.53     1_789.80     9_404.33       0.7178          NaN         6.12
IVF-PQ-nl547-m32 (self)                                7_614.53    18_416.21    26_030.74       0.6479          NaN         6.12
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.79     6_090.57     6_105.36       1.0000     0.000000        73.24
Exhaustive (self)                                         14.79    65_954.39    65_969.18       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_072.81     1_887.90     2_960.71       0.3957          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_072.81    18_818.86    19_891.67       0.2941          NaN         2.41
Exhaustive-PQ-m32 (query)                              1_746.19     4_287.67     6_033.86       0.5474          NaN         4.70
Exhaustive-PQ-m32 (self)                               1_746.19    43_023.15    44_769.34       0.4400          NaN         4.70
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          3_943.94       514.95     4_458.89       0.6934          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          3_943.94       637.82     4_581.77       0.6934          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          3_943.94       864.25     4_808.20       0.6934          NaN         3.69
IVF-PQ-nl273-m16 (self)                                3_943.94     8_679.15    12_623.09       0.5877          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                          4_649.96       849.21     5_499.16       0.8412          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                          4_649.96     1_065.10     5_715.06       0.8413          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                          4_649.96     1_476.65     6_126.60       0.8413          NaN         5.98
IVF-PQ-nl273-m32 (self)                                4_649.96    14_642.32    19_292.27       0.7868          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                          5_148.77       687.31     5_836.09       0.6975          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          5_148.77       939.81     6_088.58       0.6975          NaN         3.75
IVF-PQ-nl387-m16 (self)                                5_148.77     9_182.41    14_331.19       0.5887          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                          5_793.69     1_113.69     6_907.39       0.8444          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                          5_793.69     1_539.38     7_333.08       0.8444          NaN         6.04
IVF-PQ-nl387-m32 (self)                                5_793.69    15_579.85    21_373.55       0.7894          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                          6_706.49       777.57     7_484.06       0.7019          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          6_706.49       889.90     7_596.39       0.7019          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          6_706.49     1_078.32     7_784.80       0.7019          NaN         3.83
IVF-PQ-nl547-m16 (self)                                6_706.49    10_530.75    17_237.24       0.5901          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                          7_620.32     1_286.27     8_906.59       0.8470          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                          7_620.32     1_459.40     9_079.72       0.8470          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                          7_620.32     1_778.31     9_398.64       0.8470          NaN         6.12
IVF-PQ-nl547-m32 (self)                                7_620.32    17_755.06    25_375.38       0.7919          NaN         6.12
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 192 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is not too great, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        22.50     9_850.65     9_873.15       1.0000     0.000000       109.86
Exhaustive (self)                                         22.50   104_147.35   104_169.84       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_378.93     1_916.00     3_294.93       0.1074          NaN         2.48
Exhaustive-PQ-m16 (self)                               1_378.93    18_932.94    20_311.87       0.0860          NaN         2.48
Exhaustive-PQ-m32 (query)                              2_174.92     4_326.15     6_501.07       0.1615          NaN         4.77
Exhaustive-PQ-m32 (self)                               2_174.92    43_272.79    45_447.71       0.1093          NaN         4.77
Exhaustive-PQ-m48 (query)                              2_602.36     8_911.83    11_514.20       0.2566          NaN         7.06
Exhaustive-PQ-m48 (self)                               2_602.36    87_122.87    89_725.24       0.1766          NaN         7.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_782.85       583.59     6_366.44       0.1944          NaN         3.82
IVF-PQ-nl273-m16-np16 (query)                          5_782.85       716.14     6_498.99       0.1950          NaN         3.82
IVF-PQ-nl273-m16-np23 (query)                          5_782.85       972.65     6_755.50       0.1950          NaN         3.82
IVF-PQ-nl273-m16 (self)                                5_782.85     9_760.11    15_542.96       0.1282          NaN         3.82
IVF-PQ-nl273-m32-np13 (query)                          6_585.63     1_103.26     7_688.89       0.3603          NaN         6.11
IVF-PQ-nl273-m32-np16 (query)                          6_585.63     1_320.60     7_906.23       0.3627          NaN         6.11
IVF-PQ-nl273-m32-np23 (query)                          6_585.63     2_005.98     8_591.61       0.3631          NaN         6.11
IVF-PQ-nl273-m32 (self)                                6_585.63    18_688.08    25_273.70       0.2672          NaN         6.11
IVF-PQ-nl273-m48-np13 (query)                          6_961.20     1_343.59     8_304.79       0.5233          NaN         8.40
IVF-PQ-nl273-m48-np16 (query)                          6_961.20     1_624.84     8_586.04       0.5282          NaN         8.40
IVF-PQ-nl273-m48-np23 (query)                          6_961.20     2_310.35     9_271.56       0.5291          NaN         8.40
IVF-PQ-nl273-m48 (self)                                6_961.20    23_095.84    30_057.04       0.4409          NaN         8.40
IVF-PQ-nl387-m16-np19 (query)                          7_513.08       762.72     8_275.80       0.1953          NaN         3.91
IVF-PQ-nl387-m16-np27 (query)                          7_513.08     1_051.42     8_564.50       0.1958          NaN         3.91
IVF-PQ-nl387-m16 (self)                                7_513.08    10_365.38    17_878.46       0.1290          NaN         3.91
IVF-PQ-nl387-m32-np19 (query)                          8_263.43     1_427.62     9_691.05       0.3626          NaN         6.20
IVF-PQ-nl387-m32-np27 (query)                          8_263.43     2_008.22    10_271.66       0.3648          NaN         6.20
IVF-PQ-nl387-m32 (self)                                8_263.43    19_898.51    28_161.94       0.2682          NaN         6.20
IVF-PQ-nl387-m48-np19 (query)                          8_740.92     1_734.84    10_475.76       0.5252          NaN         8.49
IVF-PQ-nl387-m48-np27 (query)                          8_740.92     2_435.47    11_176.39       0.5296          NaN         8.49
IVF-PQ-nl387-m48 (self)                                8_740.92    24_188.18    32_929.10       0.4417          NaN         8.49
IVF-PQ-nl547-m16-np23 (query)                         10_027.07       883.48    10_910.55       0.1975          NaN         4.03
IVF-PQ-nl547-m16-np27 (query)                         10_027.07     1_011.08    11_038.15       0.1979          NaN         4.03
IVF-PQ-nl547-m16-np33 (query)                         10_027.07     1_195.46    11_222.53       0.1982          NaN         4.03
IVF-PQ-nl547-m16 (self)                               10_027.07    12_026.92    22_053.99       0.1301          NaN         4.03
IVF-PQ-nl547-m32-np23 (query)                         10_839.33     1_678.85    12_518.18       0.3614          NaN         6.32
IVF-PQ-nl547-m32-np27 (query)                         10_839.33     1_935.52    12_774.85       0.3641          NaN         6.32
IVF-PQ-nl547-m32-np33 (query)                         10_839.33     2_328.14    13_167.47       0.3652          NaN         6.32
IVF-PQ-nl547-m32 (self)                               10_839.33    23_407.93    34_247.26       0.2691          NaN         6.32
IVF-PQ-nl547-m48-np23 (query)                         12_397.64     2_045.75    14_443.39       0.5220          NaN         8.60
IVF-PQ-nl547-m48-np27 (query)                         12_397.64     2_529.97    14_927.61       0.5271          NaN         8.60
IVF-PQ-nl547-m48-np33 (query)                         12_397.64     3_359.91    15_757.55       0.5290          NaN         8.60
IVF-PQ-nl547-m48 (self)                               12_397.64    29_616.72    42_014.36       0.4418          NaN         8.60
--------------------------------------------------------------------------------------------------------------------------------
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.46     9_751.23     9_772.68       1.0000     0.000000       109.86
Exhaustive (self)                                         21.46   103_939.51   103_960.97       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_444.14     1_910.46     3_354.61       0.2620          NaN         2.48
Exhaustive-PQ-m16 (self)                               1_444.14    19_040.88    20_485.03       0.1859          NaN         2.48
Exhaustive-PQ-m32 (query)                              2_183.27     4_368.96     6_552.22       0.3861          NaN         4.77
Exhaustive-PQ-m32 (self)                               2_183.27    43_525.01    45_708.27       0.2943          NaN         4.77
Exhaustive-PQ-m48 (query)                              2_630.85     8_634.11    11_264.97       0.4652          NaN         7.06
Exhaustive-PQ-m48 (self)                               2_630.85    86_405.71    89_036.57       0.3772          NaN         7.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_859.65       592.79     6_452.44       0.4465          NaN         3.82
IVF-PQ-nl273-m16-np16 (query)                          5_859.65       710.02     6_569.67       0.4470          NaN         3.82
IVF-PQ-nl273-m16-np23 (query)                          5_859.65       978.41     6_838.07       0.4471          NaN         3.82
IVF-PQ-nl273-m16 (self)                                5_859.65    10_118.16    15_977.82       0.3455          NaN         3.82
IVF-PQ-nl273-m32-np13 (query)                          7_043.86     1_128.61     8_172.47       0.6221          NaN         6.11
IVF-PQ-nl273-m32-np16 (query)                          7_043.86     1_372.36     8_416.21       0.6233          NaN         6.11
IVF-PQ-nl273-m32-np23 (query)                          7_043.86     1_931.93     8_975.79       0.6235          NaN         6.11
IVF-PQ-nl273-m32 (self)                                7_043.86    19_250.79    26_294.65       0.5373          NaN         6.11
IVF-PQ-nl273-m48-np13 (query)                          7_364.48     1_369.47     8_733.95       0.7143          NaN         8.40
IVF-PQ-nl273-m48-np16 (query)                          7_364.48     1_748.49     9_112.97       0.7159          NaN         8.40
IVF-PQ-nl273-m48-np23 (query)                          7_364.48     2_320.07     9_684.55       0.7163          NaN         8.40
IVF-PQ-nl273-m48 (self)                                7_364.48    23_470.09    30_834.57       0.6508          NaN         8.40
IVF-PQ-nl387-m16-np19 (query)                          7_766.84       816.05     8_582.90       0.4483          NaN         3.91
IVF-PQ-nl387-m16-np27 (query)                          7_766.84     1_109.87     8_876.71       0.4487          NaN         3.91
IVF-PQ-nl387-m16 (self)                                7_766.84    11_086.38    18_853.22       0.3477          NaN         3.91
IVF-PQ-nl387-m32-np19 (query)                          8_826.57     1_518.21    10_344.79       0.6238          NaN         6.20
IVF-PQ-nl387-m32-np27 (query)                          8_826.57     2_068.38    10_894.96       0.6245          NaN         6.20
IVF-PQ-nl387-m32 (self)                                8_826.57    20_625.05    29_451.62       0.5392          NaN         6.20
IVF-PQ-nl387-m48-np19 (query)                          9_274.76     1_769.01    11_043.77       0.7167          NaN         8.49
IVF-PQ-nl387-m48-np27 (query)                          9_274.76     2_439.74    11_714.50       0.7176          NaN         8.49
IVF-PQ-nl387-m48 (self)                                9_274.76    24_346.47    33_621.23       0.6517          NaN         8.49
IVF-PQ-nl547-m16-np23 (query)                         10_200.18       952.83    11_153.01       0.4509          NaN         4.03
IVF-PQ-nl547-m16-np27 (query)                         10_200.18     1_076.90    11_277.08       0.4512          NaN         4.03
IVF-PQ-nl547-m16-np33 (query)                         10_200.18     1_252.34    11_452.51       0.4513          NaN         4.03
IVF-PQ-nl547-m16 (self)                               10_200.18    12_616.90    22_817.08       0.3504          NaN         4.03
IVF-PQ-nl547-m32-np23 (query)                         11_084.42     1_709.67    12_794.09       0.6247          NaN         6.32
IVF-PQ-nl547-m32-np27 (query)                         11_084.42     1_982.80    13_067.22       0.6255          NaN         6.32
IVF-PQ-nl547-m32-np33 (query)                         11_084.42     2_396.02    13_480.43       0.6258          NaN         6.32
IVF-PQ-nl547-m32 (self)                               11_084.42    26_420.99    37_505.40       0.5409          NaN         6.32
IVF-PQ-nl547-m48-np23 (query)                         11_886.55     1_981.03    13_867.59       0.7164          NaN         8.60
IVF-PQ-nl547-m48-np27 (query)                         11_886.55     2_309.57    14_196.13       0.7177          NaN         8.60
IVF-PQ-nl547-m48-np33 (query)                         11_886.55     2_790.30    14_676.86       0.7182          NaN         8.60
IVF-PQ-nl547-m48 (self)                               11_886.55    27_788.73    39_675.28       0.6530          NaN         8.60
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.98     9_902.07     9_924.05       1.0000     0.000000       109.86
Exhaustive (self)                                         21.98   101_130.39   101_152.38       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_376.81     1_944.73     3_321.54       0.3902          NaN         2.48
Exhaustive-PQ-m16 (self)                               1_376.81    19_332.36    20_709.16       0.2914          NaN         2.48
Exhaustive-PQ-m32 (query)                              2_230.68     4_445.30     6_675.97       0.5291          NaN         4.77
Exhaustive-PQ-m32 (self)                               2_230.68    44_495.64    46_726.31       0.4226          NaN         4.77
Exhaustive-PQ-m48 (query)                              2_732.95     8_847.11    11_580.06       0.6084          NaN         7.06
Exhaustive-PQ-m48 (self)                               2_732.95    88_608.76    91_341.71       0.5172          NaN         7.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_962.86       611.47     6_574.33       0.6602          NaN         3.82
IVF-PQ-nl273-m16-np16 (query)                          5_962.86       740.78     6_703.64       0.6603          NaN         3.82
IVF-PQ-nl273-m16-np23 (query)                          5_962.86     1_022.21     6_985.07       0.6603          NaN         3.82
IVF-PQ-nl273-m16 (self)                                5_962.86    10_144.92    16_107.79       0.5315          NaN         3.82
IVF-PQ-nl273-m32-np13 (query)                          6_912.43     1_118.63     8_031.06       0.7925          NaN         6.11
IVF-PQ-nl273-m32-np16 (query)                          6_912.43     1_349.33     8_261.76       0.7925          NaN         6.11
IVF-PQ-nl273-m32-np23 (query)                          6_912.43     1_907.04     8_819.47       0.7925          NaN         6.11
IVF-PQ-nl273-m32 (self)                                6_912.43    19_120.57    26_033.00       0.7186          NaN         6.11
IVF-PQ-nl273-m48-np13 (query)                          7_206.33     1_380.44     8_586.77       0.8685          NaN         8.40
IVF-PQ-nl273-m48-np16 (query)                          7_206.33     1_688.00     8_894.33       0.8686          NaN         8.40
IVF-PQ-nl273-m48-np23 (query)                          7_206.33     2_346.02     9_552.35       0.8686          NaN         8.40
IVF-PQ-nl273-m48 (self)                                7_206.33    23_381.84    30_588.18       0.8213          NaN         8.40
IVF-PQ-nl387-m16-np19 (query)                          7_573.71       807.63     8_381.33       0.6626          NaN         3.91
IVF-PQ-nl387-m16-np27 (query)                          7_573.71     1_107.49     8_681.20       0.6626          NaN         3.91
IVF-PQ-nl387-m16 (self)                                7_573.71    10_870.09    18_443.79       0.5271          NaN         3.91
IVF-PQ-nl387-m32-np19 (query)                          8_428.44     1_479.22     9_907.66       0.7947          NaN         6.20
IVF-PQ-nl387-m32-np27 (query)                          8_428.44     2_071.82    10_500.26       0.7947          NaN         6.20
IVF-PQ-nl387-m32 (self)                                8_428.44    20_493.37    28_921.81       0.7184          NaN         6.20
IVF-PQ-nl387-m48-np19 (query)                          9_164.55     1_760.42    10_924.97       0.8698          NaN         8.49
IVF-PQ-nl387-m48-np27 (query)                          9_164.55     2_457.15    11_621.70       0.8698          NaN         8.49
IVF-PQ-nl387-m48 (self)                                9_164.55    24_605.18    33_769.72       0.8226          NaN         8.49
IVF-PQ-nl547-m16-np23 (query)                         10_096.92       942.09    11_039.01       0.6644          NaN         4.03
IVF-PQ-nl547-m16-np27 (query)                         10_096.92     1_046.87    11_143.79       0.6644          NaN         4.03
IVF-PQ-nl547-m16-np33 (query)                         10_096.92     1_223.18    11_320.09       0.6644          NaN         4.03
IVF-PQ-nl547-m16 (self)                               10_096.92    12_322.90    22_419.82       0.5214          NaN         4.03
IVF-PQ-nl547-m32-np23 (query)                         11_069.94     1_699.82    12_769.76       0.7972          NaN         6.32
IVF-PQ-nl547-m32-np27 (query)                         11_069.94     1_969.57    13_039.51       0.7972          NaN         6.32
IVF-PQ-nl547-m32-np33 (query)                         11_069.94     2_393.01    13_462.95       0.7972          NaN         6.32
IVF-PQ-nl547-m32 (self)                               11_069.94    23_869.86    34_939.80       0.7181          NaN         6.32
IVF-PQ-nl547-m48-np23 (query)                         11_672.84     1_989.69    13_662.53       0.8716          NaN         8.60
IVF-PQ-nl547-m48-np27 (query)                         11_672.84     2_319.18    13_992.02       0.8716          NaN         8.60
IVF-PQ-nl547-m48-np33 (query)                         11_672.84     2_793.90    14_466.74       0.8716          NaN         8.60
IVF-PQ-nl547-m48 (self)                               11_672.84    27_878.40    39_551.24       0.8241          NaN         8.60
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Optimised product quantisation (Exhaustive and IVF)

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

#### Why IVF massively outperforms Exhaustive OPQ

A key observation is the large outperformance of the IVF index over the
exhaustive index. This is not incidental - it is fundamental to how PQ works,
and OPQ does not change this.

Product quantisation divides each vector into m subvectors and quantises each
to one of 256 centroids. The quality of this approximation depends critically
on the variance of the data being quantised: lower variance means the 256
centroids can tile the space more densely, yielding smaller quantisation error.

**IVF-OPQ** first clusters the dataset, then encodes residuals (vector minus
cluster centroid) rather than raw vectors. Vectors within a cluster are similar,
so their residuals are small, tightly distributed around zero, and share
correlated structure. The OPQ codebooks can represent these local patterns
efficiently.

**Exhaustive-OPQ** must encode raw vectors directly. Whilst OPQ learns a rotation
to make subspaces more independent (reducing cross-subspace correlation), it
does not reduce the overall spread of the data. The codebooks must still
represent the entire dataset's diversity - wildly different vectors compete for
the same 256 centroids per subspace. This leads to fundamentally higher
quantisation error.

In short: IVF's clustering creates locality, and locality is what PQ needs
to quantise accurately. OPQ's rotation improves subspace independence but does
not create locality - it transforms the data without reducing its intrinsic
spread. The clustering step is not optional for high-recall quantised search.

#### With 128 dimensions

Due to being optimised for high dimensional data, we start with 128 dimensions
here. One can appreciate that `m16` causes a high Recall loss as it compresses
the data too much and the more "structure" the data has (correlated features,
low rank manifolds), the better these indices perform. The PQ quantisation
however is a bit more expensive, hence, more time is being used on index
building. Querying on the other hand is very performant.

<details>
<summary><b>OPQ quantisations - Euclidean (Gaussian - 128 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.57     6_415.67     6_430.25       1.0000     0.000000        73.24
Exhaustive (self)                                         14.57    66_921.32    66_935.89       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_787.10     1_922.26     5_709.36       0.1326          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_787.10    20_321.21    24_108.31       0.0961          NaN         2.48
Exhaustive-OPQ-m32 (query)                             5_923.59     4_407.74    10_331.33       0.2489          NaN         4.77
Exhaustive-OPQ-m32 (self)                              5_923.59    46_067.78    51_991.37       0.1715          NaN         4.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         6_799.24       538.77     7_338.01       0.2665          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         6_799.24       638.28     7_437.53       0.2684          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         6_799.24       909.53     7_708.78       0.2694          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               6_799.24     9_714.79    16_514.03       0.1795          NaN         3.76
IVF-OPQ-nl273-m32-np13 (query)                         9_015.13       932.21     9_947.34       0.5091          NaN         6.05
IVF-OPQ-nl273-m32-np16 (query)                         9_015.13     1_112.76    10_127.89       0.5169          NaN         6.05
IVF-OPQ-nl273-m32-np23 (query)                         9_015.13     1_543.90    10_559.04       0.5216          NaN         6.05
IVF-OPQ-nl273-m32 (self)                               9_015.13    15_684.28    24_699.41       0.4278          NaN         6.05
IVF-OPQ-nl387-m16-np19 (query)                         7_770.55       651.17     8_421.72       0.2691          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         7_770.55       917.79     8_688.34       0.2707          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               7_770.55     9_155.99    16_926.54       0.1804          NaN         3.81
IVF-OPQ-nl387-m32-np19 (query)                         9_488.33     1_131.79    10_620.12       0.5155          NaN         6.10
IVF-OPQ-nl387-m32-np27 (query)                         9_488.33     1_581.39    11_069.72       0.5229          NaN         6.10
IVF-OPQ-nl387-m32 (self)                               9_488.33    16_586.72    26_075.04       0.4302          NaN         6.10
IVF-OPQ-nl547-m16-np23 (query)                        10_133.06       796.09    10_929.15       0.2705          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                        10_133.06       935.90    11_068.96       0.2720          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                        10_133.06     1_067.83    11_200.89       0.2730          NaN         3.89
IVF-OPQ-nl547-m16 (self)                              10_133.06    10_962.36    21_095.41       0.1822          NaN         3.89
IVF-OPQ-nl547-m32-np23 (query)                        11_462.83     1_340.46    12_803.29       0.5138          NaN         6.18
IVF-OPQ-nl547-m32-np27 (query)                        11_462.83     1_515.49    12_978.31       0.5199          NaN         6.18
IVF-OPQ-nl547-m32-np33 (query)                        11_462.83     1_823.08    13_285.91       0.5246          NaN         6.18
IVF-OPQ-nl547-m32 (self)                              11_462.83    18_517.12    29_979.95       0.4320          NaN         6.18
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>OPQ quantisations - Euclidean (Correlated - 128 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.49     6_197.47     6_212.96       1.0000     0.000000        73.24
Exhaustive (self)                                         15.49    62_200.27    62_215.76       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_750.41     1_881.84     5_632.25       0.3515          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_750.41    19_114.87    22_865.28       0.2620          NaN         2.48
Exhaustive-OPQ-m32 (query)                             5_450.95     4_318.98     9_769.93       0.4805          NaN         4.77
Exhaustive-OPQ-m32 (self)                              5_450.95    43_467.39    48_918.35       0.3934          NaN         4.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         6_515.03       522.60     7_037.62       0.5515          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         6_515.03       631.93     7_146.95       0.5521          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         6_515.03       886.12     7_401.15       0.5522          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               6_515.03     9_071.82    15_586.85       0.4539          NaN         3.76
IVF-OPQ-nl273-m32-np13 (query)                         8_379.36       898.88     9_278.24       0.7146          NaN         6.05
IVF-OPQ-nl273-m32-np16 (query)                         8_379.36     1_255.61     9_634.98       0.7160          NaN         6.05
IVF-OPQ-nl273-m32-np23 (query)                         8_379.36     1_535.47     9_914.83       0.7163          NaN         6.05
IVF-OPQ-nl273-m32 (self)                               8_379.36    15_627.18    24_006.54       0.6479          NaN         6.05
IVF-OPQ-nl387-m16-np19 (query)                         8_231.66       688.86     8_920.52       0.5537          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         8_231.66       965.83     9_197.49       0.5541          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               8_231.66     9_724.63    17_956.28       0.4563          NaN         3.81
IVF-OPQ-nl387-m32-np19 (query)                         9_873.21     1_161.78    11_035.00       0.7181          NaN         6.10
IVF-OPQ-nl387-m32-np27 (query)                         9_873.21     1_606.63    11_479.85       0.7191          NaN         6.10
IVF-OPQ-nl387-m32 (self)                               9_873.21    16_401.53    26_274.74       0.6498          NaN         6.10
IVF-OPQ-nl547-m16-np23 (query)                        10_502.84       828.09    11_330.93       0.5573          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                        10_502.84       956.02    11_458.86       0.5577          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                        10_502.84     1_113.01    11_615.85       0.5580          NaN         3.89
IVF-OPQ-nl547-m16 (self)                              10_502.84    11_018.82    21_521.65       0.4599          NaN         3.89
IVF-OPQ-nl547-m32-np23 (query)                        11_802.33     1_398.65    13_200.98       0.7175          NaN         6.18
IVF-OPQ-nl547-m32-np27 (query)                        11_802.33     1_538.39    13_340.72       0.7186          NaN         6.18
IVF-OPQ-nl547-m32-np33 (query)                        11_802.33     1_809.27    13_611.60       0.7192          NaN         6.18
IVF-OPQ-nl547-m32 (self)                              11_802.33    19_087.62    30_889.96       0.6516          NaN         6.18
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 128 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.93     6_934.27     6_950.20       1.0000     0.000000        73.24
Exhaustive (self)                                         15.93    67_947.50    67_963.43       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_730.01     1_884.72     5_614.73       0.3956          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_730.01    19_226.64    22_956.65       0.2465          NaN         2.48
Exhaustive-OPQ-m32 (query)                             5_403.84     4_362.21     9_766.06       0.5482          NaN         4.77
Exhaustive-OPQ-m32 (self)                              5_403.84    43_680.75    49_084.59       0.4057          NaN         4.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         6_614.22       498.75     7_112.97       0.7458          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         6_614.22       602.04     7_216.26       0.7458          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         6_614.22       852.63     7_466.85       0.7458          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               6_614.22     8_622.93    15_237.15       0.6682          NaN         3.76
IVF-OPQ-nl273-m32-np13 (query)                         8_303.00       860.73     9_163.73       0.8586          NaN         6.05
IVF-OPQ-nl273-m32-np16 (query)                         8_303.00     1_064.45     9_367.45       0.8587          NaN         6.05
IVF-OPQ-nl273-m32-np23 (query)                         8_303.00     1_469.40     9_772.40       0.8587          NaN         6.05
IVF-OPQ-nl273-m32 (self)                               8_303.00    15_033.61    23_336.61       0.8071          NaN         6.05
IVF-OPQ-nl387-m16-np19 (query)                         7_704.95       660.65     8_365.59       0.7468          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         7_704.95       913.31     8_618.25       0.7468          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               7_704.95     9_404.67    17_109.62       0.6706          NaN         3.81
IVF-OPQ-nl387-m32-np19 (query)                         9_507.78     1_152.05    10_659.83       0.8610          NaN         6.10
IVF-OPQ-nl387-m32-np27 (query)                         9_507.78     1_593.57    11_101.35       0.8610          NaN         6.10
IVF-OPQ-nl387-m32 (self)                               9_507.78    16_157.03    25_664.82       0.8095          NaN         6.10
IVF-OPQ-nl547-m16-np23 (query)                         9_298.59       789.54    10_088.13       0.7482          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         9_298.59       859.45    10_158.05       0.7482          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         9_298.59     1_038.94    10_337.54       0.7482          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               9_298.59    11_126.13    20_424.72       0.6729          NaN         3.89
IVF-OPQ-nl547-m32-np23 (query)                        11_863.08     1_491.80    13_354.87       0.8629          NaN         6.18
IVF-OPQ-nl547-m32-np27 (query)                        11_863.08     1_667.02    13_530.09       0.8629          NaN         6.18
IVF-OPQ-nl547-m32-np33 (query)                        11_863.08     2_137.88    14_000.96       0.8629          NaN         6.18
IVF-OPQ-nl547-m32 (self)                              11_863.08    22_644.34    34_507.41       0.8118          NaN         6.18
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 192 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is not too great, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        23.62    13_090.89    13_114.51       1.0000     0.000000       109.86
Exhaustive (self)                                         23.62   113_132.50   113_156.12       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             5_714.70     2_008.11     7_722.81       0.1067          NaN         2.62
Exhaustive-OPQ-m16 (self)                              5_714.70    21_068.90    26_783.60       0.0859          NaN         2.62
Exhaustive-OPQ-m32 (query)                             7_890.77     4_494.23    12_385.00       0.1596          NaN         4.91
Exhaustive-OPQ-m32 (self)                              7_890.77    45_712.93    53_603.70       0.1091          NaN         4.91
Exhaustive-OPQ-m48 (query)                             8_869.02     9_609.73    18_478.75       0.2528          NaN         7.20
Exhaustive-OPQ-m48 (self)                              8_869.02    92_011.81   100_880.83       0.1767          NaN         7.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                        10_944.46       661.93    11_606.39       0.1935          NaN         3.96
IVF-OPQ-nl273-m16-np16 (query)                        10_944.46       816.70    11_761.15       0.1941          NaN         3.96
IVF-OPQ-nl273-m16-np23 (query)                        10_944.46     1_108.20    12_052.65       0.1942          NaN         3.96
IVF-OPQ-nl273-m16 (self)                              10_944.46    10_954.25    21_898.70       0.1275          NaN         3.96
IVF-OPQ-nl273-m32-np13 (query)                        11_810.16     1_061.97    12_872.14       0.3588          NaN         6.25
IVF-OPQ-nl273-m32-np16 (query)                        11_810.16     1_441.02    13_251.19       0.3611          NaN         6.25
IVF-OPQ-nl273-m32-np23 (query)                        11_810.16     2_123.65    13_933.81       0.3616          NaN         6.25
IVF-OPQ-nl273-m32 (self)                              11_810.16    19_589.69    31_399.85       0.2660          NaN         6.25
IVF-OPQ-nl273-m48-np13 (query)                        13_589.29     1_487.86    15_077.15       0.5224          NaN         8.54
IVF-OPQ-nl273-m48-np16 (query)                        13_589.29     1_847.55    15_436.84       0.5270          NaN         8.54
IVF-OPQ-nl273-m48-np23 (query)                        13_589.29     2_513.75    16_103.04       0.5282          NaN         8.54
IVF-OPQ-nl273-m48 (self)                              13_589.29    25_363.22    38_952.51       0.4406          NaN         8.54
IVF-OPQ-nl387-m16-np19 (query)                        11_903.72       776.30    12_680.02       0.1943          NaN         4.05
IVF-OPQ-nl387-m16-np27 (query)                        11_903.72     1_030.27    12_933.99       0.1948          NaN         4.05
IVF-OPQ-nl387-m16 (self)                              11_903.72    11_472.90    23_376.62       0.1285          NaN         4.05
IVF-OPQ-nl387-m32-np19 (query)                        14_762.19     1_540.20    16_302.39       0.3613          NaN         6.34
IVF-OPQ-nl387-m32-np27 (query)                        14_762.19     2_076.81    16_839.01       0.3632          NaN         6.34
IVF-OPQ-nl387-m32 (self)                              14_762.19    21_805.35    36_567.54       0.2673          NaN         6.34
IVF-OPQ-nl387-m48-np19 (query)                        15_165.24     2_070.05    17_235.29       0.5260          NaN         8.63
IVF-OPQ-nl387-m48-np27 (query)                        15_165.24     2_646.46    17_811.70       0.5301          NaN         8.63
IVF-OPQ-nl387-m48 (self)                              15_165.24    27_142.22    42_307.47       0.4422          NaN         8.63
IVF-OPQ-nl547-m16-np23 (query)                        15_329.34     1_131.20    16_460.53       0.1960          NaN         4.17
IVF-OPQ-nl547-m16-np27 (query)                        15_329.34     1_283.63    16_612.97       0.1966          NaN         4.17
IVF-OPQ-nl547-m16-np33 (query)                        15_329.34     1_413.11    16_742.45       0.1968          NaN         4.17
IVF-OPQ-nl547-m16 (self)                              15_329.34    14_115.18    29_444.52       0.1293          NaN         4.17
IVF-OPQ-nl547-m32-np23 (query)                        28_943.10     2_601.58    31_544.69       0.3609          NaN         6.46
IVF-OPQ-nl547-m32-np27 (query)                        28_943.10     2_401.69    31_344.79       0.3636          NaN         6.46
IVF-OPQ-nl547-m32-np33 (query)                        28_943.10     2_990.34    31_933.44       0.3647          NaN         6.46
IVF-OPQ-nl547-m32 (self)                              28_943.10    31_805.03    60_748.14       0.2692          NaN         6.46
IVF-OPQ-nl547-m48-np23 (query)                        22_107.62     2_060.03    24_167.66       0.5222          NaN         8.75
IVF-OPQ-nl547-m48-np27 (query)                        22_107.62     2_380.25    24_487.87       0.5274          NaN         8.75
IVF-OPQ-nl547-m48-np33 (query)                        22_107.62     2_740.67    24_848.29       0.5295          NaN         8.75
IVF-OPQ-nl547-m48 (self)                              22_107.62    30_243.30    52_350.92       0.4422          NaN         8.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.67    11_903.09    11_924.76       1.0000     0.000000       109.86
Exhaustive (self)                                         21.67   106_828.84   106_850.51       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             5_653.31     1_885.23     7_538.54       0.2672          NaN         2.62
Exhaustive-OPQ-m16 (self)                              5_653.31    19_495.33    25_148.65       0.1914          NaN         2.62
Exhaustive-OPQ-m32 (query)                             7_168.87     4_335.31    11_504.17       0.3913          NaN         4.91
Exhaustive-OPQ-m32 (self)                              7_168.87    45_264.03    52_432.89       0.3015          NaN         4.91
Exhaustive-OPQ-m48 (query)                             8_942.21     8_903.38    17_845.58       0.4762          NaN         7.20
Exhaustive-OPQ-m48 (self)                              8_942.21    93_159.05   102_101.26       0.3923          NaN         7.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         9_888.24       606.47    10_494.71       0.4530          NaN         3.96
IVF-OPQ-nl273-m16-np16 (query)                         9_888.24       722.08    10_610.31       0.4535          NaN         3.96
IVF-OPQ-nl273-m16-np23 (query)                         9_888.24     1_036.12    10_924.35       0.4535          NaN         3.96
IVF-OPQ-nl273-m16 (self)                               9_888.24    10_612.16    20_500.39       0.3548          NaN         3.96
IVF-OPQ-nl273-m32-np13 (query)                        12_381.58     1_119.69    13_501.27       0.6239          NaN         6.25
IVF-OPQ-nl273-m32-np16 (query)                        12_381.58     1_356.71    13_738.29       0.6250          NaN         6.25
IVF-OPQ-nl273-m32-np23 (query)                        12_381.58     1_904.86    14_286.44       0.6252          NaN         6.25
IVF-OPQ-nl273-m32 (self)                              12_381.58    19_820.93    32_202.51       0.5415          NaN         6.25
IVF-OPQ-nl273-m48-np13 (query)                        13_191.56     1_334.03    14_525.59       0.7174          NaN         8.54
IVF-OPQ-nl273-m48-np16 (query)                        13_191.56     1_609.16    14_800.73       0.7192          NaN         8.54
IVF-OPQ-nl273-m48-np23 (query)                        13_191.56     2_274.42    15_465.98       0.7195          NaN         8.54
IVF-OPQ-nl273-m48 (self)                              13_191.56    24_471.24    37_662.81       0.6565          NaN         8.54
IVF-OPQ-nl387-m16-np19 (query)                        12_182.86       755.82    12_938.69       0.4553          NaN         4.05
IVF-OPQ-nl387-m16-np27 (query)                        12_182.86     1_029.08    13_211.94       0.4556          NaN         4.05
IVF-OPQ-nl387-m16 (self)                              12_182.86    10_924.19    23_107.05       0.3577          NaN         4.05
IVF-OPQ-nl387-m32-np19 (query)                        13_552.89     1_463.14    15_016.03       0.6259          NaN         6.34
IVF-OPQ-nl387-m32-np27 (query)                        13_552.89     2_051.78    15_604.68       0.6265          NaN         6.34
IVF-OPQ-nl387-m32 (self)                              13_552.89    21_256.92    34_809.81       0.5436          NaN         6.34
IVF-OPQ-nl387-m48-np19 (query)                        14_829.41     1_719.21    16_548.62       0.7200          NaN         8.63
IVF-OPQ-nl387-m48-np27 (query)                        14_829.41     2_400.28    17_229.69       0.7209          NaN         8.63
IVF-OPQ-nl387-m48 (self)                              14_829.41    24_584.54    39_413.95       0.6576          NaN         8.63
IVF-OPQ-nl547-m16-np23 (query)                        14_222.44       892.33    15_114.77       0.4570          NaN         4.17
IVF-OPQ-nl547-m16-np27 (query)                        14_222.44     1_018.54    15_240.98       0.4574          NaN         4.17
IVF-OPQ-nl547-m16-np33 (query)                        14_222.44     1_235.93    15_458.37       0.4574          NaN         4.17
IVF-OPQ-nl547-m16 (self)                              14_222.44    12_747.55    26_969.99       0.3599          NaN         4.17
IVF-OPQ-nl547-m32-np23 (query)                        16_309.04     1_688.83    17_997.88       0.6269          NaN         6.46
IVF-OPQ-nl547-m32-np27 (query)                        16_309.04     1_940.19    18_249.23       0.6278          NaN         6.46
IVF-OPQ-nl547-m32-np33 (query)                        16_309.04     2_356.58    18_665.63       0.6279          NaN         6.46
IVF-OPQ-nl547-m32 (self)                              16_309.04    24_060.86    40_369.90       0.5454          NaN         6.46
IVF-OPQ-nl547-m48-np23 (query)                        17_193.85     1_967.30    19_161.15       0.7192          NaN         8.75
IVF-OPQ-nl547-m48-np27 (query)                        17_193.85     2_375.65    19_569.49       0.7206          NaN         8.75
IVF-OPQ-nl547-m48-np33 (query)                        17_193.85     2_766.91    19_960.76       0.7209          NaN         8.75
IVF-OPQ-nl547-m48 (self)                              17_193.85    27_924.27    45_118.12       0.6585          NaN         8.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 192 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 192D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        22.00     9_894.14     9_916.15       1.0000     0.000000       109.86
Exhaustive (self)                                         22.00    98_649.24    98_671.24       1.0000     0.000000       109.86
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             5_901.11     1_920.89     7_822.01       0.3898          NaN         2.62
Exhaustive-OPQ-m16 (self)                              5_901.11    19_548.53    25_449.64       0.2344          NaN         2.62
Exhaustive-OPQ-m32 (query)                             7_265.38     4_654.62    11_920.00       0.5299          NaN         4.91
Exhaustive-OPQ-m32 (self)                              7_265.38    43_824.77    51_090.15       0.3785          NaN         4.91
Exhaustive-OPQ-m48 (query)                             8_231.72     8_607.27    16_838.99       0.6092          NaN         7.20
Exhaustive-OPQ-m48 (self)                              8_231.72    89_864.94    98_096.65       0.4875          NaN         7.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         9_958.25       569.96    10_528.21       0.7168          NaN         3.96
IVF-OPQ-nl273-m16-np16 (query)                         9_958.25       681.47    10_639.71       0.7168          NaN         3.96
IVF-OPQ-nl273-m16-np23 (query)                         9_958.25       951.23    10_909.48       0.7168          NaN         3.96
IVF-OPQ-nl273-m16 (self)                               9_958.25    10_097.27    20_055.52       0.6340          NaN         3.96
IVF-OPQ-nl273-m32-np13 (query)                        11_613.27     1_072.64    12_685.91       0.8280          NaN         6.25
IVF-OPQ-nl273-m32-np16 (query)                        11_613.27     1_296.34    12_909.61       0.8281          NaN         6.25
IVF-OPQ-nl273-m32-np23 (query)                        11_613.27     1_832.77    13_446.04       0.8281          NaN         6.25
IVF-OPQ-nl273-m32 (self)                              11_613.27    18_959.47    30_572.74       0.7651          NaN         6.25
IVF-OPQ-nl273-m48-np13 (query)                        12_535.43     1_328.54    13_863.97       0.8854          NaN         8.54
IVF-OPQ-nl273-m48-np16 (query)                        12_535.43     1_649.29    14_184.72       0.8855          NaN         8.54
IVF-OPQ-nl273-m48-np23 (query)                        12_535.43     2_304.33    14_839.76       0.8855          NaN         8.54
IVF-OPQ-nl273-m48 (self)                              12_535.43    23_481.73    36_017.16       0.8386          NaN         8.54
IVF-OPQ-nl387-m16-np19 (query)                        11_510.81       764.12    12_274.94       0.7190          NaN         4.05
IVF-OPQ-nl387-m16-np27 (query)                        11_510.81     1_036.09    12_546.90       0.7190          NaN         4.05
IVF-OPQ-nl387-m16 (self)                              11_510.81    10_969.64    22_480.46       0.6355          NaN         4.05
IVF-OPQ-nl387-m32-np19 (query)                        13_433.64     1_459.26    14_892.90       0.8284          NaN         6.34
IVF-OPQ-nl387-m32-np27 (query)                        13_433.64     2_018.37    15_452.01       0.8284          NaN         6.34
IVF-OPQ-nl387-m32 (self)                              13_433.64    20_671.85    34_105.49       0.7674          NaN         6.34
IVF-OPQ-nl387-m48-np19 (query)                        14_400.03     1_726.80    16_126.83       0.8851          NaN         8.63
IVF-OPQ-nl387-m48-np27 (query)                        14_400.03     2_423.41    16_823.45       0.8851          NaN         8.63
IVF-OPQ-nl387-m48 (self)                              14_400.03    24_709.57    39_109.60       0.8400          NaN         8.63
IVF-OPQ-nl547-m16-np23 (query)                        14_083.46       884.61    14_968.07       0.7195          NaN         4.17
IVF-OPQ-nl547-m16-np27 (query)                        14_083.46     1_039.62    15_123.08       0.7195          NaN         4.17
IVF-OPQ-nl547-m16-np33 (query)                        14_083.46     1_216.44    15_299.90       0.7195          NaN         4.17
IVF-OPQ-nl547-m16 (self)                              14_083.46    12_658.10    26_741.56       0.6378          NaN         4.17
IVF-OPQ-nl547-m32-np23 (query)                        16_052.92     1_671.76    17_724.68       0.8272          NaN         6.46
IVF-OPQ-nl547-m32-np27 (query)                        16_052.92     1_925.60    17_978.52       0.8273          NaN         6.46
IVF-OPQ-nl547-m32-np33 (query)                        16_052.92     2_328.07    18_380.99       0.8273          NaN         6.46
IVF-OPQ-nl547-m32 (self)                              16_052.92    24_015.75    40_068.67       0.7681          NaN         6.46
IVF-OPQ-nl547-m48-np23 (query)                        17_699.54     1_987.37    19_686.90       0.8851          NaN         8.75
IVF-OPQ-nl547-m48-np27 (query)                        17_699.54     2_327.63    20_027.16       0.8852          NaN         8.75
IVF-OPQ-nl547-m48-np33 (query)                        17_699.54     2_785.07    20_484.60       0.8852          NaN         8.75
IVF-OPQ-nl547-m48 (self)                              17_699.54    28_383.11    46_082.64       0.8412          NaN         8.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
