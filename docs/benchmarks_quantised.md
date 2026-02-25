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
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.26     1_577.74     1_581.00       1.0000     0.000000        18.31
Exhaustive (self)                                          3.26    16_492.07    16_495.33       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    4.95     1_271.59     1_276.53       0.9867     0.065388         9.16
Exhaustive-BF16 (self)                                     4.95    17_189.70    17_194.65       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              856.10       133.46       989.56       0.9770     0.096258        10.34
IVF-BF16-nl273-np16 (query)                              856.10       160.08     1_016.18       0.9839     0.071045        10.34
IVF-BF16-nl273-np23 (query)                              856.10       216.18     1_072.28       0.9867     0.065388        10.34
IVF-BF16-nl273 (self)                                    856.10     2_209.46     3_065.56       0.9830     0.094689        10.34
IVF-BF16-nl387-np19 (query)                            1_177.33       139.11     1_316.44       0.9804     0.093345        10.35
IVF-BF16-nl387-np27 (query)                            1_177.33       197.33     1_374.66       0.9865     0.066036        10.35
IVF-BF16-nl387 (self)                                  1_177.33     1_910.45     3_087.78       0.9828     0.095270        10.35
IVF-BF16-nl547-np23 (query)                            1_639.72       121.29     1_761.01       0.9765     0.103120        10.37
IVF-BF16-nl547-np27 (query)                            1_639.72       139.91     1_779.63       0.9833     0.078310        10.37
IVF-BF16-nl547-np33 (query)                            1_639.72       173.03     1_812.75       0.9864     0.066442        10.37
IVF-BF16-nl547 (self)                                  1_639.72     1_676.20     3_315.93       0.9827     0.095533        10.37
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
Exhaustive (query)                                         3.99     1_659.68     1_663.67       1.0000     0.000000        18.88
Exhaustive (self)                                          3.99    16_059.01    16_063.01       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.91     1_277.85     1_283.76       0.9240     0.000230         9.44
Exhaustive-BF16 (self)                                     5.91    15_799.40    15_805.31       1.0000     0.000000         9.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              780.61       144.12       924.73       0.9176     0.000245        10.62
IVF-BF16-nl273-np16 (query)                              780.61       177.34       957.95       0.9222     0.000233        10.62
IVF-BF16-nl273-np23 (query)                              780.61       248.44     1_029.05       0.9240     0.000230        10.62
IVF-BF16-nl273 (self)                                    780.61     2_526.08     3_306.69       0.9229     0.001251        10.62
IVF-BF16-nl387-np19 (query)                            1_067.49       154.03     1_221.52       0.9199     0.000242        10.64
IVF-BF16-nl387-np27 (query)                            1_067.49       213.34     1_280.83       0.9239     0.000230        10.64
IVF-BF16-nl387 (self)                                  1_067.49     2_169.41     3_236.90       0.9228     0.001251        10.64
IVF-BF16-nl547-np23 (query)                            1_495.26       134.50     1_629.76       0.9177     0.000245        10.66
IVF-BF16-nl547-np27 (query)                            1_495.26       160.32     1_655.58       0.9219     0.000235        10.66
IVF-BF16-nl547-np33 (query)                            1_495.26       188.99     1_684.25       0.9238     0.000231        10.66
IVF-BF16-nl547 (self)                                  1_495.26     1_886.93     3_382.19       0.9228     0.001251        10.66
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
Exhaustive (query)                                         3.19     1_672.13     1_675.33       1.0000     0.000000        18.31
Exhaustive (self)                                          3.19    17_348.00    17_351.19       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.05     1_257.38     1_262.43       0.9649     0.019029         9.16
Exhaustive-BF16 (self)                                     5.05    16_597.30    16_602.35       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              845.56       145.40       990.96       0.9648     0.019042        10.34
IVF-BF16-nl273-np16 (query)                              845.56       159.03     1_004.59       0.9649     0.019030        10.34
IVF-BF16-nl273-np23 (query)                              845.56       229.36     1_074.92       0.9649     0.019029        10.34
IVF-BF16-nl273 (self)                                    845.56     2_327.21     3_172.78       0.9561     0.026167        10.34
IVF-BF16-nl387-np19 (query)                            1_113.69       135.57     1_249.26       0.9649     0.019036        10.35
IVF-BF16-nl387-np27 (query)                            1_113.69       202.80     1_316.49       0.9649     0.019029        10.35
IVF-BF16-nl387 (self)                                  1_113.69     1_990.98     3_104.68       0.9561     0.026167        10.35
IVF-BF16-nl547-np23 (query)                            1_555.29       138.26     1_693.55       0.9649     0.019042        10.37
IVF-BF16-nl547-np27 (query)                            1_555.29       157.08     1_712.37       0.9649     0.019037        10.37
IVF-BF16-nl547-np33 (query)                            1_555.29       174.54     1_729.84       0.9649     0.019030        10.37
IVF-BF16-nl547 (self)                                  1_555.29     1_914.05     3_469.34       0.9561     0.026169        10.37
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
Exhaustive (query)                                         3.16     1_651.70     1_654.86       1.0000     0.000000        18.31
Exhaustive (self)                                          3.16    17_143.63    17_146.79       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    4.30     1_175.07     1_179.37       0.9348     0.158338         9.16
Exhaustive-BF16 (self)                                     4.30    16_351.05    16_355.35       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              797.43       139.09       936.51       0.9348     0.158344        10.34
IVF-BF16-nl273-np16 (query)                              797.43       170.45       967.88       0.9348     0.158338        10.34
IVF-BF16-nl273-np23 (query)                              797.43       228.30     1_025.72       0.9348     0.158338        10.34
IVF-BF16-nl273 (self)                                    797.43     2_352.94     3_150.36       0.9174     0.221294        10.34
IVF-BF16-nl387-np19 (query)                            1_148.23       160.94     1_309.17       0.9348     0.158338        10.35
IVF-BF16-nl387-np27 (query)                            1_148.23       218.10     1_366.33       0.9348     0.158338        10.35
IVF-BF16-nl387 (self)                                  1_148.23     2_019.00     3_167.23       0.9174     0.221294        10.35
IVF-BF16-nl547-np23 (query)                            1_609.95       135.05     1_745.01       0.9348     0.158344        10.37
IVF-BF16-nl547-np27 (query)                            1_609.95       142.29     1_752.24       0.9348     0.158344        10.37
IVF-BF16-nl547-np33 (query)                            1_609.95       189.69     1_799.64       0.9348     0.158338        10.37
IVF-BF16-nl547 (self)                                  1_609.95     1_785.23     3_395.19       0.9174     0.221294        10.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.58     6_631.10     6_645.68       1.0000     0.000000        73.24
Exhaustive (self)                                         14.58    67_046.09    67_060.67       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                   22.72     5_374.55     5_397.27       0.9714     0.573744        36.62
Exhaustive-BF16 (self)                                    22.72    66_987.95    67_010.67       1.0000     0.000000        36.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                            1_412.50       711.05     2_123.55       0.9712     0.575307        37.90
IVF-BF16-nl273-np16 (query)                            1_412.50       825.75     2_238.25       0.9714     0.573751        37.90
IVF-BF16-nl273-np23 (query)                            1_412.50     1_146.41     2_558.91       0.9714     0.573744        37.90
IVF-BF16-nl273 (self)                                  1_412.50    11_397.96    12_810.46       0.9637     0.816885        37.90
IVF-BF16-nl387-np19 (query)                            2_168.11       684.34     2_852.44       0.9713     0.573883        37.96
IVF-BF16-nl387-np27 (query)                            2_168.11       945.05     3_113.15       0.9714     0.573744        37.96
IVF-BF16-nl387 (self)                                  2_168.11     9_895.94    12_064.05       0.9637     0.816885        37.96
IVF-BF16-nl547-np23 (query)                            4_318.71       640.15     4_958.85       0.9713     0.574204        38.04
IVF-BF16-nl547-np27 (query)                            4_318.71       730.70     5_049.41       0.9713     0.573846        38.04
IVF-BF16-nl547-np33 (query)                            4_318.71       851.54     5_170.24       0.9714     0.573747        38.04
IVF-BF16-nl547 (self)                                  4_318.71     8_988.66    13_307.36       0.9637     0.816885        38.04
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
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.15     1_635.45     1_638.61       1.0000     0.000000        18.31
Exhaustive (self)                                          3.15    16_493.23    16_496.38       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.08       732.14       739.22       0.8011          NaN         4.58
Exhaustive-SQ8 (self)                                      7.08     7_965.73     7_972.81       0.8007          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               828.72        68.71       897.43       0.7971          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               828.72        76.06       904.78       0.8000          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               828.72        93.23       921.95       0.8011          NaN         5.76
IVF-SQ8-nl273 (self)                                     828.72       915.71     1_744.43       0.8007          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_175.67        63.87     1_239.53       0.7985          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_175.67        79.03     1_254.70       0.8011          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_175.67       770.32     1_945.99       0.8007          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_644.68        56.33     1_701.01       0.7968          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_644.68        65.38     1_710.05       0.7996          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_644.68        73.89     1_718.57       0.8010          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_644.68       698.74     2_343.42       0.8006          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.02     1_601.35     1_605.37       1.0000     0.000000        18.88
Exhaustive (self)                                          4.02    16_555.90    16_559.92       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     8.54       712.49       721.03       0.8501          NaN         5.15
Exhaustive-SQ8 (self)                                      8.54     7_564.72     7_573.26       0.8497          NaN         5.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               800.55        62.65       863.20       0.8456          NaN         6.33
IVF-SQ8-nl273-np16 (query)                               800.55        73.83       874.37       0.8487          NaN         6.33
IVF-SQ8-nl273-np23 (query)                               800.55        99.32       899.87       0.8501          NaN         6.33
IVF-SQ8-nl273 (self)                                     800.55       956.34     1_756.88       0.8497          NaN         6.33
IVF-SQ8-nl387-np19 (query)                             1_068.59        75.58     1_144.17       0.8473          NaN         6.34
IVF-SQ8-nl387-np27 (query)                             1_068.59       100.05     1_168.64       0.8500          NaN         6.34
IVF-SQ8-nl387 (self)                                   1_068.59       829.75     1_898.34       0.8497          NaN         6.34
IVF-SQ8-nl547-np23 (query)                             1_490.89        57.00     1_547.90       0.8457          NaN         6.37
IVF-SQ8-nl547-np27 (query)                             1_490.89        65.21     1_556.11       0.8487          NaN         6.37
IVF-SQ8-nl547-np33 (query)                             1_490.89        77.25     1_568.14       0.8499          NaN         6.37
IVF-SQ8-nl547 (self)                                   1_490.89       722.88     2_213.77       0.8496          NaN         6.37
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.34     1_624.74     1_628.08       1.0000     0.000000        18.31
Exhaustive (self)                                          3.34    16_434.44    16_437.77       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.00       738.26       745.26       0.6828          NaN         4.58
Exhaustive-SQ8 (self)                                      7.00     7_897.03     7_904.04       0.6835          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               800.02        61.95       861.97       0.6829          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               800.02        70.13       870.15       0.6829          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               800.02        96.01       896.03       0.6828          NaN         5.76
IVF-SQ8-nl273 (self)                                     800.02       944.07     1_744.08       0.6835          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_117.24        58.13     1_175.37       0.6829          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_117.24        76.29     1_193.54       0.6828          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_117.24       752.73     1_869.97       0.6835          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_584.89        55.50     1_640.39       0.6830          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_584.89        62.52     1_647.41       0.6830          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_584.89        78.51     1_663.40       0.6829          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_584.89       742.65     2_327.54       0.6835          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.11     1_588.76     1_591.87       1.0000     0.000000        18.31
Exhaustive (self)                                          3.11    16_139.32    16_142.43       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    17.04       789.19       806.23       0.4800          NaN         4.58
Exhaustive-SQ8 (self)                                     17.04     7_906.99     7_924.04       0.4862          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               806.24        55.72       861.96       0.4802          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               806.24        67.30       873.55       0.4801          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               806.24        84.48       890.72       0.4801          NaN         5.76
IVF-SQ8-nl273 (self)                                     806.24       838.21     1_644.45       0.4864          NaN         5.76
IVF-SQ8-nl387-np19 (query)                             1_129.47        56.56     1_186.04       0.4802          NaN         5.77
IVF-SQ8-nl387-np27 (query)                             1_129.47        74.87     1_204.34       0.4801          NaN         5.77
IVF-SQ8-nl387 (self)                                   1_129.47       735.57     1_865.05       0.4864          NaN         5.77
IVF-SQ8-nl547-np23 (query)                             1_540.66        55.02     1_595.68       0.4802          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_540.66        66.12     1_606.78       0.4801          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_540.66        72.50     1_613.15       0.4800          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_540.66       665.06     2_205.72       0.4864          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### More dimensions

With higher dimensions, the Recall in more structured, correlated data does
become better again.

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.33     6_671.36     6_686.69       1.0000     0.000000        73.24
Exhaustive (self)                                         15.33    66_390.79    66_406.12       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    36.63     1_731.03     1_767.66       0.8081          NaN        18.31
Exhaustive-SQ8 (self)                                     36.63    17_763.97    17_800.60       0.8095          NaN        18.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                             1_434.36       244.95     1_679.31       0.8081          NaN        19.59
IVF-SQ8-nl273-np16 (query)                             1_434.36       297.92     1_732.29       0.8081          NaN        19.59
IVF-SQ8-nl273-np23 (query)                             1_434.36       393.62     1_827.98       0.8082          NaN        19.59
IVF-SQ8-nl273 (self)                                   1_434.36     3_872.38     5_306.75       0.8096          NaN        19.59
IVF-SQ8-nl387-np19 (query)                             2_368.64       253.19     2_621.83       0.8081          NaN        19.65
IVF-SQ8-nl387-np27 (query)                             2_368.64       332.26     2_700.90       0.8081          NaN        19.65
IVF-SQ8-nl387 (self)                                   2_368.64     3_499.46     5_868.10       0.8096          NaN        19.65
IVF-SQ8-nl547-np23 (query)                             4_557.07       252.22     4_809.29       0.8081          NaN        19.73
IVF-SQ8-nl547-np27 (query)                             4_557.07       277.97     4_835.04       0.8081          NaN        19.73
IVF-SQ8-nl547-np33 (query)                             4_557.07       313.47     4_870.54       0.8081          NaN        19.73
IVF-SQ8-nl547 (self)                                   4_557.07     3_159.55     7_716.61       0.8095          NaN        19.73
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
Exhaustive (query)                                        14.83     6_403.73     6_418.56       1.0000     0.000000        73.24
Exhaustive (self)                                         14.83    67_236.90    67_251.72       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 918.31       899.10     1_817.40       0.0969          NaN         1.27
Exhaustive-PQ-m8 (self)                                  918.31     9_070.95     9_989.26       0.0835          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_088.06     1_974.98     3_063.04       0.1328          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_088.06    19_604.51    20_692.57       0.0967          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           3_814.04       318.47     4_132.51       0.1524          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           3_814.04       395.97     4_210.01       0.1529          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           3_814.04       556.58     4_370.62       0.1529          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 3_814.04     5_578.78     9_392.82       0.1046          NaN         2.55
IVF-PQ-nl273-m16-np13 (query)                          4_116.05       531.36     4_647.41       0.2690          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          4_116.05       623.52     4_739.57       0.2712          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          4_116.05       909.76     5_025.81       0.2723          NaN         3.69
IVF-PQ-nl273-m16 (self)                                4_116.05     8_885.03    13_001.08       0.1804          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           4_965.06       427.34     5_392.41       0.1534          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           4_965.06       593.84     5_558.90       0.1537          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 4_965.06     5_891.72    10_856.79       0.1044          NaN         2.61
IVF-PQ-nl387-m16-np19 (query)                          5_269.05       715.54     5_984.60       0.2696          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          5_269.05       951.16     6_220.21       0.2711          NaN         3.75
IVF-PQ-nl387-m16 (self)                                5_269.05     9_754.66    15_023.71       0.1812          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           8_919.24       521.31     9_440.56       0.1565          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           8_919.24       619.19     9_538.43       0.1569          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           8_919.24       726.74     9_645.98       0.1569          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 8_919.24     7_311.52    16_230.76       0.1060          NaN         2.69
IVF-PQ-nl547-m16-np23 (query)                          9_238.15       740.13     9_978.29       0.2706          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          9_238.15       852.72    10_090.88       0.2722          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          9_238.15     1_027.98    10_266.13       0.2732          NaN         3.83
IVF-PQ-nl547-m16 (self)                                9_238.15    10_701.30    19_939.45       0.1826          NaN         3.83
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 128 dim)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.78     6_753.52     6_768.30       1.0000     0.000000        73.24
Exhaustive (self)                                         14.78    67_387.57    67_402.35       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 811.50       929.68     1_741.18       0.2215          NaN         1.27
Exhaustive-PQ-m8 (self)                                  811.50     9_015.05     9_826.56       0.1616          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_073.82     1_912.71     2_986.54       0.3518          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_073.82    19_493.34    20_567.16       0.2621          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           2_867.64       313.03     3_180.67       0.3774          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           2_867.64       380.49     3_248.13       0.3776          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           2_867.64       551.25     3_418.89       0.3776          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 2_867.64     5_492.90     8_360.54       0.2766          NaN         2.55
IVF-PQ-nl273-m16-np13 (query)                          3_214.94       507.38     3_722.32       0.5509          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          3_214.94       629.01     3_843.95       0.5516          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          3_214.94       877.34     4_092.28       0.5518          NaN         3.69
IVF-PQ-nl273-m16 (self)                                3_214.94     8_825.98    12_040.93       0.4539          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           3_973.66       408.36     4_382.02       0.3818          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           3_973.66       576.80     4_550.47       0.3819          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 3_973.66     5_642.39     9_616.05       0.2799          NaN         2.61
IVF-PQ-nl387-m16-np19 (query)                          4_320.81       693.42     5_014.22       0.5522          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          4_320.81       977.47     5_298.28       0.5527          NaN         3.75
IVF-PQ-nl387-m16 (self)                                4_320.81     9_968.10    14_288.91       0.4564          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           7_115.02       511.33     7_626.34       0.3859          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           7_115.02       593.68     7_708.70       0.3860          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           7_115.02       700.88     7_815.90       0.3859          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 7_115.02     7_032.07    14_147.09       0.2834          NaN         2.69
IVF-PQ-nl547-m16-np23 (query)                          7_419.75       730.05     8_149.79       0.5560          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          7_419.75       895.86     8_315.61       0.5566          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          7_419.75     1_051.29     8_471.04       0.5568          NaN         3.83
IVF-PQ-nl547-m16 (self)                                7_419.75    10_537.23    17_956.98       0.4588          NaN         3.83
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
Exhaustive (query)                                        15.02     6_766.59     6_781.61       1.0000     0.000000        73.24
Exhaustive (self)                                         15.02    69_496.30    69_511.31       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                               1_038.48       902.51     1_940.98       0.2587          NaN         1.27
Exhaustive-PQ-m8 (self)                                1_038.48     9_010.15    10_048.63       0.1987          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_158.07     1_962.43     3_120.50       0.3957          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_158.07    19_612.81    20_770.88       0.2941          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           2_296.99       307.50     2_604.49       0.5495          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           2_296.99       395.34     2_692.33       0.5495          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           2_296.99       534.32     2_831.31       0.5495          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 2_296.99     5_384.36     7_681.35       0.4039          NaN         2.55
IVF-PQ-nl273-m16-np13 (query)                          2_563.19       517.13     3_080.31       0.6939          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          2_563.19       616.94     3_180.13       0.6940          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          2_563.19       862.76     3_425.95       0.6940          NaN         3.69
IVF-PQ-nl273-m16 (self)                                2_563.19     8_733.10    11_296.29       0.5875          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           3_183.39       425.47     3_608.87       0.5546          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           3_183.39       595.74     3_779.13       0.5546          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 3_183.39     5_855.83     9_039.22       0.4051          NaN         2.61
IVF-PQ-nl387-m16-np19 (query)                          3_584.27       690.66     4_274.93       0.6972          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          3_584.27       942.24     4_526.51       0.6972          NaN         3.75
IVF-PQ-nl387-m16 (self)                                3_584.27     9_518.45    13_102.72       0.5890          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           5_570.14       482.87     6_053.01       0.5599          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           5_570.14       551.46     6_121.59       0.5599          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           5_570.14       669.77     6_239.91       0.5599          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 5_570.14     7_022.28    12_592.42       0.4052          NaN         2.69
IVF-PQ-nl547-m16-np23 (query)                          6_000.63       744.41     6_745.05       0.7017          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          6_000.63       852.29     6_852.93       0.7017          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          6_000.63     1_057.82     7_058.46       0.7017          NaN         3.83
IVF-PQ-nl547-m16 (self)                                6_000.63    10_584.70    16_585.33       0.5896          NaN         3.83
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 256 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is just bad, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 256 dim)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.91    15_691.05    15_721.96       1.0000     0.000000       146.48
Exhaustive (self)                                         30.91   172_323.21   172_354.11       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              2_447.07     1_980.41     4_427.47       0.0968          NaN         2.54
Exhaustive-PQ-m16 (self)                               2_447.07    19_627.86    22_074.93       0.0817          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_376.45     4_497.39     6_873.84       0.1321          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_376.45    45_057.36    47_433.81       0.0941          NaN         4.83
Exhaustive-PQ-m64 (query)                              4_023.62    12_144.23    16_167.85       0.2669          NaN         9.41
Exhaustive-PQ-m64 (self)                               4_023.62   121_558.03   125_581.65       0.1882          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_957.15       609.03     6_566.18       0.1527          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          5_957.15       730.94     6_688.09       0.1531          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          5_957.15     1_032.95     6_990.10       0.1533          NaN         3.95
IVF-PQ-nl273-m16 (self)                                5_957.15    10_390.54    16_347.69       0.1039          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          6_250.61       960.89     7_211.51       0.2665          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          6_250.61     1_142.97     7_393.58       0.2686          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          6_250.61     1_616.93     7_867.55       0.2702          NaN         6.24
IVF-PQ-nl273-m32 (self)                                6_250.61    16_204.91    22_455.52       0.1840          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          7_982.05     1_823.22     9_805.28       0.5111          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          7_982.05     2_425.14    10_407.20       0.5178          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          7_982.05     3_213.65    11_195.70       0.5232          NaN        10.82
IVF-PQ-nl273-m64 (self)                                7_982.05    31_305.54    39_287.59       0.4364          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          9_060.20       830.46     9_890.67       0.1540          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          9_060.20     1_157.82    10_218.02       0.1543          NaN         4.06
IVF-PQ-nl387-m16 (self)                                9_060.20    11_198.65    20_258.85       0.1043          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          9_256.08     1_309.78    10_565.86       0.2698          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          9_256.08     1_812.29    11_068.37       0.2719          NaN         6.35
IVF-PQ-nl387-m32 (self)                                9_256.08    17_934.10    27_190.19       0.1851          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                         10_517.18     2_380.04    12_897.22       0.5147          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                         10_517.18     3_300.01    13_817.19       0.5223          NaN        10.93
IVF-PQ-nl387-m64 (self)                               10_517.18    32_696.92    43_214.10       0.4366          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                         13_642.27       899.42    14_541.69       0.1540          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                         13_642.27     1_036.45    14_678.72       0.1545          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                         13_642.27     1_250.84    14_893.11       0.1547          NaN         4.22
IVF-PQ-nl547-m16 (self)                               13_642.27    12_627.50    26_269.77       0.1053          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                         14_017.63     1_422.95    15_440.58       0.2693          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                         14_017.63     1_751.98    15_769.61       0.2714          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                         14_017.63     2_008.11    16_025.74       0.2723          NaN         6.51
IVF-PQ-nl547-m32 (self)                               14_017.63    20_170.96    34_188.58       0.1854          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         15_398.30     2_591.86    17_990.16       0.5134          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         15_398.30     3_026.78    18_425.08       0.5202          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         15_398.30     3_661.39    19_059.69       0.5232          NaN        11.09
IVF-PQ-nl547-m64 (self)                               15_398.30    36_550.71    51_949.01       0.4370          NaN        11.09
--------------------------------------------------------------------------------------------------------------------------------
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 256 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.25    15_886.87    15_917.11       1.0000     0.000000       146.48
Exhaustive (self)                                         30.25   159_055.72   159_085.97       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              2_058.41     1_947.40     4_005.81       0.2357          NaN         2.54
Exhaustive-PQ-m16 (self)                               2_058.41    19_570.59    21_628.99       0.1675          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_230.18     4_455.33     6_685.52       0.3554          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_230.18    45_556.69    47_786.88       0.2679          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_562.93    12_105.68    15_668.61       0.4755          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_562.93   121_916.10   125_479.03       0.3907          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          4_483.46       625.64     5_109.10       0.3867          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          4_483.46       767.13     5_250.58       0.3875          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          4_483.46     1_073.14     5_556.59       0.3879          NaN         3.95
IVF-PQ-nl273-m16 (self)                                4_483.46    11_138.71    15_622.17       0.2883          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          5_076.39     1_002.33     6_078.72       0.5552          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          5_076.39     1_176.25     6_252.64       0.5569          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          5_076.39     1_732.23     6_808.62       0.5577          NaN         6.24
IVF-PQ-nl273-m32 (self)                                5_076.39    17_313.22    22_389.61       0.4669          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          6_410.67     1_865.21     8_275.88       0.7091          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          6_410.67     2_283.75     8_694.42       0.7122          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          6_410.67     3_316.92     9_727.59       0.7137          NaN        10.82
IVF-PQ-nl273-m64 (self)                                6_410.67    32_794.40    39_205.07       0.6506          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          5_967.18       814.31     6_781.49       0.3881          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          5_967.18     1_136.70     7_103.88       0.3887          NaN         4.06
IVF-PQ-nl387-m16 (self)                                5_967.18    11_440.75    17_407.93       0.2912          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          6_585.95     1_325.14     7_911.09       0.5584          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          6_585.95     1_808.31     8_394.26       0.5601          NaN         6.35
IVF-PQ-nl387-m32 (self)                                6_585.95    19_119.36    25_705.32       0.4688          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                          8_569.15     2_848.09    11_417.24       0.7134          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          8_569.15     3_600.19    12_169.34       0.7159          NaN        10.93
IVF-PQ-nl387-m64 (self)                                8_569.15    32_699.99    41_269.14       0.6513          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          9_435.92       929.22    10_365.14       0.3892          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          9_435.92     1_063.29    10_499.21       0.3897          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          9_435.92     1_298.45    10_734.37       0.3900          NaN         4.22
IVF-PQ-nl547-m16 (self)                                9_435.92    13_184.01    22_619.93       0.2924          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                         10_107.61     1_485.00    11_592.61       0.5590          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                         10_107.61     1_687.34    11_794.95       0.5607          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                         10_107.61     2_127.99    12_235.60       0.5617          NaN         6.51
IVF-PQ-nl547-m32 (self)                               10_107.61    20_110.25    30_217.86       0.4705          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         11_593.97     2_604.13    14_198.10       0.7118          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         11_593.97     3_102.42    14_696.39       0.7145          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         11_593.97     4_051.02    15_644.99       0.7161          NaN        11.09
IVF-PQ-nl547-m64 (self)                               11_593.97    37_118.55    48_712.52       0.6529          NaN        11.09
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 256 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        31.76    15_932.86    15_964.62       1.0000     0.000000       146.48
Exhaustive (self)                                         31.76   162_143.97   162_175.73       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_601.12     1_997.22     3_598.35       0.3801          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_601.12    19_382.60    20_983.73       0.2823          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_154.19     4_630.67     6_784.86       0.5114          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_154.19    45_520.01    47_674.20       0.4053          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_909.28    12_959.99    16_869.27       0.6355          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_909.28   123_593.42   127_502.70       0.5576          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          4_701.77       588.73     5_290.50       0.6312          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          4_701.77       749.08     5_450.84       0.6312          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          4_701.77     1_034.38     5_736.15       0.6312          NaN         3.95
IVF-PQ-nl273-m16 (self)                                4_701.77    10_359.90    15_061.66       0.4847          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          4_743.75       976.32     5_720.08       0.7578          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          4_743.75     1_173.05     5_916.80       0.7578          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          4_743.75     1_664.63     6_408.39       0.7578          NaN         6.24
IVF-PQ-nl273-m32 (self)                                4_743.75    16_358.44    21_102.20       0.6650          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          6_159.18     1_800.04     7_959.21       0.8807          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          6_159.18     2_231.11     8_390.29       0.8808          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          6_159.18     3_019.90     9_179.08       0.8808          NaN        10.82
IVF-PQ-nl273-m64 (self)                                6_159.18    30_672.89    36_832.06       0.8376          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          5_081.51       809.75     5_891.26       0.6305          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          5_081.51     1_121.31     6_202.82       0.6305          NaN         4.06
IVF-PQ-nl387-m16 (self)                                5_081.51    11_214.26    16_295.77       0.4753          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          5_978.03     1_634.35     7_612.38       0.7585          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          5_978.03     2_246.01     8_224.04       0.7585          NaN         6.35
IVF-PQ-nl387-m32 (self)                                5_978.03    19_062.31    25_040.35       0.6607          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                          7_459.57     2_362.40     9_821.96       0.8824          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          7_459.57     3_563.12    11_022.68       0.8824          NaN        10.93
IVF-PQ-nl387-m64 (self)                                7_459.57    32_536.22    39_995.79       0.8385          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          8_704.12       900.85     9_604.97       0.6328          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          8_704.12     1_091.35     9_795.47       0.6328          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          8_704.12     1_245.05     9_949.17       0.6328          NaN         4.22
IVF-PQ-nl547-m16 (self)                                8_704.12    12_612.16    21_316.28       0.4665          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                          9_431.84     1_492.41    10_924.25       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                          9_431.84     1_696.16    11_127.99       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                          9_431.84     2_076.49    11_508.33       0.7595          NaN         6.51
IVF-PQ-nl547-m32 (self)                                9_431.84    19_491.20    28_923.04       0.6562          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         11_121.23     2_801.37    13_922.60       0.8834          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         11_121.23     3_409.98    14_531.21       0.8835          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         11_121.23     3_711.41    14_832.64       0.8835          NaN        11.09
IVF-PQ-nl547-m64 (self)                               11_121.23    38_982.79    50_104.01       0.8392          NaN        11.09
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
Exhaustive (query)                                        15.06     7_271.25     7_286.31       1.0000     0.000000        73.24
Exhaustive (self)                                         15.06    70_087.22    70_102.29       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              4_188.52       898.99     5_087.52       0.0961          NaN         1.33
Exhaustive-OPQ-m8 (self)                               4_188.52     9_238.12    13_426.65       0.0833          NaN         1.33
Exhaustive-OPQ-m16 (query)                             4_416.57     1_977.10     6_393.67       0.1326          NaN         2.48
Exhaustive-OPQ-m16 (self)                              4_416.57    19_618.84    24_035.41       0.0961          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          6_434.57       370.61     6_805.18       0.1527          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          6_434.57       473.49     6_908.06       0.1530          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          6_434.57       588.94     7_023.52       0.1531          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                6_434.57     5_456.05    11_890.62       0.1043          NaN         2.61
IVF-OPQ-nl273-m16-np13 (query)                         7_154.06       607.02     7_761.08       0.2672          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         7_154.06       774.62     7_928.68       0.2692          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         7_154.06       956.43     8_110.48       0.2700          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               7_154.06     8_881.70    16_035.75       0.1800          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          7_528.67       522.72     8_051.39       0.1529          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          7_528.67       584.31     8_112.98       0.1532          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                7_528.67     5_966.74    13_495.41       0.1046          NaN         2.67
IVF-OPQ-nl387-m16-np19 (query)                         8_814.02       673.59     9_487.60       0.2695          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         8_814.02       954.28     9_768.30       0.2709          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               8_814.02     9_433.48    18_247.49       0.1812          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                         12_930.83       544.62    13_475.45       0.1574          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                         12_930.83       706.14    13_636.97       0.1576          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                         12_930.83       776.24    13_707.07       0.1576          NaN         2.75
IVF-OPQ-nl547-m8 (self)                               12_930.83     7_833.08    20_763.91       0.1064          NaN         2.75
IVF-OPQ-nl547-m16-np23 (query)                        13_226.21       725.41    13_951.62       0.2690          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                        13_226.21       868.02    14_094.22       0.2705          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                        13_226.21     1_078.91    14_305.12       0.2715          NaN         3.89
IVF-OPQ-nl547-m16 (self)                              13_226.21    11_218.16    24_444.37       0.1824          NaN         3.89
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
Exhaustive (query)                                        14.96     6_853.67     6_868.63       1.0000     0.000000        73.24
Exhaustive (self)                                         14.96    69_818.19    69_833.15       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              4_172.51       895.58     5_068.09       0.2219          NaN         1.33
Exhaustive-OPQ-m8 (self)                               4_172.51     9_276.50    13_449.01       0.1616          NaN         1.33
Exhaustive-OPQ-m16 (query)                             4_604.57     1_932.71     6_537.28       0.3515          NaN         2.48
Exhaustive-OPQ-m16 (self)                              4_604.57    19_862.05    24_466.62       0.2620          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          5_220.59       339.84     5_560.42       0.3774          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          5_220.59       406.53     5_627.12       0.3776          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          5_220.59       571.36     5_791.94       0.3776          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                5_220.59     5_524.33    10_744.92       0.2761          NaN         2.61
IVF-OPQ-nl273-m16-np13 (query)                         6_063.87       538.89     6_602.75       0.5495          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         6_063.87       668.30     6_732.17       0.5502          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         6_063.87       911.85     6_975.72       0.5504          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               6_063.87     9_100.70    15_164.56       0.4538          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          6_481.61       442.36     6_923.97       0.3807          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          6_481.61       607.70     7_089.31       0.3807          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                6_481.61     6_226.28    12_707.89       0.2798          NaN         2.67
IVF-OPQ-nl387-m16-np19 (query)                         6_879.53       665.42     7_544.96       0.5528          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         6_879.53     1_003.61     7_883.14       0.5533          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               6_879.53     9_735.26    16_614.80       0.4568          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          9_196.40       482.17     9_678.57       0.3849          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          9_196.40       561.83     9_758.23       0.3849          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          9_196.40       687.08     9_883.48       0.3848          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                9_196.40     7_114.72    16_311.11       0.2832          NaN         2.75
IVF-OPQ-nl547-m16-np23 (query)                        10_061.41       727.27    10_788.68       0.5562          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                        10_061.41       879.60    10_941.01       0.5567          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                        10_061.41     1_035.15    11_096.56       0.5569          NaN         3.89
IVF-OPQ-nl547-m16 (self)                              10_061.41    10_722.70    20_784.11       0.4588          NaN         3.89
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
