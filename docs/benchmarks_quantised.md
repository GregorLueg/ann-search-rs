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
structure is lost during the lossy compression (at least in lower dimensions).

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
Exhaustive (query)                                        14.65     6_201.30     6_215.95       1.0000     0.000000        73.24
Exhaustive (self)                                         14.65    64_082.46    64_097.10       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    36.46     1_705.85     1_742.30       0.8081          NaN        18.31
Exhaustive-SQ8 (self)                                     36.46    17_569.62    17_606.08       0.8095          NaN        18.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                             1_397.87       246.23     1_644.10       0.8081          NaN        19.59
IVF-SQ8-nl273-np16 (query)                             1_397.87       290.10     1_687.97       0.8081          NaN        19.59
IVF-SQ8-nl273-np23 (query)                             1_397.87       384.09     1_781.96       0.8082          NaN        19.59
IVF-SQ8-nl273 (self)                                   1_397.87     3_800.12     5_198.00       0.8096          NaN        19.59
IVF-SQ8-nl387-np19 (query)                             2_148.38       256.83     2_405.21       0.8081          NaN        19.65
IVF-SQ8-nl387-np27 (query)                             2_148.38       343.17     2_491.55       0.8081          NaN        19.65
IVF-SQ8-nl387 (self)                                   2_148.38     3_340.96     5_489.34       0.8096          NaN        19.65
IVF-SQ8-nl547-np23 (query)                             4_087.39       232.90     4_320.29       0.8081          NaN        19.73
IVF-SQ8-nl547-np27 (query)                             4_087.39       262.39     4_349.78       0.8081          NaN        19.73
IVF-SQ8-nl547-np33 (query)                             4_087.39       314.81     4_402.20       0.8081          NaN        19.73
IVF-SQ8-nl547 (self)                                   4_087.39     3_039.26     7_126.65       0.8095          NaN        19.73
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
Exhaustive (query)                                        29.08    15_630.26    15_659.34       1.0000     0.000000       146.48
Exhaustive (self)                                         29.08   156_297.81   156_326.89       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_756.75     1_958.72     3_715.47       0.0968          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_756.75    19_463.01    21_219.77       0.0817          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_104.79     4_456.01     6_560.80       0.1321          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_104.79    44_868.04    46_972.82       0.0941          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_667.00    12_105.21    15_772.21       0.2669          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_667.00   121_181.29   124_848.30       0.1882          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          5_317.49       609.93     5_927.42       0.1527          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          5_317.49       739.58     6_057.08       0.1531          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          5_317.49     1_032.13     6_349.62       0.1533          NaN         3.95
IVF-PQ-nl273-m16 (self)                                5_317.49    10_534.99    15_852.48       0.1039          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          5_715.96       931.88     6_647.85       0.2665          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          5_715.96     1_145.86     6_861.83       0.2686          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          5_715.96     1_643.73     7_359.69       0.2702          NaN         6.24
IVF-PQ-nl273-m32 (self)                                5_715.96    16_472.99    22_188.95       0.1840          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          7_244.12     1_788.92     9_033.04       0.5111          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          7_244.12     2_187.79     9_431.91       0.5178          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          7_244.12     3_079.57    10_323.69       0.5232          NaN        10.82
IVF-PQ-nl273-m64 (self)                                7_244.12    30_901.03    38_145.15       0.4364          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          8_560.34       799.34     9_359.68       0.1540          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          8_560.34     1_115.92     9_676.26       0.1543          NaN         4.06
IVF-PQ-nl387-m16 (self)                                8_560.34    10_884.99    19_445.33       0.1043          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          8_719.43     1_269.78     9_989.21       0.2698          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          8_719.43     1_768.48    10_487.90       0.2719          NaN         6.35
IVF-PQ-nl387-m32 (self)                                8_719.43    17_406.31    26_125.74       0.1851          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                          9_881.96     2_326.47    12_208.43       0.5147          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          9_881.96     3_259.18    13_141.14       0.5223          NaN        10.93
IVF-PQ-nl387-m64 (self)                                9_881.96    32_111.01    41_992.98       0.4366          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                         12_912.98       903.32    13_816.30       0.1540          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                         12_912.98     1_049.29    13_962.27       0.1545          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                         12_912.98     1_252.17    14_165.15       0.1547          NaN         4.22
IVF-PQ-nl547-m16 (self)                               12_912.98    12_630.83    25_543.81       0.1053          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                         13_513.33     1_424.02    14_937.35       0.2693          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                         13_513.33     1_631.99    15_145.32       0.2714          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                         13_513.33     2_098.66    15_611.99       0.2723          NaN         6.51
IVF-PQ-nl547-m32 (self)                               13_513.33    20_448.30    33_961.63       0.1854          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         15_040.38     2_657.45    17_697.83       0.5134          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         15_040.38     2_989.49    18_029.87       0.5202          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         15_040.38     3_647.65    18_688.03       0.5232          NaN        11.09
IVF-PQ-nl547-m64 (self)                               15_040.38    36_190.89    51_231.26       0.4370          NaN        11.09
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
Exhaustive (query)                                        29.20    15_594.17    15_623.37       1.0000     0.000000       146.48
Exhaustive (self)                                         29.20   156_918.13   156_947.33       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              2_134.25     1_945.54     4_079.79       0.2357          NaN         2.54
Exhaustive-PQ-m16 (self)                               2_134.25    19_524.49    21_658.74       0.1675          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_083.47     4_461.74     6_545.21       0.3554          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_083.47    44_656.90    46_740.36       0.2679          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_611.07    12_095.54    15_706.62       0.4755          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_611.07   121_031.57   124_642.64       0.3907          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          4_543.00       631.93     5_174.94       0.3867          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          4_543.00       758.61     5_301.61       0.3875          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          4_543.00     1_074.89     5_617.90       0.3879          NaN         3.95
IVF-PQ-nl273-m16 (self)                                4_543.00    10_667.78    15_210.78       0.2883          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          4_936.93       969.18     5_906.11       0.5552          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          4_936.93     1_197.95     6_134.88       0.5569          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          4_936.93     1_713.26     6_650.19       0.5577          NaN         6.24
IVF-PQ-nl273-m32 (self)                                4_936.93    17_196.39    22_133.32       0.4669          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          6_374.21     1_785.13     8_159.34       0.7091          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          6_374.21     2_178.56     8_552.77       0.7122          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          6_374.21     3_089.06     9_463.26       0.7137          NaN        10.82
IVF-PQ-nl273-m64 (self)                                6_374.21    30_813.37    37_187.58       0.6506          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          6_111.10       822.36     6_933.46       0.3881          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          6_111.10     1_146.81     7_257.91       0.3887          NaN         4.06
IVF-PQ-nl387-m16 (self)                                6_111.10    11_283.30    17_394.40       0.2912          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          6_722.88     1_307.91     8_030.79       0.5584          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          6_722.88     1_799.96     8_522.84       0.5601          NaN         6.35
IVF-PQ-nl387-m32 (self)                                6_722.88    17_643.47    24_366.35       0.4688          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                          8_174.35     2_309.58    10_483.92       0.7134          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          8_174.35     3_218.37    11_392.71       0.7159          NaN        10.93
IVF-PQ-nl387-m64 (self)                                8_174.35    32_011.03    40_185.38       0.6513          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          9_629.54       914.76    10_544.29       0.3892          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          9_629.54     1_085.46    10_715.00       0.3897          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          9_629.54     1_326.29    10_955.83       0.3900          NaN         4.22
IVF-PQ-nl547-m16 (self)                                9_629.54    12_995.58    22_625.12       0.2924          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                         10_209.06     1_421.28    11_630.35       0.5590          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                         10_209.06     1_635.50    11_844.56       0.5607          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                         10_209.06     2_006.71    12_215.77       0.5617          NaN         6.51
IVF-PQ-nl547-m32 (self)                               10_209.06    20_018.02    30_227.09       0.4705          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         11_609.79     2_523.41    14_133.20       0.7118          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         11_609.79     2_955.20    14_564.99       0.7145          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         11_609.79     3_611.20    15_220.99       0.7161          NaN        11.09
IVF-PQ-nl547-m64 (self)                               11_609.79    35_760.11    47_369.90       0.6529          NaN        11.09
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
Exhaustive (query)                                        29.47    15_541.91    15_571.38       1.0000     0.000000       146.48
Exhaustive (self)                                         29.47   156_403.86   156_433.33       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_940.66     1_950.97     3_891.63       0.3801          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_940.66    19_540.05    21_480.71       0.2823          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_117.83     4_444.03     6_561.87       0.5114          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_117.83    44_497.21    46_615.04       0.4053          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_543.63    12_092.40    15_636.02       0.6355          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_543.63   120_970.37   124_514.00       0.5576          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          4_007.56       608.90     4_616.46       0.6312          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          4_007.56       747.29     4_754.85       0.6312          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          4_007.56     1_059.28     5_066.83       0.6312          NaN         3.95
IVF-PQ-nl273-m16 (self)                                4_007.56    10_601.31    14_608.87       0.4847          NaN         3.95
IVF-PQ-nl273-m32-np13 (query)                          4_484.70       953.34     5_438.04       0.7578          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          4_484.70     1_181.83     5_666.53       0.7578          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          4_484.70     1_665.36     6_150.06       0.7578          NaN         6.24
IVF-PQ-nl273-m32 (self)                                4_484.70    16_597.99    21_082.69       0.6650          NaN         6.24
IVF-PQ-nl273-m64-np13 (query)                          6_081.17     1_769.92     7_851.09       0.8807          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          6_081.17     2_176.35     8_257.52       0.8808          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          6_081.17     3_075.21     9_156.38       0.8808          NaN        10.82
IVF-PQ-nl273-m64 (self)                                6_081.17    30_459.41    36_540.58       0.8376          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          5_292.49       809.43     6_101.93       0.6305          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          5_292.49     1_117.35     6_409.84       0.6305          NaN         4.06
IVF-PQ-nl387-m16 (self)                                5_292.49    11_040.73    16_333.22       0.4753          NaN         4.06
IVF-PQ-nl387-m32-np19 (query)                          5_777.39     1_282.60     7_059.99       0.7585          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          5_777.39     1_764.88     7_542.27       0.7585          NaN         6.35
IVF-PQ-nl387-m32 (self)                                5_777.39    17_799.42    23_576.82       0.6607          NaN         6.35
IVF-PQ-nl387-m64-np19 (query)                          7_042.77     2_286.59     9_329.36       0.8824          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          7_042.77     3_254.34    10_297.12       0.8824          NaN        10.93
IVF-PQ-nl387-m64 (self)                                7_042.77    32_099.41    39_142.18       0.8385          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          8_404.79       902.98     9_307.77       0.6328          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          8_404.79     1_071.03     9_475.82       0.6328          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          8_404.79     1_292.38     9_697.17       0.6328          NaN         4.22
IVF-PQ-nl547-m16 (self)                                8_404.79    12_726.80    21_131.59       0.4665          NaN         4.22
IVF-PQ-nl547-m32-np23 (query)                          9_093.25     1_433.01    10_526.26       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                          9_093.25     1_642.91    10_736.17       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                          9_093.25     1_981.96    11_075.21       0.7595          NaN         6.51
IVF-PQ-nl547-m32 (self)                                9_093.25    20_165.23    29_258.48       0.6562          NaN         6.51
IVF-PQ-nl547-m64-np23 (query)                         10_711.61     2_542.58    13_254.18       0.8834          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                         10_711.61     3_002.87    13_714.48       0.8835          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                         10_711.61     3_637.87    14_349.47       0.8835          NaN        11.09
IVF-PQ-nl547-m64 (self)                               10_711.61    35_969.51    46_681.12       0.8392          NaN        11.09
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
Exhaustive (query)                                        14.53     6_324.33     6_338.86       1.0000     0.000000        73.24
Exhaustive (self)                                         14.53    65_166.04    65_180.57       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              6_855.30       896.57     7_751.87       0.0961          NaN         1.33
Exhaustive-OPQ-m8 (self)                               6_855.30     9_225.38    16_080.68       0.0833          NaN         1.33
Exhaustive-OPQ-m16 (query)                             4_396.32     2_075.30     6_471.62       0.1326          NaN         2.48
Exhaustive-OPQ-m16 (self)                              4_396.32    20_371.31    24_767.64       0.0961          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                         13_218.30       315.62    13_533.92       0.1527          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                         13_218.30       371.96    13_590.25       0.1530          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                         13_218.30       521.76    13_740.06       0.1531          NaN         2.61
IVF-OPQ-nl273-m8 (self)                               13_218.30     5_470.90    18_689.20       0.1043          NaN         2.61
IVF-OPQ-nl273-m16-np13 (query)                         7_268.70       513.97     7_782.67       0.2672          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         7_268.70       635.51     7_904.21       0.2692          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         7_268.70       892.26     8_160.96       0.2700          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               7_268.70     9_035.75    16_304.45       0.1800          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          8_314.77       424.84     8_739.61       0.1529          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          8_314.77       608.53     8_923.29       0.1532          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                8_314.77     6_177.35    14_492.11       0.1046          NaN         2.67
IVF-OPQ-nl387-m16-np19 (query)                         8_426.59       686.61     9_113.20       0.2695          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         8_426.59       962.98     9_389.57       0.2709          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               8_426.59     9_757.14    18_183.73       0.1812          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                         11_588.20       522.46    12_110.67       0.1574          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                         11_588.20       576.97    12_165.17       0.1576          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                         11_588.20       724.45    12_312.66       0.1576          NaN         2.75
IVF-OPQ-nl547-m8 (self)                               11_588.20     7_291.04    18_879.24       0.1064          NaN         2.75
IVF-OPQ-nl547-m16-np23 (query)                        11_961.02       766.25    12_727.27       0.2690          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                        11_961.02       909.98    12_871.00       0.2705          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                        11_961.02     1_077.93    13_038.95       0.2715          NaN         3.89
IVF-OPQ-nl547-m16 (self)                              11_961.02    11_125.44    23_086.46       0.1824          NaN         3.89
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
Exhaustive (query)                                        15.29     6_539.31     6_554.59       1.0000     0.000000        73.24
Exhaustive (self)                                         15.29    65_147.16    65_162.45       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              3_212.56       894.77     4_107.33       0.2219          NaN         1.33
Exhaustive-OPQ-m8 (self)                               3_212.56     9_167.35    12_379.92       0.1616          NaN         1.33
Exhaustive-OPQ-m16 (query)                             3_691.13     2_048.68     5_739.82       0.3515          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_691.13    19_633.35    23_324.48       0.2620          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          5_475.47       312.96     5_788.42       0.3774          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          5_475.47       379.02     5_854.49       0.3776          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          5_475.47       552.10     6_027.57       0.3776          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                5_475.47     5_525.41    11_000.88       0.2761          NaN         2.61
IVF-OPQ-nl273-m16-np13 (query)                         5_864.50       500.55     6_365.05       0.5495          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         5_864.50       610.74     6_475.25       0.5502          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         5_864.50       881.62     6_746.12       0.5504          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               5_864.50     8_827.85    14_692.35       0.4538          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          6_844.44       407.63     7_252.07       0.3807          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          6_844.44       585.08     7_429.52       0.3807          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                6_844.44     5_930.74    12_775.17       0.2798          NaN         2.67
IVF-OPQ-nl387-m16-np19 (query)                         6_915.40       654.57     7_569.97       0.5528          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         6_915.40       897.69     7_813.08       0.5533          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               6_915.40     9_313.51    16_228.91       0.4568          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          9_521.08       484.91    10_005.99       0.3849          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          9_521.08       547.12    10_068.20       0.3849          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          9_521.08       677.16    10_198.24       0.3848          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                9_521.08     7_111.39    16_632.47       0.2832          NaN         2.75
IVF-OPQ-nl547-m16-np23 (query)                         9_835.14       774.19    10_609.33       0.5562          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         9_835.14       872.14    10_707.28       0.5567          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         9_835.14     1_081.67    10_916.81       0.5569          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               9_835.14    10_743.69    20_578.83       0.4588          NaN         3.89
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
Exhaustive (query)                                        14.35     6_460.96     6_475.31       1.0000     0.000000        73.24
Exhaustive (self)                                         14.35    65_057.58    65_071.93       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              4_155.75       887.42     5_043.17       0.2589          NaN         1.33
Exhaustive-OPQ-m8 (self)                               4_155.75     9_157.98    13_313.72       0.1606          NaN         1.33
Exhaustive-OPQ-m16 (query)                             3_754.57     1_932.50     5_687.07       0.3956          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_754.57    19_630.89    23_385.46       0.2465          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          4_909.98       316.18     5_226.16       0.6180          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          4_909.98       389.74     5_299.72       0.6180          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          4_909.98       538.58     5_448.56       0.6180          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                4_909.98     5_557.83    10_467.81       0.5162          NaN         2.61
IVF-OPQ-nl273-m16-np13 (query)                         5_265.33       496.64     5_761.96       0.7461          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         5_265.33       601.59     5_866.91       0.7462          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         5_265.33       867.89     6_133.22       0.7462          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               5_265.33     8_761.34    14_026.67       0.6672          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          5_838.77       406.44     6_245.21       0.6229          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          5_838.77       593.47     6_432.24       0.6229          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                5_838.77     5_886.12    11_724.89       0.5230          NaN         2.67
IVF-OPQ-nl387-m16-np19 (query)                         6_235.36       690.88     6_926.25       0.7476          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         6_235.36       925.75     7_161.12       0.7476          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               6_235.36     9_571.52    15_806.88       0.6708          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          8_605.94       468.74     9_074.68       0.6283          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          8_605.94       544.42     9_150.36       0.6283          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          8_605.94       692.96     9_298.91       0.6283          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                8_605.94     6_901.72    15_507.67       0.5297          NaN         2.75
IVF-OPQ-nl547-m16-np23 (query)                         9_144.45       780.21     9_924.65       0.7490          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         9_144.45       895.79    10_040.24       0.7490          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         9_144.45     1_078.25    10_222.70       0.7490          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               9_144.45    11_180.11    20_324.56       0.6722          NaN         3.89
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 256 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is not too great, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 256 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k cells, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.72    15_749.64    15_779.36       1.0000     0.000000       146.48
Exhaustive (self)                                         29.72   157_156.09   157_185.81       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             8_148.71     1_951.50    10_100.20       0.0952          NaN         2.79
Exhaustive-OPQ-m16 (self)                              8_148.71    20_482.07    28_630.78       0.0815          NaN         2.79
Exhaustive-OPQ-m32 (query)                             8_039.27     4_455.30    12_494.58       0.1298          NaN         5.08
Exhaustive-OPQ-m32 (self)                              8_039.27    45_794.40    53_833.67       0.0938          NaN         5.08
Exhaustive-OPQ-m64 (query)                            12_687.79    12_357.94    25_045.73       0.2633          NaN         9.66
Exhaustive-OPQ-m64 (self)                             12_687.79   124_217.55   136_905.34       0.1888          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                        11_233.24       582.22    11_815.46       0.1516          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                        11_233.24       712.21    11_945.45       0.1518          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                        11_233.24     1_028.80    12_262.04       0.1521          NaN         4.20
IVF-OPQ-nl273-m16 (self)                              11_233.24    11_361.20    22_594.45       0.1033          NaN         4.20
IVF-OPQ-nl273-m32-np13 (query)                        11_824.81       974.10    12_798.91       0.2644          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                        11_824.81     1_169.68    12_994.48       0.2664          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                        11_824.81     1_664.93    13_489.74       0.2680          NaN         6.49
IVF-OPQ-nl273-m32 (self)                              11_824.81    17_739.74    29_564.54       0.1835          NaN         6.49
IVF-OPQ-nl273-m64-np13 (query)                        16_255.25     1_806.23    18_061.48       0.5109          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        16_255.25     2_154.51    18_409.76       0.5176          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        16_255.25     3_061.07    19_316.32       0.5231          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              16_255.25    31_918.69    48_173.94       0.4363          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                        13_551.06       798.88    14_349.93       0.1520          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                        13_551.06     1_080.17    14_631.22       0.1522          NaN         4.31
IVF-OPQ-nl387-m16 (self)                              13_551.06    11_815.36    25_366.42       0.1042          NaN         4.31
IVF-OPQ-nl387-m32-np19 (query)                        14_261.91     1_271.58    15_533.49       0.2663          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                        14_261.91     1_742.16    16_004.07       0.2679          NaN         6.60
IVF-OPQ-nl387-m32 (self)                              14_261.91    18_440.36    32_702.27       0.1832          NaN         6.60
IVF-OPQ-nl387-m64-np19 (query)                        18_467.55     2_280.86    20_748.41       0.5143          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        18_467.55     3_225.96    21_693.51       0.5219          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              18_467.55    33_384.00    51_851.55       0.4357          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                        17_648.77       896.19    18_544.95       0.1531          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                        17_648.77     1_019.39    18_668.16       0.1534          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                        17_648.77     1_236.05    18_884.81       0.1536          NaN         4.47
IVF-OPQ-nl547-m16 (self)                              17_648.77    13_687.25    31_336.02       0.1046          NaN         4.47
IVF-OPQ-nl547-m32-np23 (query)                        18_524.47     1_413.31    19_937.78       0.2666          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        18_524.47     1_638.88    20_163.35       0.2687          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        18_524.47     2_008.45    20_532.92       0.2698          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              18_524.47    21_085.10    39_609.57       0.1848          NaN         6.76
IVF-OPQ-nl547-m64-np23 (query)                        22_845.94     2_552.79    25_398.73       0.5124          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        22_845.94     2_944.80    25_790.75       0.5195          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        22_845.94     3_885.74    26_731.68       0.5227          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              22_845.94    37_317.91    60_163.85       0.4371          NaN        11.34
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
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
Exhaustive (query)                                        30.32    15_637.46    15_667.79       1.0000     0.000000       146.48
Exhaustive (self)                                         30.32   157_041.57   157_071.89       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             7_893.27     1_960.92     9_854.20       0.2358          NaN         2.79
Exhaustive-OPQ-m16 (self)                              7_893.27    20_504.19    28_397.46       0.1667          NaN         2.79
Exhaustive-OPQ-m32 (query)                             8_017.02     4_448.73    12_465.75       0.3541          NaN         5.08
Exhaustive-OPQ-m32 (self)                              8_017.02    48_821.68    56_838.70       0.2676          NaN         5.08
Exhaustive-OPQ-m64 (query)                            12_341.96    12_302.45    24_644.41       0.4827          NaN         9.66
Exhaustive-OPQ-m64 (self)                             12_341.96   123_915.96   136_257.91       0.4017          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                        10_302.42       605.37    10_907.79       0.3874          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                        10_302.42       729.17    11_031.59       0.3881          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                        10_302.42     1_025.79    11_328.21       0.3884          NaN         4.20
IVF-OPQ-nl273-m16 (self)                              10_302.42    11_506.64    21_809.06       0.2890          NaN         4.20
IVF-OPQ-nl273-m32-np13 (query)                        11_260.03       951.36    12_211.39       0.5554          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                        11_260.03     1_198.09    12_458.12       0.5572          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                        11_260.03     1_676.19    12_936.22       0.5581          NaN         6.49
IVF-OPQ-nl273-m32 (self)                              11_260.03    17_902.85    29_162.88       0.4671          NaN         6.49
IVF-OPQ-nl273-m64-np13 (query)                        15_295.54     1_747.99    17_043.53       0.7124          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        15_295.54     2_183.73    17_479.27       0.7154          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        15_295.54     3_096.12    18_391.66       0.7170          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              15_295.54    31_958.36    47_253.91       0.6559          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                        11_924.72       764.60    12_689.32       0.3874          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                        11_924.72     1_061.41    12_986.13       0.3881          NaN         4.31
IVF-OPQ-nl387-m16 (self)                              11_924.72    11_736.34    23_661.05       0.2908          NaN         4.31
IVF-OPQ-nl387-m32-np19 (query)                        12_606.58     1_254.80    13_861.38       0.5572          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                        12_606.58     1_764.83    14_371.41       0.5591          NaN         6.60
IVF-OPQ-nl387-m32 (self)                              12_606.58    18_500.82    31_107.40       0.4688          NaN         6.60
IVF-OPQ-nl387-m64-np19 (query)                        16_788.60     2_268.44    19_057.04       0.7153          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        16_788.60     3_187.84    19_976.44       0.7181          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              16_788.60    32_966.02    49_754.62       0.6565          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                        15_100.00       910.74    16_010.73       0.3894          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                        15_100.00     1_044.12    16_144.12       0.3899          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                        15_100.00     1_244.24    16_344.24       0.3901          NaN         4.47
IVF-OPQ-nl547-m16 (self)                              15_100.00    13_519.16    28_619.16       0.2922          NaN         4.47
IVF-OPQ-nl547-m32-np23 (query)                        15_755.26     1_446.73    17_201.99       0.5569          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        15_755.26     1_670.65    17_425.90       0.5585          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        15_755.26     2_021.61    17_776.87       0.5594          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              15_755.26    21_758.22    37_513.48       0.4701          NaN         6.76
IVF-OPQ-nl547-m64-np23 (query)                        20_151.14     2_507.46    22_658.59       0.7141          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        20_151.14     2_930.97    23_082.11       0.7167          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        20_151.14     3_578.09    23_729.23       0.7184          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              20_151.14    37_213.86    57_364.99       0.6578          NaN        11.34
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
Exhaustive (query)                                        29.19    15_457.01    15_486.21       1.0000     0.000000       146.48
Exhaustive (self)                                         29.19   156_633.06   156_662.25       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             7_683.60     2_040.30     9_723.90       0.3798          NaN         2.79
Exhaustive-OPQ-m16 (self)                              7_683.60    22_074.02    29_757.62       0.2244          NaN         2.79
Exhaustive-OPQ-m32 (query)                             8_093.27     4_458.65    12_551.92       0.5110          NaN         5.08
Exhaustive-OPQ-m32 (self)                              8_093.27    45_609.92    53_703.19       0.3522          NaN         5.08
Exhaustive-OPQ-m64 (query)                            12_570.39    12_260.97    24_831.36       0.6356          NaN         9.66
Exhaustive-OPQ-m64 (self)                             12_570.39   124_125.87   136_696.25       0.5341          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                        10_692.60       588.49    11_281.10       0.6899          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                        10_692.60       761.33    11_453.93       0.6899          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                        10_692.60     1_006.18    11_698.78       0.6899          NaN         4.20
IVF-OPQ-nl273-m16 (self)                              10_692.60    11_411.94    22_104.55       0.6034          NaN         4.20
IVF-OPQ-nl273-m32-np13 (query)                        10_485.70       964.78    11_450.48       0.7975          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                        10_485.70     1_141.02    11_626.72       0.7976          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                        10_485.70     1_612.69    12_098.40       0.7976          NaN         6.49
IVF-OPQ-nl273-m32 (self)                              10_485.70    17_385.71    27_871.42       0.7278          NaN         6.49
IVF-OPQ-nl273-m64-np13 (query)                        14_750.02     1_773.59    16_523.60       0.8954          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        14_750.02     2_185.84    16_935.86       0.8955          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        14_750.02     3_056.83    17_806.85       0.8955          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              14_750.02    31_720.11    46_470.13       0.8517          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                        10_963.59       833.11    11_796.71       0.6905          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                        10_963.59     1_127.51    12_091.10       0.6905          NaN         4.31
IVF-OPQ-nl387-m16 (self)                              10_963.59    12_281.43    23_245.02       0.6049          NaN         4.31
IVF-OPQ-nl387-m32-np19 (query)                        11_791.05     1_238.67    13_029.71       0.7969          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                        11_791.05     1_732.13    13_523.18       0.7969          NaN         6.60
IVF-OPQ-nl387-m32 (self)                              11_791.05    18_141.03    29_932.08       0.7284          NaN         6.60
IVF-OPQ-nl387-m64-np19 (query)                        15_978.00     2_300.40    18_278.40       0.8959          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        15_978.00     3_278.36    19_256.35       0.8959          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              15_978.00    33_400.44    49_378.43       0.8522          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                        15_666.01       900.62    16_566.62       0.6905          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                        15_666.01     1_094.04    16_760.04       0.6905          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                        15_666.01     1_260.26    16_926.27       0.6905          NaN         4.47
IVF-OPQ-nl547-m16 (self)                              15_666.01    13_432.06    29_098.07       0.6075          NaN         4.47
IVF-OPQ-nl547-m32-np23 (query)                        15_630.95     1_417.92    17_048.87       0.7952          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        15_630.95     1_659.98    17_290.93       0.7952          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        15_630.95     2_036.69    17_667.64       0.7952          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              15_630.95    21_614.32    37_245.27       0.7286          NaN         6.76
IVF-OPQ-nl547-m64-np23 (query)                        19_659.63     2_556.57    22_216.19       0.8954          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        19_659.63     2_949.75    22_609.38       0.8954          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        19_659.63     3_592.54    23_252.17       0.8954          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              19_659.63    37_129.39    56_789.02       0.8534          NaN        11.34
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
