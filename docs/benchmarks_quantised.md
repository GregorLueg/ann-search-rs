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
Exhaustive (query)                                         3.06     1_514.49     1_517.55       1.0000     0.000000        18.31
Exhaustive (self)                                          3.06    15_150.30    15_153.36       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.22     1_164.10     1_169.33       0.9867     0.065388         9.16
Exhaustive-BF16 (self)                                     5.22    15_494.97    15_500.20       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              389.44       122.81       512.25       0.9758     0.137513        10.34
IVF-BF16-nl273-np16 (query)                              389.44       146.86       536.30       0.9845     0.088329        10.34
IVF-BF16-nl273-np23 (query)                              389.44       212.86       602.30       0.9867     0.065388        10.34
IVF-BF16-nl273 (self)                                    389.44     2_083.92     2_473.35       0.9830     0.094689        10.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              746.54       129.53       876.06       0.9802     0.091487        10.35
IVF-BF16-nl387-np27 (query)                              746.54       172.37       918.91       0.9865     0.065640        10.35
IVF-BF16-nl387 (self)                                    746.54     1_735.95     2_482.48       0.9828     0.094875        10.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_433.66       113.19     1_546.84       0.9773     0.105124        10.37
IVF-BF16-nl547-np27 (query)                            1_433.66       128.16     1_561.82       0.9842     0.077306        10.37
IVF-BF16-nl547-np33 (query)                            1_433.66       152.48     1_586.14       0.9866     0.065956        10.37
IVF-BF16-nl547 (self)                                  1_433.66     1_605.42     3_039.08       0.9828     0.095761        10.37
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
Exhaustive (query)                                         4.05     1_491.48     1_495.53       1.0000     0.000000        18.88
Exhaustive (self)                                          4.05    15_102.46    15_106.52       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.97     1_217.08     1_223.06       0.9240     0.000230         9.44
Exhaustive-BF16 (self)                                     5.97    15_006.58    15_012.55       1.0000     0.000000         9.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              371.78       137.82       509.60       0.9163     0.000265        10.62
IVF-BF16-nl273-np16 (query)                              371.78       166.14       537.93       0.9222     0.000241        10.62
IVF-BF16-nl273-np23 (query)                              371.78       238.43       610.21       0.9240     0.000230        10.62
IVF-BF16-nl273 (self)                                    371.78     2_411.33     2_783.11       0.9229     0.001251        10.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              699.96       149.59       849.56       0.9200     0.000243        10.64
IVF-BF16-nl387-np27 (query)                              699.96       214.58       914.55       0.9239     0.000230        10.64
IVF-BF16-nl387 (self)                                    699.96     2_089.33     2_789.29       0.9228     0.001251        10.64
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_362.43       135.04     1_497.47       0.9183     0.000248        10.66
IVF-BF16-nl547-np27 (query)                            1_362.43       159.78     1_522.21       0.9225     0.000235        10.66
IVF-BF16-nl547-np33 (query)                            1_362.43       188.13     1_550.56       0.9239     0.000231        10.66
IVF-BF16-nl547 (self)                                  1_362.43     1_873.53     3_235.95       0.9228     0.001251        10.66
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
Exhaustive (query)                                         3.15     1_531.96     1_535.10       1.0000     0.000000        18.31
Exhaustive (self)                                          3.15    16_352.14    16_355.29       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.44     1_171.94     1_177.38       0.9649     0.019029         9.16
Exhaustive-BF16 (self)                                     5.44    15_371.24    15_376.68       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              400.25       128.78       529.03       0.9649     0.019034        10.34
IVF-BF16-nl273-np16 (query)                              400.25       156.64       556.90       0.9649     0.019029        10.34
IVF-BF16-nl273-np23 (query)                              400.25       207.98       608.23       0.9649     0.019029        10.34
IVF-BF16-nl273 (self)                                    400.25     2_100.39     2_500.65       0.9561     0.026167        10.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              742.74       149.28       892.02       0.9649     0.019040        10.35
IVF-BF16-nl387-np27 (query)                              742.74       182.34       925.08       0.9649     0.019029        10.35
IVF-BF16-nl387 (self)                                    742.74     1_830.42     2_573.16       0.9561     0.026167        10.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_444.66       124.23     1_568.89       0.9649     0.019032        10.37
IVF-BF16-nl547-np27 (query)                            1_444.66       140.37     1_585.04       0.9649     0.019031        10.37
IVF-BF16-nl547-np33 (query)                            1_444.66       165.51     1_610.18       0.9649     0.019029        10.37
IVF-BF16-nl547 (self)                                  1_444.66     1_685.07     3_129.74       0.9561     0.026167        10.37
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
Exhaustive (query)                                         3.14     1_615.77     1_618.92       1.0000     0.000000        18.31
Exhaustive (self)                                          3.14    15_326.09    15_329.24       1.0000     0.000000        18.
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    4.82     1_163.53     1_168.35       0.9348     0.158338         9.16
Exhaustive-BF16 (self)                                     4.82    15_372.66    15_377.48       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              421.08       130.12       551.20       0.9348     0.158338        10.34
IVF-BF16-nl273-np16 (query)                              421.08       156.29       577.37       0.9348     0.158338        10.34
IVF-BF16-nl273-np23 (query)                              421.08       216.28       637.36       0.9348     0.158338        10.34
IVF-BF16-nl273 (self)                                    421.08     2_211.75     2_632.83       0.9174     0.221294        10.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              752.40       132.20       884.60       0.9348     0.158338        10.35
IVF-BF16-nl387-np27 (query)                              752.40       180.91       933.31       0.9348     0.158338        10.35
IVF-BF16-nl387 (self)                                    752.40     1_806.52     2_558.92       0.9174     0.221294        10.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_452.33       120.66     1_572.98       0.9348     0.158338        10.37
IVF-BF16-nl547-np27 (query)                            1_452.33       137.62     1_589.95       0.9348     0.158338        10.37
IVF-BF16-nl547-np33 (query)                            1_452.33       164.23     1_616.56       0.9348     0.158338        10.37
IVF-BF16-nl547 (self)                                  1_452.33     1_661.90     3_114.23       0.9174     0.221294        10.37
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
Exhaustive (query)                                        14.59     6_118.45     6_133.04       1.0000     0.000000        73.24
Exhaustive (self)                                         14.59    60_807.27    60_821.86       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                   24.06     5_099.24     5_123.30       0.9714     0.573744        36.62
Exhaustive-BF16 (self)                                    24.06    62_855.65    62_879.70       1.0000     0.000000        36.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              430.40       711.38     1_141.77       0.9712     0.574446        37.90
IVF-BF16-nl273-np16 (query)                              430.40       868.07     1_298.47       0.9713     0.573771        37.90
IVF-BF16-nl273-np23 (query)                              430.40     1_236.70     1_667.10       0.9714     0.573744        37.90
IVF-BF16-nl273 (self)                                    430.40    12_645.30    13_075.70       0.9637     0.816885        37.90
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              760.18       740.35     1_500.53       0.9713     0.573817        37.96
IVF-BF16-nl387-np27 (query)                              760.18     1_006.86     1_767.04       0.9714     0.573744        37.96
IVF-BF16-nl387 (self)                                    760.18    10_241.48    11_001.66       0.9637     0.816885        37.96
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_501.54       636.96     2_138.51       0.9713     0.573810        38.04
IVF-BF16-nl547-np27 (query)                            1_501.54       737.38     2_238.93       0.9714     0.573744        38.04
IVF-BF16-nl547-np33 (query)                            1_501.54       907.61     2_409.15       0.9714     0.573744        38.04
IVF-BF16-nl547 (self)                                  1_501.54     9_162.56    10_664.11       0.9637     0.816885        38.04
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
Exhaustive (query)                                         3.31     1_532.72     1_536.03       1.0000     0.000000        18.31
Exhaustive (self)                                          3.31    15_821.13    15_824.44       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.92       729.57       737.49       0.8011          NaN         4.58
Exhaustive-SQ8 (self)                                      7.92     7_653.98     7_661.90       0.8007          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               412.39        54.75       467.14       0.7779          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               412.39        62.74       475.13       0.7813          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               412.39        88.47       500.86       0.7822          NaN         5.76
IVF-SQ8-nl273 (self)                                     412.39       855.66     1_268.06       0.7819          NaN         5.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               756.17        56.44       812.61       0.7854          NaN         5.77
IVF-SQ8-nl387-np27 (query)                               756.17        72.67       828.84       0.7878          NaN         5.77
IVF-SQ8-nl387 (self)                                     756.17       714.58     1_470.74       0.7873          NaN         5.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_463.99        57.56     1_521.55       0.7972          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_463.99        61.26     1_525.25       0.8001          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_463.99        70.76     1_534.75       0.8011          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_463.99       691.09     2_155.08       0.8007          NaN         5.79
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
Exhaustive (query)                                         4.18     1_527.85     1_532.02       1.0000     0.000000        18.88
Exhaustive (self)                                          4.18    15_944.83    15_949.01       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.16       704.48       711.64       0.8501          NaN         5.15
Exhaustive-SQ8 (self)                                      7.16     7_284.33     7_291.49       0.8497          NaN         5.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               400.69        57.38       458.07       0.8423          NaN         6.33
IVF-SQ8-nl273-np16 (query)                               400.69        68.07       468.76       0.8463          NaN         6.33
IVF-SQ8-nl273-np23 (query)                               400.69        93.38       494.07       0.8473          NaN         6.33
IVF-SQ8-nl273 (self)                                     400.69       916.05     1_316.74       0.8467          NaN         6.33
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               707.90        62.96       770.86       0.8420          NaN         6.34
IVF-SQ8-nl387-np27 (query)                               707.90        84.17       792.07       0.8449          NaN         6.34
IVF-SQ8-nl387 (self)                                     707.90       803.92     1_511.82       0.8446          NaN         6.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_371.65        59.96     1_431.61       0.8437          NaN         6.37
IVF-SQ8-nl547-np27 (query)                             1_371.65        63.72     1_435.37       0.8467          NaN         6.37
IVF-SQ8-nl547-np33 (query)                             1_371.65        74.01     1_445.66       0.8477          NaN         6.37
IVF-SQ8-nl547 (self)                                   1_371.65       705.57     2_077.22       0.8473          NaN         6.37
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
Exhaustive (query)                                         3.17     1_542.32     1_545.49       1.0000     0.000000        18.31
Exhaustive (self)                                          3.17    15_525.52    15_528.69       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.44       717.08       724.52       0.6828          NaN         4.58
Exhaustive-SQ8 (self)                                      7.44     7_582.56     7_590.01       0.6835          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               388.58        54.75       443.33       0.6821          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               388.58        63.44       452.02       0.6820          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               388.58        85.83       474.41       0.6820          NaN         5.76
IVF-SQ8-nl273 (self)                                     388.58       817.65     1_206.23       0.6833          NaN         5.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               731.49        57.97       789.46       0.6843          NaN         5.77
IVF-SQ8-nl387-np27 (query)                               731.49        82.96       814.45       0.6843          NaN         5.77
IVF-SQ8-nl387 (self)                                     731.49       734.60     1_466.09       0.6851          NaN         5.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_421.67        53.51     1_475.17       0.6828          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_421.67        59.98     1_481.65       0.6828          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_421.67        73.76     1_495.43       0.6828          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_421.67       667.93     2_089.60       0.6833          NaN         5.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
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
Exhaustive (query)                                         3.33     1_518.22     1_521.56       1.0000     0.000000        18.31
Exhaustive (self)                                          3.33    15_444.95    15_448.28       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.27       714.16       721.43       0.4800          NaN         4.58
Exhaustive-SQ8 (self)                                      7.27     7_564.65     7_571.92       0.4862          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               388.70        53.48       442.18       0.4789          NaN         5.76
IVF-SQ8-nl273-np16 (query)                               388.70        64.68       453.38       0.4788          NaN         5.76
IVF-SQ8-nl273-np23 (query)                               388.70        87.70       476.40       0.4787          NaN         5.76
IVF-SQ8-nl273 (self)                                     388.70       849.90     1_238.60       0.4863          NaN         5.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               738.82        56.97       795.79       0.4790          NaN         5.77
IVF-SQ8-nl387-np27 (query)                               738.82        74.76       813.59       0.4790          NaN         5.77
IVF-SQ8-nl387 (self)                                     738.82       724.25     1_463.07       0.4862          NaN         5.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_417.10        52.34     1_469.44       0.4803          NaN         5.79
IVF-SQ8-nl547-np27 (query)                             1_417.10        59.86     1_476.95       0.4802          NaN         5.79
IVF-SQ8-nl547-np33 (query)                             1_417.10        69.46     1_486.56       0.4802          NaN         5.79
IVF-SQ8-nl547 (self)                                   1_417.10       661.10     2_078.20       0.4866          NaN         5.79
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
Exhaustive (query)                                        14.28     6_018.25     6_032.52       1.0000     0.000000        73.24
Exhaustive (self)                                         14.28    60_845.88    60_860.16       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    39.86     1_625.05     1_664.91       0.8081          NaN        18.31
Exhaustive-SQ8 (self)                                     39.86    16_991.63    17_031.49       0.8095          NaN        18.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               423.41       238.19       661.59       0.8063          NaN        19.59
IVF-SQ8-nl273-np16 (query)                               423.41       286.32       709.73       0.8063          NaN        19.59
IVF-SQ8-nl273-np23 (query)                               423.41       392.99       816.39       0.8063          NaN        19.59
IVF-SQ8-nl273 (self)                                     423.41     3_900.55     4_323.95       0.8082          NaN        19.59
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               761.01       248.97     1_009.98       0.8063          NaN        19.65
IVF-SQ8-nl387-np27 (query)                               761.01       325.67     1_086.68       0.8063          NaN        19.65
IVF-SQ8-nl387 (self)                                     761.01     3_237.61     3_998.63       0.8086          NaN        19.65
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_518.34       235.01     1_753.34       0.8078          NaN        19.73
IVF-SQ8-nl547-np27 (query)                             1_518.34       257.37     1_775.71       0.8078          NaN        19.73
IVF-SQ8-nl547-np33 (query)                             1_518.34       303.26     1_821.59       0.8078          NaN        19.73
IVF-SQ8-nl547 (self)                                   1_518.34     3_007.62     4_525.96       0.8097          NaN        19.73
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
Exhaustive (query)                                        14.65     6_079.45     6_094.11       1.0000     0.000000        73.24
Exhaustive (self)                                         14.65    61_613.98    61_628.63       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 775.01       872.95     1_647.96       0.0969          NaN         1.27
Exhaustive-PQ-m8 (self)                                  775.01     8_670.70     9_445.71       0.0835          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_035.59     1_914.18     2_949.77       0.1328          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_035.59    18_756.13    19_791.71       0.0967          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_286.10       305.30     1_591.41       0.1530          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           1_286.10       364.86     1_650.97       0.1532          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           1_286.10       523.28     1_809.38       0.1533          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 1_286.10     5_118.19     6_404.30       0.1037          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_501.27       504.23     2_005.50       0.2676          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          1_501.27       610.92     2_112.20       0.2689          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          1_501.27       876.42     2_377.69       0.2696          NaN         3.69
IVF-PQ-nl273-m16 (self)                                1_501.27     8_460.68     9_961.95       0.1781          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_835.00       419.85     2_254.85       0.1545          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           1_835.00       579.16     2_414.16       0.1545          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 1_835.00     5_738.88     7_573.87       0.1042          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          2_170.65       666.60     2_837.25       0.2703          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          2_170.65       884.96     3_055.61       0.2711          NaN         3.75
IVF-PQ-nl387-m16 (self)                                2_170.65     8_904.87    11_075.53       0.1807          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           3_062.69       498.34     3_561.03       0.1570          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           3_062.69       579.06     3_641.75       0.1571          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           3_062.69       713.05     3_775.74       0.1572          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 3_062.69     6_974.10    10_036.79       0.1059          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          3_362.81       735.52     4_098.33       0.2711          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          3_362.81       855.82     4_218.63       0.2723          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          3_362.81     1_024.08     4_386.89       0.2727          NaN         3.83
IVF-PQ-nl547-m16 (self)                                3_362.81    10_286.15    13_648.95       0.1824          NaN         3.83
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
Exhaustive (query)                                        15.01     6_185.76     6_200.77       1.0000     0.000000        73.24
Exhaustive (self)                                         15.01    60_778.28    60_793.29       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 765.41       863.78     1_629.19       0.2215          NaN         1.27
Exhaustive-PQ-m8 (self)                                  765.41     8_694.17     9_459.59       0.1616          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_037.73     1_870.26     2_907.99       0.3518          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_037.73    18_712.09    19_749.82       0.2621          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_099.21       308.52     1_407.73       0.3766          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           1_099.21       404.20     1_503.40       0.3768          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           1_099.21       526.71     1_625.92       0.3768          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 1_099.21     5_190.09     6_289.30       0.2757          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_398.66       487.47     1_886.13       0.5492          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          1_398.66       607.52     2_006.19       0.5498          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          1_398.66       860.73     2_259.39       0.5502          NaN         3.69
IVF-PQ-nl273-m16 (self)                                1_398.66     8_447.65     9_846.32       0.4521          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_624.11       418.08     2_042.19       0.3817          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           1_624.11       570.08     2_194.19       0.3817          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 1_624.11     5_668.09     7_292.20       0.2788          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          1_950.94       667.99     2_618.94       0.5528          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          1_950.94       916.66     2_867.60       0.5532          NaN         3.75
IVF-PQ-nl387-m16 (self)                                1_950.94     9_028.20    10_979.14       0.4547          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           2_429.50       500.30     2_929.80       0.3862          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           2_429.50       561.62     2_991.12       0.3862          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           2_429.50       681.88     3_111.38       0.3863          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 2_429.50     6_872.47     9_301.98       0.2832          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          2_714.39       727.23     3_441.63       0.5576          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          2_714.39       857.52     3_571.91       0.5581          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          2_714.39     1_032.66     3_747.05       0.5583          NaN         3.83
IVF-PQ-nl547-m16 (self)                                2_714.39    10_269.25    12_983.64       0.4599          NaN         3.83
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
Exhaustive (query)                                        14.98     6_075.21     6_090.19       1.0000     0.000000        73.24
Exhaustive (self)                                         14.98    61_926.28    61_941.25       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 784.04       871.12     1_655.15       0.2587          NaN         1.27
Exhaustive-PQ-m8 (self)                                  784.04     8_654.46     9_438.50       0.1987          NaN         1.27
Exhaustive-PQ-m16 (query)                              1_043.32     1_877.09     2_920.40       0.3957          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_043.32    18_740.57    19_783.89       0.2941          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_231.94       322.56     1_554.51       0.5459          NaN         2.55
IVF-PQ-nl273-m8-np16 (query)                           1_231.94       388.29     1_620.24       0.5459          NaN         2.55
IVF-PQ-nl273-m8-np23 (query)                           1_231.94       563.55     1_795.49       0.5459          NaN         2.55
IVF-PQ-nl273-m8 (self)                                 1_231.94     5_395.18     6_627.12       0.4006          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_424.81       521.32     1_946.13       0.6915          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                          1_424.81       632.36     2_057.16       0.6915          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                          1_424.81       893.06     2_317.87       0.6915          NaN         3.69
IVF-PQ-nl273-m16 (self)                                1_424.81     8_468.57     9_893.38       0.5839          NaN         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_463.84       415.33     1_879.18       0.5538          NaN         2.61
IVF-PQ-nl387-m8-np27 (query)                           1_463.84       560.57     2_024.41       0.5538          NaN         2.61
IVF-PQ-nl387-m8 (self)                                 1_463.84     5_746.53     7_210.37       0.4015          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          1_791.93       637.58     2_429.51       0.6976          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                          1_791.93       904.95     2_696.88       0.6976          NaN         3.75
IVF-PQ-nl387-m16 (self)                                1_791.93     8_894.88    10_686.81       0.5860          NaN         3.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           2_279.15       481.67     2_760.82       0.5594          NaN         2.69
IVF-PQ-nl547-m8-np27 (query)                           2_279.15       564.86     2_844.01       0.5594          NaN         2.69
IVF-PQ-nl547-m8-np33 (query)                           2_279.15       681.93     2_961.08       0.5594          NaN         2.69
IVF-PQ-nl547-m8 (self)                                 2_279.15     6_878.71     9_157.86       0.4052          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          2_622.02       685.66     3_307.68       0.7013          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                          2_622.02       852.78     3_474.80       0.7013          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                          2_622.02     1_016.41     3_638.42       0.7013          NaN         3.83
IVF-PQ-nl547-m16 (self)                                2_622.02    10_122.90    12_744.91       0.5895          NaN         3.83
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
Exhaustive (query)                                        28.97    14_899.95    14_928.93       1.0000     0.000000       146.48
Exhaustive (self)                                         28.97   151_065.31   151_094.28       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_482.97     1_886.87     3_369.84       0.0968          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_482.97    18_929.38    20_412.35       0.0817          NaN         2.54
Exhaustive-PQ-m32 (query)                              1_991.71     4_290.92     6_282.63       0.1321          NaN         4.83
Exhaustive-PQ-m32 (self)                               1_991.71    42_977.02    44_968.73       0.0941          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_413.40    11_627.29    15_040.70       0.2669          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_413.40   116_282.00   119_695.40       0.1882          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_325.17       598.52     2_923.69       0.1529          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          2_325.17       729.49     3_054.65       0.1534          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          2_325.17     1_033.68     3_358.85       0.1536          NaN         3.95
IVF-PQ-nl273-m16 (self)                                2_325.17    10_352.14    12_677.31       0.1033          NaN         3.95
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_836.66       933.63     3_770.28       0.2653          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          2_836.66     1_133.36     3_970.02       0.2673          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          2_836.66     1_607.54     4_444.19       0.2678          NaN         6.24
IVF-PQ-nl273-m32 (self)                                2_836.66    16_145.64    18_982.30       0.1821          NaN         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_162.11     1_736.16     5_898.27       0.5114          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          4_162.11     2_105.27     6_267.37       0.5180          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          4_162.11     2_984.40     7_146.51       0.5205          NaN        10.82
IVF-PQ-nl273-m64 (self)                                4_162.11    32_414.85    36_576.96       0.4336          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          3_301.37       813.82     4_115.18       0.1535          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          3_301.37     1_112.93     4_414.30       0.1537          NaN         4.06
IVF-PQ-nl387-m16 (self)                                3_301.37    11_128.40    14_429.77       0.1036          NaN         4.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_948.34     1_260.20     5_208.54       0.2664          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          3_948.34     1_749.47     5_697.81       0.2674          NaN         6.35
IVF-PQ-nl387-m32 (self)                                3_948.34    17_668.76    21_617.11       0.1827          NaN         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          5_697.45     2_468.78     8_166.24       0.5160          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          5_697.45     3_372.67     9_070.13       0.5202          NaN        10.93
IVF-PQ-nl387-m64 (self)                                5_697.45    31_773.78    37_471.23       0.4347          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          5_164.29       927.45     6_091.75       0.1549          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          5_164.29     1_068.78     6_233.07       0.1553          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          5_164.29     1_285.20     6_449.49       0.1555          NaN         4.22
IVF-PQ-nl547-m16 (self)                                5_164.29    12_820.85    17_985.14       0.1049          NaN         4.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          5_651.18     1_407.57     7_058.75       0.2686          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                          5_651.18     1_626.55     7_277.73       0.2704          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                          5_651.18     1_972.01     7_623.19       0.2708          NaN         6.51
IVF-PQ-nl547-m32 (self)                                5_651.18    20_021.40    25_672.58       0.1839          NaN         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          7_082.08     2_538.59     9_620.66       0.5158          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                          7_082.08     2_952.95    10_035.03       0.5221          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                          7_082.08     3_586.45    10_668.53       0.5240          NaN        11.09
IVF-PQ-nl547-m64 (self)                                7_082.08    35_548.38    42_630.45       0.4379          NaN        11.09
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
Exhaustive (query)                                        29.26    15_316.79    15_346.05       1.0000     0.000000       146.48
Exhaustive (self)                                         29.26   159_994.47   160_023.73       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_398.14     1_893.72     3_291.86       0.2357          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_398.14    18_955.91    20_354.05       0.1675          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_000.71     4_298.74     6_299.45       0.3554          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_000.71    43_038.39    45_039.10       0.2679          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_587.00    11_670.34    15_257.33       0.4755          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_587.00   117_383.01   120_970.01       0.3907          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_353.98       604.78     2_958.77       0.3836          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          2_353.98       753.66     3_107.65       0.3847          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          2_353.98     1_044.26     3_398.24       0.3852          NaN         3.95
IVF-PQ-nl273-m16 (self)                                2_353.98    10_409.54    12_763.52       0.2876          NaN         3.95
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_859.86       972.25     3_832.11       0.5548          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          2_859.86     1_192.56     4_052.42       0.5569          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          2_859.86     1_684.22     4_544.08       0.5578          NaN         6.24
IVF-PQ-nl273-m32 (self)                                2_859.86    16_432.87    19_292.73       0.4667          NaN         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_449.93     1_769.04     6_218.97       0.7063          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          4_449.93     2_173.74     6_623.67       0.7100          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          4_449.93     3_038.67     7_488.60       0.7118          NaN        10.82
IVF-PQ-nl273-m64 (self)                                4_449.93    30_317.03    34_766.96       0.6484          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          3_340.30       837.43     4_177.73       0.3869          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          3_340.30     1_144.95     4_485.25       0.3875          NaN         4.06
IVF-PQ-nl387-m16 (self)                                3_340.30    11_263.43    14_603.74       0.2905          NaN         4.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_835.49     1_266.22     5_101.71       0.5561          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          3_835.49     1_771.32     5_606.81       0.5576          NaN         6.35
IVF-PQ-nl387-m32 (self)                                3_835.49    17_362.84    21_198.33       0.4682          NaN         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          5_190.43     2_275.28     7_465.71       0.7108          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          5_190.43     3_185.23     8_375.66       0.7133          NaN        10.93
IVF-PQ-nl387-m64 (self)                                5_190.43    31_689.15    36_879.58       0.6499          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          4_367.65       917.98     5_285.64       0.3895          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          4_367.65     1_084.32     5_451.97       0.3901          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          4_367.65     1_332.72     5_700.37       0.3904          NaN         4.22
IVF-PQ-nl547-m16 (self)                                4_367.65    12_894.07    17_261.72       0.2925          NaN         4.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          5_021.69     1_416.43     6_438.12       0.5574          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                          5_021.69     1_644.38     6_666.06       0.5590          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                          5_021.69     2_001.72     7_023.41       0.5600          NaN         6.51
IVF-PQ-nl547-m32 (self)                                5_021.69    19_916.57    24_938.25       0.4697          NaN         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          6_376.68     2_502.88     8_879.56       0.7121          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                          6_376.68     2_930.16     9_306.84       0.7150          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                          6_376.68     3_568.11     9_944.80       0.7166          NaN        11.09
IVF-PQ-nl547-m64 (self)                                6_376.68    35_480.78    41_857.46       0.6535          NaN        11.09
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
Exhaustive (query)                                        29.31    15_411.00    15_440.31       1.0000     0.000000       146.48
Exhaustive (self)                                         29.31   154_381.14   154_410.45       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_570.03     1_938.41     3_508.44       0.3801          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_570.03    19_338.14    20_908.17       0.2823          NaN         2.54
Exhaustive-PQ-m32 (query)                              2_179.53     4_389.35     6_568.88       0.5114          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_179.53    44_013.75    46_193.28       0.4053          NaN         4.83
Exhaustive-PQ-m64 (query)                              3_667.85    11_930.54    15_598.38       0.6355          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_667.85   120_908.79   124_576.64       0.5576          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_380.25       618.66     2_998.91       0.6299          NaN         3.95
IVF-PQ-nl273-m16-np16 (query)                          2_380.25       746.61     3_126.86       0.6300          NaN         3.95
IVF-PQ-nl273-m16-np23 (query)                          2_380.25     1_051.95     3_432.19       0.6300          NaN         3.95
IVF-PQ-nl273-m16 (self)                                2_380.25    10_506.62    12_886.86       0.4800          NaN         3.95
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_720.17       955.48     3_675.66       0.7571          NaN         6.24
IVF-PQ-nl273-m32-np16 (query)                          2_720.17     1_162.58     3_882.75       0.7572          NaN         6.24
IVF-PQ-nl273-m32-np23 (query)                          2_720.17     1_647.78     4_367.95       0.7572          NaN         6.24
IVF-PQ-nl273-m32 (self)                                2_720.17    16_535.21    19_255.38       0.6617          NaN         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_110.69     1_741.00     5_851.69       0.8790          NaN        10.82
IVF-PQ-nl273-m64-np16 (query)                          4_110.69     2_129.53     6_240.21       0.8792          NaN        10.82
IVF-PQ-nl273-m64-np23 (query)                          4_110.69     3_030.72     7_141.40       0.8792          NaN        10.82
IVF-PQ-nl273-m64 (self)                                4_110.69    30_101.85    34_212.54       0.8360          NaN        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          2_734.21       795.82     3_530.03       0.6291          NaN         4.06
IVF-PQ-nl387-m16-np27 (query)                          2_734.21     1_161.29     3_895.49       0.6291          NaN         4.06
IVF-PQ-nl387-m16 (self)                                2_734.21    11_171.40    13_905.61       0.4710          NaN         4.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_321.01     1_237.20     4_558.21       0.7564          NaN         6.35
IVF-PQ-nl387-m32-np27 (query)                          3_321.01     1_737.62     5_058.63       0.7564          NaN         6.35
IVF-PQ-nl387-m32 (self)                                3_321.01    17_204.63    20_525.65       0.6572          NaN         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          4_754.60     2_246.38     7_000.98       0.8820          NaN        10.93
IVF-PQ-nl387-m64-np27 (query)                          4_754.60     3_148.75     7_903.35       0.8820          NaN        10.93
IVF-PQ-nl387-m64 (self)                                4_754.60    31_873.94    36_628.54       0.8375          NaN        10.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          4_197.34       924.95     5_122.30       0.6317          NaN         4.22
IVF-PQ-nl547-m16-np27 (query)                          4_197.34     1_070.53     5_267.88       0.6317          NaN         4.22
IVF-PQ-nl547-m16-np33 (query)                          4_197.34     1_305.10     5_502.44       0.6317          NaN         4.22
IVF-PQ-nl547-m16 (self)                                4_197.34    13_166.76    17_364.11       0.4646          NaN         4.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          4_609.69     1_398.74     6_008.43       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np27 (query)                          4_609.69     1_647.26     6_256.95       0.7595          NaN         6.51
IVF-PQ-nl547-m32-np33 (query)                          4_609.69     1_997.50     6_607.19       0.7595          NaN         6.51
IVF-PQ-nl547-m32 (self)                                4_609.69    19_983.52    24_593.21       0.6549          NaN         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          5_956.68     2_514.67     8_471.35       0.8845          NaN        11.09
IVF-PQ-nl547-m64-np27 (query)                          5_956.68     2_942.62     8_899.31       0.8845          NaN        11.09
IVF-PQ-nl547-m64-np33 (query)                          5_956.68     3_563.81     9_520.50       0.8845          NaN        11.09
IVF-PQ-nl547-m64 (self)                                5_956.68    35_523.83    41_480.52       0.8389          NaN        11.09
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
Exhaustive (query)                                        14.68     6_328.40     6_343.08       1.0000     0.000000        73.24
Exhaustive (self)                                         14.68    62_319.57    62_334.25       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              3_226.95       898.64     4_125.59       0.0958          NaN         1.33
Exhaustive-OPQ-m8 (self)                               3_226.95     9_031.95    12_258.90       0.0833          NaN         1.33
Exhaustive-OPQ-m16 (query)                             3_665.49     1_938.80     5_604.29       0.1326          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_665.49    19_383.69    23_049.18       0.0961          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          3_679.98       310.11     3_990.09       0.1519          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          3_679.98       381.63     4_061.61       0.1520          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          3_679.98       544.31     4_224.29       0.1521          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                3_679.98     5_489.02     9_169.00       0.1034          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         4_440.00       533.41     4_973.42       0.2671          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         4_440.00       615.84     5_055.85       0.2686          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         4_440.00       873.30     5_313.30       0.2690          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               4_440.00     9_009.12    13_449.12       0.1785          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          4_704.06       416.96     5_121.02       0.1547          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          4_704.06       575.52     5_279.58       0.1549          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                4_704.06     5_942.34    10_646.40       0.1045          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         5_079.58       622.81     5_702.39       0.2683          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         5_079.58       917.65     5_997.24       0.2690          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               5_079.58     9_266.36    14_345.95       0.1801          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          5_688.13       482.45     6_170.58       0.1559          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          5_688.13       583.43     6_271.56       0.1562          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          5_688.13       692.59     6_380.72       0.1562          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                5_688.13     7_139.16    12_827.29       0.1053          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         6_331.64       783.65     7_115.29       0.2693          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         6_331.64       856.62     7_188.26       0.2705          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         6_331.64     1_028.69     7_360.33       0.2710          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               6_331.64    10_805.75    17_137.39       0.1820          NaN         3.89
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
Exhaustive (query)                                        15.16     6_196.18     6_211.35       1.0000     0.000000        73.24
Exhaustive (self)                                         15.16    63_309.54    63_324.70       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              3_220.03       865.78     4_085.81       0.2221          NaN         1.33
Exhaustive-OPQ-m8 (self)                               3_220.03     8_915.02    12_135.05       0.1616          NaN         1.33
Exhaustive-OPQ-m16 (query)                             3_582.83     1_873.81     5_456.64       0.3510          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_582.83    19_290.96    22_873.79       0.2620          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          3_523.75       299.45     3_823.20       0.3759          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          3_523.75       360.37     3_884.12       0.3760          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          3_523.75       515.76     4_039.51       0.3761          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                3_523.75     5_253.95     8_777.70       0.2759          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         4_105.41       484.83     4_590.24       0.5484          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         4_105.41       577.27     4_682.68       0.5491          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         4_105.41       821.40     4_926.81       0.5493          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               4_105.41     8_441.94    12_547.34       0.4515          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          3_767.22       409.71     4_176.93       0.3820          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          3_767.22       572.76     4_339.98       0.3821          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                3_767.22     5_945.52     9_712.75       0.2792          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         4_660.58       645.75     5_306.32       0.5522          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         4_660.58       927.86     5_588.43       0.5525          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               4_660.58     9_423.42    14_084.00       0.4549          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          4_631.61       460.09     5_091.70       0.3861          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          4_631.61       544.40     5_176.00       0.3862          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          4_631.61       652.33     5_283.94       0.3862          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                4_631.61     6_806.36    11_437.97       0.2836          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         5_667.63       744.89     6_412.51       0.5567          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         5_667.63       891.73     6_559.36       0.5572          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         5_667.63     1_113.26     6_780.89       0.5574          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               5_667.63    10_942.59    16_610.22       0.4593          NaN         3.89
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
Exhaustive (query)                                        14.84     6_121.34     6_136.18       1.0000     0.000000        73.24
Exhaustive (self)                                         14.84    61_252.18    61_267.02       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              2_893.62       863.18     3_756.79       0.2585          NaN         1.33
Exhaustive-OPQ-m8 (self)                               2_893.62     8_919.36    11_812.97       0.1648          NaN         1.33
Exhaustive-OPQ-m16 (query)                             3_711.29     1_949.77     5_661.05       0.3956          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_711.29    19_268.05    22_979.34       0.2451          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          3_257.34       299.40     3_556.73       0.6165          NaN         2.61
IVF-OPQ-nl273-m8-np16 (query)                          3_257.34       381.98     3_639.31       0.6165          NaN         2.61
IVF-OPQ-nl273-m8-np23 (query)                          3_257.34       530.73     3_788.07       0.6165          NaN         2.61
IVF-OPQ-nl273-m8 (self)                                3_257.34     5_285.82     8_543.15       0.5159          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         4_154.23       471.93     4_626.16       0.7432          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                         4_154.23       588.76     4_742.99       0.7432          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                         4_154.23       883.53     5_037.76       0.7432          NaN         3.76
IVF-OPQ-nl273-m16 (self)                               4_154.23     8_455.87    12_610.11       0.6640          NaN         3.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          3_641.13       395.71     4_036.84       0.6235          NaN         2.67
IVF-OPQ-nl387-m8-np27 (query)                          3_641.13       556.65     4_197.78       0.6235          NaN         2.67
IVF-OPQ-nl387-m8 (self)                                3_641.13     5_746.04     9_387.16       0.5234          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         4_515.55       659.83     5_175.38       0.7467          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                         4_515.55       905.49     5_421.04       0.7467          NaN         3.81
IVF-OPQ-nl387-m16 (self)                               4_515.55     9_333.28    13_848.83       0.6683          NaN         3.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          4_441.56       458.05     4_899.61       0.6278          NaN         2.75
IVF-OPQ-nl547-m8-np27 (query)                          4_441.56       540.43     4_981.99       0.6278          NaN         2.75
IVF-OPQ-nl547-m8-np33 (query)                          4_441.56       654.01     5_095.57       0.6278          NaN         2.75
IVF-OPQ-nl547-m8 (self)                                4_441.56     6_809.21    11_250.78       0.5284          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         5_409.88       729.53     6_139.41       0.7479          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                         5_409.88       837.09     6_246.98       0.7479          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                         5_409.88     1_017.34     6_427.23       0.7479          NaN         3.89
IVF-OPQ-nl547-m16 (self)                               5_409.88    10_577.31    15_987.19       0.6713          NaN         3.89
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
Exhaustive (query)                                        29.62    14_863.24    14_892.86       1.0000     0.000000       146.48
Exhaustive (self)                                         29.62   152_557.96   152_587.58       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             6_106.83     1_894.24     8_001.07       0.0953          NaN         2.79
Exhaustive-OPQ-m16 (self)                              6_106.83    19_829.09    25_935.92       0.0813          NaN         2.79
Exhaustive-OPQ-m32 (query)                             7_662.08     4_302.67    11_964.75       0.1296          NaN         5.08
Exhaustive-OPQ-m32 (self)                              7_662.08    44_050.42    51_712.50       0.0937          NaN         5.08
Exhaustive-OPQ-m64 (query)                            11_730.77    12_035.75    23_766.52       0.2634          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_730.77   119_511.06   131_241.83       0.1887          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         7_979.50       595.38     8_574.88       0.1507          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                         7_979.50       713.84     8_693.34       0.1514          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                         7_979.50       992.34     8_971.84       0.1515          NaN         4.20
IVF-OPQ-nl273-m16 (self)                               7_979.50    10_979.40    18_958.90       0.1025          NaN         4.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         8_533.81       913.18     9_446.99       0.2648          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                         8_533.81     1_119.10     9_652.91       0.2670          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                         8_533.81     1_581.10    10_114.92       0.2675          NaN         6.49
IVF-OPQ-nl273-m32 (self)                               8_533.81    16_921.61    25_455.42       0.1822          NaN         6.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        12_523.84     1_729.30    14_253.14       0.5108          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        12_523.84     2_120.16    14_644.00       0.5174          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        12_523.84     2_958.97    15_482.81       0.5199          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              12_523.84    30_799.17    43_323.02       0.4333          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         7_855.49       803.49     8_658.98       0.1524          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                         7_855.49     1_116.09     8_971.58       0.1525          NaN         4.31
IVF-OPQ-nl387-m16 (self)                               7_855.49    12_034.19    19_889.67       0.1040          NaN         4.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_485.43     1_216.31    10_701.74       0.2645          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                         9_485.43     1_683.54    11_168.97       0.2654          NaN         6.60
IVF-OPQ-nl387-m32 (self)                               9_485.43    17_663.29    27_148.72       0.1812          NaN         6.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        13_480.97     2_263.42    15_744.39       0.5155          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        13_480.97     3_131.60    16_612.57       0.5201          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              13_480.97    33_335.50    46_816.48       0.4343          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         9_568.57       897.67    10_466.24       0.1537          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                         9_568.57     1_049.84    10_618.41       0.1540          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                         9_568.57     1_249.48    10_818.05       0.1541          NaN         4.47
IVF-OPQ-nl547-m16 (self)                               9_568.57    13_354.67    22_923.24       0.1044          NaN         4.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                        11_480.29     1_373.40    12_853.69       0.2664          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        11_480.29     1_673.21    13_153.49       0.2683          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        11_480.29     1_912.58    13_392.87       0.2689          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              11_480.29    20_016.22    31_496.51       0.1840          NaN         6.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        15_394.25     2_514.31    17_908.55       0.5154          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        15_394.25     2_920.45    18_314.70       0.5220          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        15_394.25     3_502.61    18_896.86       0.5244          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              15_394.25    36_142.49    51_536.74       0.4373          NaN        11.34
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
Exhaustive (query)                                        29.90    15_012.79    15_042.69       1.0000     0.000000       146.48
Exhaustive (self)                                         29.90   150_219.67   150_249.58       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             6_416.05     1_991.40     8_407.45       0.2364          NaN         2.79
Exhaustive-OPQ-m16 (self)                              6_416.05    20_239.00    26_655.04       0.1672          NaN         2.79
Exhaustive-OPQ-m32 (query)                             8_163.78     4_429.83    12_593.61       0.3537          NaN         5.08
Exhaustive-OPQ-m32 (self)                              8_163.78    44_743.34    52_907.12       0.2677          NaN         5.08
Exhaustive-OPQ-m64 (query)                            11_778.39    11_847.56    23_625.95       0.4830          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_778.39   119_554.02   131_332.41       0.4018          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         7_066.24       579.24     7_645.48       0.3841          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                         7_066.24       713.70     7_779.94       0.3851          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                         7_066.24     1_000.99     8_067.23       0.3855          NaN         4.20
IVF-OPQ-nl273-m16 (self)                               7_066.24    11_005.15    18_071.39       0.2879          NaN         4.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         8_456.24       903.04     9_359.29       0.5535          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                         8_456.24     1_123.11     9_579.36       0.5559          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                         8_456.24     1_560.78    10_017.03       0.5570          NaN         6.49
IVF-OPQ-nl273-m32 (self)                               8_456.24    16_616.70    25_072.95       0.4668          NaN         6.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        12_433.43     1_729.56    14_162.99       0.7100          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        12_433.43     2_108.24    14_541.67       0.7139          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        12_433.43     2_970.92    15_404.34       0.7156          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              12_433.43    30_672.41    43_105.84       0.6537          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         7_721.31       749.62     8_470.93       0.3882          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                         7_721.31     1_048.90     8_770.20       0.3887          NaN         4.31
IVF-OPQ-nl387-m16 (self)                               7_721.31    11_443.38    19_164.68       0.2903          NaN         4.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_602.82     1_217.98    10_820.81       0.5569          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                         9_602.82     1_673.00    11_275.82       0.5584          NaN         6.60
IVF-OPQ-nl387-m32 (self)                               9_602.82    18_138.92    27_741.75       0.4675          NaN         6.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        14_324.77     2_247.14    16_571.91       0.7131          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        14_324.77     3_186.60    17_511.37       0.7156          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              14_324.77    32_765.85    47_090.62       0.6546          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         9_298.92       850.73    10_149.66       0.3895          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                         9_298.92     1_025.83    10_324.75       0.3902          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                         9_298.92     1_218.50    10_517.42       0.3905          NaN         4.47
IVF-OPQ-nl547-m16 (self)                               9_298.92    13_182.54    22_481.46       0.2926          NaN         4.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                        11_078.43     1_396.15    12_474.58       0.5567          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        11_078.43     1_628.73    12_707.17       0.5584          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        11_078.43     1_975.30    13_053.73       0.5593          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              11_078.43    20_541.82    31_620.25       0.4697          NaN         6.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        15_421.89     2_499.50    17_921.39       0.7134          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        15_421.89     2_912.91    18_334.80       0.7161          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        15_421.89     3_545.87    18_967.76       0.7176          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              15_421.89    36_627.03    52_048.92       0.6576          NaN        11.34
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
Exhaustive (query)                                        29.15    15_467.89    15_497.04       1.0000     0.000000       146.48
Exhaustive (self)                                         29.15   154_507.27   154_536.42       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             6_936.47     1_934.45     8_870.93       0.3792          NaN         2.79
Exhaustive-OPQ-m16 (self)                              6_936.47    20_632.44    27_568.91       0.2243          NaN         2.79
Exhaustive-OPQ-m32 (query)                             7_969.37     4_405.51    12_374.88       0.5109          NaN         5.08
Exhaustive-OPQ-m32 (self)                              7_969.37    44_797.11    52_766.48       0.3554          NaN         5.08
Exhaustive-OPQ-m64 (query)                            11_826.69    11_878.47    23_705.15       0.6362          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_826.69   120_999.56   132_826.25       0.5351          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         7_047.80       577.90     7_625.70       0.6880          NaN         4.20
IVF-OPQ-nl273-m16-np16 (query)                         7_047.80       708.53     7_756.33       0.6880          NaN         4.20
IVF-OPQ-nl273-m16-np23 (query)                         7_047.80       999.51     8_047.31       0.6880          NaN         4.20
IVF-OPQ-nl273-m16 (self)                               7_047.80    11_094.03    18_141.83       0.6019          NaN         4.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         8_908.25       923.95     9_832.21       0.7965          NaN         6.49
IVF-OPQ-nl273-m32-np16 (query)                         8_908.25     1_145.76    10_054.01       0.7966          NaN         6.49
IVF-OPQ-nl273-m32-np23 (query)                         8_908.25     1_637.41    10_545.66       0.7966          NaN         6.49
IVF-OPQ-nl273-m32 (self)                               8_908.25    16_894.94    25_803.19       0.7263          NaN         6.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        13_619.18     1_744.49    15_363.66       0.8951          NaN        11.07
IVF-OPQ-nl273-m64-np16 (query)                        13_619.18     2_132.84    15_752.02       0.8952          NaN        11.07
IVF-OPQ-nl273-m64-np23 (query)                        13_619.18     3_038.38    16_657.56       0.8952          NaN        11.07
IVF-OPQ-nl273-m64 (self)                              13_619.18    31_433.99    45_053.17       0.8498          NaN        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         7_641.19       774.41     8_415.61       0.6892          NaN         4.31
IVF-OPQ-nl387-m16-np27 (query)                         7_641.19     1_070.40     8_711.59       0.6892          NaN         4.31
IVF-OPQ-nl387-m16 (self)                               7_641.19    11_714.84    19_356.04       0.6035          NaN         4.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_507.72     1_217.74    10_725.45       0.7955          NaN         6.60
IVF-OPQ-nl387-m32-np27 (query)                         9_507.72     1_695.01    11_202.73       0.7955          NaN         6.60
IVF-OPQ-nl387-m32 (self)                               9_507.72    17_863.15    27_370.87       0.7274          NaN         6.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        13_321.02     2_243.89    15_564.91       0.8963          NaN        11.18
IVF-OPQ-nl387-m64-np27 (query)                        13_321.02     3_134.19    16_455.22       0.8963          NaN        11.18
IVF-OPQ-nl387-m64 (self)                              13_321.02    32_287.55    45_608.58       0.8510          NaN        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         8_515.89       909.04     9_424.93       0.6894          NaN         4.47
IVF-OPQ-nl547-m16-np27 (query)                         8_515.89     1_058.45     9_574.34       0.6894          NaN         4.47
IVF-OPQ-nl547-m16-np33 (query)                         8_515.89     1_300.37     9_816.26       0.6894          NaN         4.47
IVF-OPQ-nl547-m16 (self)                               8_515.89    14_098.97    22_614.86       0.6073          NaN         4.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                        10_226.81     1_370.20    11_597.00       0.7953          NaN         6.76
IVF-OPQ-nl547-m32-np27 (query)                        10_226.81     1_589.69    11_816.50       0.7953          NaN         6.76
IVF-OPQ-nl547-m32-np33 (query)                        10_226.81     1_955.20    12_182.01       0.7953          NaN         6.76
IVF-OPQ-nl547-m32 (self)                              10_226.81    20_281.00    30_507.81       0.7286          NaN         6.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        14_269.24     2_471.87    16_741.10       0.8962          NaN        11.34
IVF-OPQ-nl547-m64-np27 (query)                        14_269.24     2_910.56    17_179.80       0.8962          NaN        11.34
IVF-OPQ-nl547-m64-np33 (query)                        14_269.24     3_534.93    17_804.17       0.8962          NaN        11.34
IVF-OPQ-nl547-m64 (self)                              14_269.24    36_390.01    50_659.25       0.8529          NaN        11.34
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
