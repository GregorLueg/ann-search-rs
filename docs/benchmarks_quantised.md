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
benchmarked. Index size in memory is also provided. Compared to other
benchmarks, we will use the `"correlated"`, `"lowrank"` and `"quantisation"`
with higher dimensionality, but reduced samples (for the sake of fast'ish
benchmarking). The different synthetic data types pose different challenges
for the quantisation methods.

## Table of Contents

- [BF16 quantisation](#bf16-ivf-and-exhaustive)
- [SQ8 quantisation](#sq8-ivf-and-exhaustive)
- [Product quantisation](#product-quantisation-exhaustive-and-ivf)
- [Optimised product quantisation](#optimised-product-quantisation-exhaustive-and-ivf)

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
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.11     1_567.27     1_570.38       1.0000     0.000000        18.31
Exhaustive (self)                                          3.11    15_125.00    15_128.12       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                                    5.02     1_163.59     1_168.60       0.9881     0.078075         9.16
Exhaustive-BF16 (self)                                     5.02    15_489.98    15_494.99       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                              391.78        87.59       479.37       0.9830     0.093524         9.19
IVF-BF16-nl273-np16 (query)                              391.78       105.23       497.02       0.9870     0.080498         9.19
IVF-BF16-nl273-np23 (query)                              391.78       140.82       532.60       0.9881     0.078075         9.19
IVF-BF16-nl273 (self)                                    391.78     1_425.45     1_817.23       0.9852     0.115046         9.19
IVF-BF16-nl387-np19 (query)                              739.18        86.68       825.86       0.9860     0.088865         9.21
IVF-BF16-nl387-np27 (query)                              739.18       117.19       856.37       0.9881     0.078075         9.21
IVF-BF16-nl387 (self)                                    739.18     1_186.47     1_925.65       0.9852     0.115046         9.21
IVF-BF16-nl547-np23 (query)                            1_459.62        83.02     1_542.64       0.9845     0.092194         9.23
IVF-BF16-nl547-np27 (query)                            1_459.62        91.81     1_551.43       0.9874     0.079872         9.23
IVF-BF16-nl547-np33 (query)                            1_459.62       108.38     1_568.01       0.9880     0.078200         9.23
IVF-BF16-nl547 (self)                                  1_459.62     1_092.43     2_552.05       0.9851     0.115167         9.23
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.92     1_537.15     1_541.07       1.0000     0.000000        18.88
Exhaustive (self)                                          3.92    15_137.21    15_141.13       1.0000     0.000000        18.88
Exhaustive-BF16 (query)                                    7.14     1_231.72     1_238.86       0.9383     0.000243         9.44
Exhaustive-BF16 (self)                                     7.14    15_029.61    15_036.75       1.0000     0.000000         9.44
IVF-BF16-nl273-np13 (query)                              368.00        92.39       460.39       0.9352     0.000249         9.48
IVF-BF16-nl273-np16 (query)                              368.00       110.87       478.86       0.9377     0.000244         9.48
IVF-BF16-nl273-np23 (query)                              368.00       150.41       518.41       0.9383     0.000243         9.48
IVF-BF16-nl273 (self)                                    368.00     1_524.65     1_892.64       0.9371     0.001235         9.48
IVF-BF16-nl387-np19 (query)                              695.71        92.67       788.38       0.9369     0.000248         9.49
IVF-BF16-nl387-np27 (query)                              695.71       125.52       821.23       0.9383     0.000243         9.49
IVF-BF16-nl387 (self)                                    695.71     1_270.92     1_966.63       0.9371     0.001235         9.49
IVF-BF16-nl547-np23 (query)                            1_350.81        86.18     1_436.98       0.9358     0.000251         9.51
IVF-BF16-nl547-np27 (query)                            1_350.81       100.12     1_450.92       0.9379     0.000244         9.51
IVF-BF16-nl547-np33 (query)                            1_350.81       115.26     1_466.07       0.9382     0.000244         9.51
IVF-BF16-nl547 (self)                                  1_350.81     1_168.11     2_518.91       0.9370     0.001236         9.51
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.35     1_553.93     1_557.28       1.0000     0.000000        18.31
Exhaustive (self)                                          3.35    15_353.29    15_356.64       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                                    5.15     1_167.21     1_172.36       0.9635     0.020884         9.16
Exhaustive-BF16 (self)                                     5.15    15_456.47    15_461.62       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                              384.49        91.36       475.85       0.9635     0.020890         9.19
IVF-BF16-nl273-np16 (query)                              384.49       107.55       492.03       0.9635     0.020884         9.19
IVF-BF16-nl273-np23 (query)                              384.49       140.26       524.75       0.9635     0.020884         9.19
IVF-BF16-nl273 (self)                                    384.49     1_414.34     1_798.83       0.9543     0.028602         9.19
IVF-BF16-nl387-np19 (query)                              727.40        92.15       819.55       0.9635     0.020884         9.21
IVF-BF16-nl387-np27 (query)                              727.40       124.03       851.43       0.9635     0.020884         9.21
IVF-BF16-nl387 (self)                                    727.40     1_252.51     1_979.91       0.9543     0.028602         9.21
IVF-BF16-nl547-np23 (query)                            1_435.07        80.85     1_515.92       0.9635     0.020884         9.23
IVF-BF16-nl547-np27 (query)                            1_435.07        92.88     1_527.95       0.9635     0.020884         9.23
IVF-BF16-nl547-np33 (query)                            1_435.07       109.72     1_544.79       0.9635     0.020884         9.23
IVF-BF16-nl547 (self)                                  1_435.07     1_107.77     2_542.83       0.9543     0.028602         9.23
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.11     1_510.43     1_513.54       1.0000     0.000000        18.31
Exhaustive (self)                                          3.11    15_309.40    15_312.50       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                                    4.72     1_167.33     1_172.04       0.9421     0.154192         9.16
Exhaustive-BF16 (self)                                     4.72    15_487.41    15_492.12       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                              385.63        89.76       475.39       0.9421     0.154190         9.19
IVF-BF16-nl273-np16 (query)                              385.63       102.70       488.33       0.9421     0.154192         9.19
IVF-BF16-nl273-np23 (query)                              385.63       141.70       527.32       0.9421     0.154192         9.19
IVF-BF16-nl273 (self)                                    385.63     1_417.04     1_802.67       0.9258     0.215906         9.19
IVF-BF16-nl387-np19 (query)                              732.32        89.33       821.65       0.9421     0.154192         9.21
IVF-BF16-nl387-np27 (query)                              732.32       118.11       850.43       0.9421     0.154192         9.21
IVF-BF16-nl387 (self)                                    732.32     1_221.62     1_953.94       0.9258     0.215906         9.21
IVF-BF16-nl547-np23 (query)                            1_422.69        82.50     1_505.19       0.9421     0.154192         9.23
IVF-BF16-nl547-np27 (query)                            1_422.69        93.08     1_515.78       0.9421     0.154192         9.23
IVF-BF16-nl547-np33 (query)                            1_422.69       110.38     1_533.07       0.9421     0.154192         9.23
IVF-BF16-nl547 (self)                                  1_422.69     1_114.56     2_537.25       0.9258     0.215906         9.23
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.10     6_040.82     6_055.92       1.0000     0.000000        73.24
Exhaustive (self)                                         15.10    60_448.32    60_463.41       1.0000     0.000000        73.24
Exhaustive-BF16 (query)                                   23.24     5_412.79     5_436.02       0.9503     1.781717        36.62
Exhaustive-BF16 (self)                                    23.24    63_231.58    63_254.82       1.0000     0.000000        36.62
IVF-BF16-nl273-np13 (query)                              462.42       292.72       755.14       0.9501     1.785588        36.76
IVF-BF16-nl273-np16 (query)                              462.42       354.72       817.14       0.9503     1.781861        36.76
IVF-BF16-nl273-np23 (query)                              462.42       493.46       955.88       0.9503     1.781717        36.76
IVF-BF16-nl273 (self)                                    462.42     5_084.93     5_547.36       0.9374     2.611285        36.76
IVF-BF16-nl387-np19 (query)                              826.14       301.12     1_127.26       0.9503     1.781717        36.81
IVF-BF16-nl387-np27 (query)                              826.14       412.59     1_238.73       0.9503     1.781717        36.81
IVF-BF16-nl387 (self)                                    826.14     4_225.99     5_052.13       0.9374     2.611285        36.81
IVF-BF16-nl547-np23 (query)                            1_762.04       266.14     2_028.18       0.9502     1.783295        36.89
IVF-BF16-nl547-np27 (query)                            1_762.04       310.80     2_072.84       0.9503     1.781855        36.89
IVF-BF16-nl547-np33 (query)                            1_762.04       374.99     2_137.04       0.9503     1.781717        36.89
IVF-BF16-nl547 (self)                                  1_762.04     3_788.31     5_550.35       0.9374     2.611285        36.89
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
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.05     1_536.21     1_539.26       1.0000     0.000000        18.31
Exhaustive (self)                                          3.05    15_295.46    15_298.51       1.0000     0.000000        18.31
Exhaustive-SQ8 (query)                                     7.41       717.79       725.20       0.8337          NaN         4.58
Exhaustive-SQ8 (self)                                      7.41     7_573.61     7_581.02       0.8340          NaN         4.58
IVF-SQ8-nl273-np13 (query)                               382.55        51.43       433.98       0.8305          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               382.55        59.30       441.85       0.8325          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               382.55        79.01       461.56       0.8330          NaN         4.61
IVF-SQ8-nl273 (self)                                     382.55       797.19     1_179.74       0.8328          NaN         4.61
IVF-SQ8-nl387-np19 (query)                               738.42        50.51       788.94       0.8401          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               738.42        67.70       806.12       0.8411          NaN         4.63
IVF-SQ8-nl387 (self)                                     738.42       669.42     1_407.84       0.8411          NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_441.64        49.09     1_490.73       0.8346          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_441.64        54.48     1_496.13       0.8359          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_441.64        62.64     1_504.28       0.8361          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_441.64       623.99     2_065.64       0.8357          NaN         4.65
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.05     1_551.78     1_555.83       1.0000     0.000000        18.88
Exhaustive (self)                                          4.05    15_281.15    15_285.20       1.0000     0.000000        18.88
Exhaustive-SQ8 (query)                                     7.36       691.31       698.67       0.8537          NaN         5.15
Exhaustive-SQ8 (self)                                      7.36     7_244.97     7_252.33       0.8536          NaN         5.15
IVF-SQ8-nl273-np13 (query)                               369.49        49.89       419.38       0.8371          NaN         5.19
IVF-SQ8-nl273-np16 (query)                               369.49        58.18       427.66       0.8389          NaN         5.19
IVF-SQ8-nl273-np23 (query)                               369.49        77.32       446.81       0.8392          NaN         5.19
IVF-SQ8-nl273 (self)                                     369.49       772.11     1_141.59       0.8384          NaN         5.19
IVF-SQ8-nl387-np19 (query)                               704.61        50.30       754.90       0.8490          NaN         5.20
IVF-SQ8-nl387-np27 (query)                               704.61        67.72       772.32       0.8499          NaN         5.20
IVF-SQ8-nl387 (self)                                     704.61       662.36     1_366.96       0.8493          NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_363.77        48.51     1_412.28       0.8478          NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_363.77        53.18     1_416.95       0.8491          NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_363.77        62.85     1_426.62       0.8494          NaN         5.22
IVF-SQ8-nl547 (self)                                   1_363.77       616.40     1_980.17       0.8493          NaN         5.22
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.10     1_537.48     1_540.58       1.0000     0.000000        18.31
Exhaustive (self)                                          3.10    15_144.60    15_147.70       1.0000     0.000000        18.31
Exhaustive-SQ8 (query)                                     7.23       713.09       720.32       0.7103          NaN         4.58
Exhaustive-SQ8 (self)                                      7.23     7_556.81     7_564.04       0.7118          NaN         4.58
IVF-SQ8-nl273-np13 (query)                               385.29        52.12       437.40       0.7139          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               385.29        63.70       448.99       0.7139          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               385.29        79.96       465.24       0.7138          NaN         4.61
IVF-SQ8-nl273 (self)                                     385.29       788.65     1_173.93       0.7147          NaN         4.61
IVF-SQ8-nl387-np19 (query)                               731.32        53.74       785.06       0.7150          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               731.32        70.53       801.85       0.7150          NaN         4.63
IVF-SQ8-nl387 (self)                                     731.32       697.67     1_428.99       0.7156          NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_424.11        48.09     1_472.20       0.7108          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_424.11        54.66     1_478.77       0.7108          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_424.11        64.96     1_489.08       0.7108          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_424.11       636.46     2_060.57       0.7122          NaN         4.65
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.17     1_511.10     1_514.27       1.0000     0.000000        18.31
Exhaustive (self)                                          3.17    14_984.43    14_987.60       1.0000     0.000000        18.31
Exhaustive-SQ8 (query)                                     7.09       713.48       720.58       0.4744          NaN         4.58
Exhaustive-SQ8 (self)                                      7.09     7_582.58     7_589.68       0.4837          NaN         4.58
IVF-SQ8-nl273-np13 (query)                               383.68        50.22       433.90       0.4753          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               383.68        59.52       443.20       0.4752          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               383.68        80.40       464.08       0.4751          NaN         4.61
IVF-SQ8-nl273 (self)                                     383.68       784.70     1_168.39       0.4846          NaN         4.61
IVF-SQ8-nl387-np19 (query)                               735.03        52.76       787.80       0.4745          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               735.03        67.48       802.51       0.4744          NaN         4.63
IVF-SQ8-nl387 (self)                                     735.03       665.33     1_400.36       0.4843          NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_423.63        48.45     1_472.08       0.4745          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_423.63        58.29     1_481.92       0.4745          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_423.63        63.81     1_487.44       0.4744          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_423.63       632.88     2_056.51       0.4837          NaN         4.65
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
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.64     6_052.71     6_067.35       1.0000     0.000000        73.24
Exhaustive (self)                                         14.64    60_480.61    60_495.25       1.0000     0.000000        73.24
Exhaustive-SQ8 (query)                                    39.36     1_729.94     1_769.29       0.7135          NaN        18.31
Exhaustive-SQ8 (self)                                     39.36    17_113.37    17_152.73       0.7165          NaN        18.31
IVF-SQ8-nl273-np13 (query)                               464.38       102.90       567.27       0.7137          NaN        18.45
IVF-SQ8-nl273-np16 (query)                               464.38       120.47       584.84       0.7137          NaN        18.45
IVF-SQ8-nl273-np23 (query)                               464.38       162.98       627.36       0.7137          NaN        18.45
IVF-SQ8-nl273 (self)                                     464.38     1_560.02     2_024.40       0.7164          NaN        18.45
IVF-SQ8-nl387-np19 (query)                               812.53       107.85       920.38       0.7140          NaN        18.50
IVF-SQ8-nl387-np27 (query)                               812.53       143.74       956.27       0.7140          NaN        18.50
IVF-SQ8-nl387 (self)                                     812.53     1_341.64     2_154.17       0.7166          NaN        18.50
IVF-SQ8-nl547-np23 (query)                             1_771.27       101.25     1_872.53       0.7141          NaN        18.58
IVF-SQ8-nl547-np27 (query)                             1_771.27       114.49     1_885.76       0.7141          NaN        18.58
IVF-SQ8-nl547-np33 (query)                             1_771.27       134.92     1_906.19       0.7141          NaN        18.58
IVF-SQ8-nl547 (self)                                   1_771.27     1_257.73     3_029.00       0.7166          NaN        18.58
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

#### Correlated data

<details>
<summary><b>Correlated data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         5.04     1_828.05     1_833.09       1.0000     0.000000        24.41
Exhaustive (self)                                          5.04     5_878.11     5_883.14       1.0000     0.000000        24.41
Exhaustive-PQ-m8 (query)                                 564.67       314.81       879.48       0.2776          NaN         0.51
Exhaustive-PQ-m8 (self)                                  564.67     1_026.25     1_590.93       0.2082          NaN         0.51
Exhaustive-PQ-m16 (query)                                619.10       648.68     1_267.78       0.3909          NaN         0.89
Exhaustive-PQ-m16 (self)                                 619.10     2_167.48     2_786.59       0.3043          NaN         0.89
IVF-PQ-nl158-m8-np7 (query)                            1_279.00       147.07     1_426.07       0.4485          NaN         0.59
IVF-PQ-nl158-m8-np12 (query)                           1_279.00       246.57     1_525.57       0.4497          NaN         0.59
IVF-PQ-nl158-m8-np17 (query)                           1_279.00       340.33     1_619.33       0.4497          NaN         0.59
IVF-PQ-nl158-m8 (self)                                 1_279.00     1_155.94     2_434.94       0.3500          NaN         0.59
IVF-PQ-nl158-m16-np7 (query)                           1_337.47       234.98     1_572.44       0.6076          NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_337.47       388.35     1_725.82       0.6106          NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_337.47       529.21     1_866.68       0.6109          NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_337.47     1_771.29     3_108.76       0.5250          NaN         0.97
IVF-PQ-nl223-m8-np11 (query)                             835.09       219.24     1_054.33       0.4535          NaN         0.62
IVF-PQ-nl223-m8-np14 (query)                             835.09       272.81     1_107.90       0.4535          NaN         0.62
IVF-PQ-nl223-m8-np21 (query)                             835.09       404.88     1_239.97       0.4535          NaN         0.62
IVF-PQ-nl223-m8 (self)                                   835.09     1_335.11     2_170.20       0.3553          NaN         0.62
IVF-PQ-nl223-m16-np11 (query)                            899.72       334.61     1_234.33       0.6138          NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            899.72       426.52     1_326.24       0.6141          NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            899.72       634.38     1_534.09       0.6141          NaN         1.00
IVF-PQ-nl223-m16 (self)                                  899.72     2_083.86     2_983.58       0.5296          NaN         1.00
IVF-PQ-nl316-m8-np15 (query)                             978.06       275.20     1_253.26       0.4583          NaN         0.66
IVF-PQ-nl316-m8-np17 (query)                             978.06       313.07     1_291.13       0.4582          NaN         0.66
IVF-PQ-nl316-m8-np25 (query)                             978.06       449.81     1_427.87       0.4583          NaN         0.66
IVF-PQ-nl316-m8 (self)                                   978.06     1_506.81     2_484.87       0.3597          NaN         0.66
IVF-PQ-nl316-m16-np15 (query)                          1_032.22       419.69     1_451.91       0.6158          NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_032.22       486.31     1_518.53       0.6159          NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_032.22       698.48     1_730.70       0.6160          NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_032.22     2_371.71     3_403.93       0.5325          NaN         1.05
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.08     4_333.73     4_343.81       1.0000     0.000000        48.83
Exhaustive (self)                                         10.08    14_491.85    14_501.93       1.0000     0.000000        48.83
Exhaustive-PQ-m16 (query)                              1_634.33       676.56     2_310.89       0.2925          NaN         1.01
Exhaustive-PQ-m16 (self)                               1_634.33     2_240.81     3_875.14       0.2190          NaN         1.01
Exhaustive-PQ-m32 (query)                              1_305.74     1_511.49     2_817.23       0.4037          NaN         1.78
Exhaustive-PQ-m32 (self)                               1_305.74     5_091.01     6_396.75       0.3186          NaN         1.78
Exhaustive-PQ-m64 (query)                              2_074.53     4_031.47     6_106.00       0.5129          NaN         3.30
Exhaustive-PQ-m64 (self)                               2_074.53    13_565.89    15_640.42       0.4362          NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_853.29       294.64     3_147.93       0.4543          NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_853.29       488.69     3_341.98       0.4578          NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_853.29       679.04     3_532.33       0.4580          NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_853.29     2_259.54     5_112.82       0.3635          NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_652.88       437.66     3_090.54       0.6099          NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_652.88       729.83     3_382.71       0.6160          NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_652.88     1_016.13     3_669.01       0.6164          NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_652.88     3_347.50     6_000.38       0.5368          NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_446.33       785.50     4_231.83       0.7413          NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_446.33     1_308.88     4_755.21       0.7505          NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_446.33     1_886.51     5_332.84       0.7511          NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_446.33     6_173.18     9_619.51       0.6977          NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_754.42       408.70     2_163.11       0.4614          NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_754.42       522.56     2_276.97       0.4616          NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_754.42       780.78     2_535.20       0.4616          NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_754.42     2_587.04     4_341.45       0.3665          NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_805.27       636.96     2_442.23       0.6184          NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_805.27       821.02     2_626.29       0.6189          NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_805.27     1_189.23     2_994.50       0.6189          NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_805.27     3_978.76     5_784.03       0.5396          NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_580.25     1_105.19     3_685.44       0.7529          NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_580.25     1_387.62     3_967.88       0.7538          NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_580.25     2_088.95     4_669.20       0.7538          NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_580.25     6_940.67     9_520.92       0.6994          NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_983.71       563.97     2_547.67       0.4622          NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_983.71       627.27     2_610.97       0.4624          NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_983.71       908.33     2_892.03       0.4625          NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_983.71     2_997.28     4_980.98       0.3678          NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_981.44       826.59     2_808.03       0.6196          NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_981.44       936.95     2_918.39       0.6202          NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_981.44     1_358.08     3_339.52       0.6204          NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_981.44     4_522.84     6_504.28       0.5401          NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_772.52     1_440.32     4_212.84       0.7529          NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_772.52     1_623.33     4_395.85       0.7541          NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_772.52     2_361.15     5_133.68       0.7547          NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_772.52     7_873.88    10_646.40       0.7015          NaN         3.61
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.41     9_622.50     9_642.90       1.0000     0.000000        97.66
Exhaustive (self)                                         20.41    32_794.33    32_814.74       1.0000     0.000000        97.66
Exhaustive-PQ-m16 (query)                              1_182.28       675.70     1_857.98       0.2177          NaN         1.26
Exhaustive-PQ-m16 (self)                               1_182.28     2_244.53     3_426.81       0.1641          NaN         1.26
Exhaustive-PQ-m32 (query)                              2_294.94     1_494.93     3_789.87       0.2946          NaN         2.03
Exhaustive-PQ-m32 (self)                               2_294.94     4_973.98     7_268.93       0.2209          NaN         2.03
Exhaustive-PQ-m64 (query)                              2_515.30     3_947.11     6_462.41       0.4093          NaN         3.55
Exhaustive-PQ-m64 (self)                               2_515.30    13_152.57    15_667.87       0.3273          NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_939.66       399.78     4_339.44       0.3232          NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_939.66       652.71     4_592.38       0.3251          NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_939.66       907.08     4_846.74       0.3252          NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_939.66     2_982.88     6_922.54       0.2396          NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_987.41       548.63     5_536.04       0.4589          NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_987.41       905.99     5_893.41       0.4632          NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_987.41     1_259.75     6_247.16       0.4635          NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_987.41     4_188.20     9_175.61       0.3726          NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_189.80       886.73     6_076.52       0.6098          NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_189.80     1_440.62     6_630.42       0.6178          NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_189.80     1_993.45     7_183.25       0.6183          NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_189.80     6_624.19    11_813.99       0.5438          NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_188.04       562.86     2_750.90       0.3277          NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_188.04       701.73     2_889.77       0.3278          NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_188.04     1_036.86     3_224.90       0.3278          NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_188.04     3_397.49     5_585.53       0.2416          NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_306.87       786.37     4_093.25       0.4635          NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_306.87       986.60     4_293.48       0.4636          NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_306.87     1_462.01     4_768.89       0.4636          NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_306.87     4_856.36     8_163.23       0.3739          NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_522.45     1_250.18     4_772.63       0.6206          NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_522.45     1_579.20     5_101.65       0.6208          NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_522.45     2_451.88     5_974.33       0.6208          NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_522.45     7_807.56    11_330.01       0.5448          NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_614.02       745.33     3_359.35       0.3298          NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_614.02       840.72     3_454.74       0.3298          NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_614.02     1_213.91     3_827.93       0.3298          NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_614.02     4_017.18     6_631.20       0.2429          NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_722.58     1_047.48     4_770.07       0.4668          NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_722.58     1_176.96     4_899.55       0.4669          NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_722.58     1_703.05     5_425.63       0.4671          NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_722.58     5_587.10     9_309.68       0.3772          NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_957.28     1_623.69     5_580.97       0.6209          NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_957.28     1_829.76     5_787.04       0.6212          NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_957.28     2_661.46     6_618.75       0.6213          NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_957.28     8_866.82    12_824.10       0.5467          NaN         4.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.64     1_748.39     1_753.03       1.0000     0.000000        24.41
Exhaustive (self)                                          4.64     5_981.91     5_986.54       1.0000     0.000000        24.41
Exhaustive-PQ-m8 (query)                                 790.01       315.21     1_105.22       0.2342          NaN         0.51
Exhaustive-PQ-m8 (self)                                  790.01     1_042.46     1_832.47       0.1839          NaN         0.51
Exhaustive-PQ-m16 (query)                                651.52       663.16     1_314.68       0.3314          NaN         0.89
Exhaustive-PQ-m16 (self)                                 651.52     2_200.05     2_851.56       0.2480          NaN         0.89
IVF-PQ-nl158-m8-np7 (query)                            1_392.68       145.44     1_538.13       0.4556          NaN         0.59
IVF-PQ-nl158-m8-np12 (query)                           1_392.68       252.59     1_645.27       0.4557          NaN         0.59
IVF-PQ-nl158-m8-np17 (query)                           1_392.68       342.20     1_734.88       0.4557          NaN         0.59
IVF-PQ-nl158-m8 (self)                                 1_392.68     1_138.30     2_530.98       0.3381          NaN         0.59
IVF-PQ-nl158-m16-np7 (query)                           1_407.44       246.32     1_653.75       0.6065          NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_407.44       411.45     1_818.88       0.6068          NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_407.44       576.50     1_983.94       0.6068          NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_407.44     1_917.68     3_325.11       0.5146          NaN         0.97
IVF-PQ-nl223-m8-np11 (query)                           1_078.55       233.08     1_311.63       0.4613          NaN         0.62
IVF-PQ-nl223-m8-np14 (query)                           1_078.55       297.87     1_376.42       0.4613          NaN         0.62
IVF-PQ-nl223-m8-np21 (query)                           1_078.55       445.15     1_523.71       0.4613          NaN         0.62
IVF-PQ-nl223-m8 (self)                                 1_078.55     1_404.80     2_483.36       0.3412          NaN         0.62
IVF-PQ-nl223-m16-np11 (query)                            931.56       359.98     1_291.54       0.6107          NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            931.56       441.85     1_373.42       0.6108          NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            931.56       664.50     1_596.06       0.6108          NaN         1.00
IVF-PQ-nl223-m16 (self)                                  931.56     2_188.84     3_120.40       0.5169          NaN         1.00
IVF-PQ-nl316-m8-np15 (query)                           1_347.05       287.34     1_634.39       0.4642          NaN         0.66
IVF-PQ-nl316-m8-np17 (query)                           1_347.05       305.69     1_652.73       0.4643          NaN         0.66
IVF-PQ-nl316-m8-np25 (query)                           1_347.05       443.27     1_790.32       0.4643          NaN         0.66
IVF-PQ-nl316-m8 (self)                                 1_347.05     1_503.36     2_850.41       0.3418          NaN         0.66
IVF-PQ-nl316-m16-np15 (query)                          1_102.72       451.13     1_553.85       0.6158          NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_102.72       513.23     1_615.95       0.6158          NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_102.72       740.86     1_843.58       0.6158          NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_102.72     2_538.30     3_641.02       0.5197          NaN         1.05
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.29     4_365.51     4_375.80       1.0000     0.000000        48.83
Exhaustive (self)                                         10.29    14_353.13    14_363.43       1.0000     0.000000        48.83
Exhaustive-PQ-m16 (query)                              1_321.97       678.45     2_000.42       0.2302          NaN         1.01
Exhaustive-PQ-m16 (self)                               1_321.97     2_238.25     3_560.22       0.1707          NaN         1.01
Exhaustive-PQ-m32 (query)                              1_312.28     1_514.26     2_826.54       0.3162          NaN         1.78
Exhaustive-PQ-m32 (self)                               1_312.28     5_030.69     6_342.97       0.2268          NaN         1.78
Exhaustive-PQ-m64 (query)                              2_129.82     4_023.36     6_153.18       0.4261          NaN         3.30
Exhaustive-PQ-m64 (self)                               2_129.82    13_474.88    15_604.71       0.3314          NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_674.21       286.26     2_960.47       0.4118          NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_674.21       471.74     3_145.96       0.4121          NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_674.21       669.01     3_343.22       0.4121          NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_674.21     2_168.14     4_842.36       0.2984          NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_727.08       442.48     3_169.56       0.5623          NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_727.08       738.06     3_465.14       0.5630          NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_727.08     1_037.12     3_764.20       0.5630          NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_727.08     3_406.60     6_133.68       0.4803          NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_487.11       764.19     4_251.30       0.7827          NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_487.11     1_311.37     4_798.48       0.7841          NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_487.11     1_852.15     5_339.27       0.7841          NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_487.11     6_174.14     9_661.26       0.7386          NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_784.39       422.13     2_206.52       0.4147          NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_784.39       530.38     2_314.77       0.4148          NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_784.39       782.54     2_566.93       0.4148          NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_784.39     2_573.12     4_357.50       0.2981          NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_849.94       652.76     2_502.71       0.5652          NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_849.94       819.52     2_669.46       0.5654          NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_849.94     1_243.74     3_093.68       0.5654          NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_849.94     3_946.66     5_796.61       0.4819          NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_612.75     1_102.63     3_715.39       0.7861          NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_612.75     1_453.81     4_066.56       0.7864          NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_612.75     2_056.31     4_669.06       0.7864          NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_612.75     6_865.82     9_478.57       0.7403          NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_031.44       577.76     2_609.19       0.4159          NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_031.44       654.50     2_685.93       0.4159          NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_031.44       926.06     2_957.49       0.4159          NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_031.44     2_989.14     5_020.58       0.2969          NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_043.00       829.22     2_872.22       0.5682          NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_043.00       936.60     2_979.61       0.5683          NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_043.00     1_353.27     3_396.27       0.5683          NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_043.00     4_483.38     6_526.38       0.4813          NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_913.93     1_434.70     4_348.63       0.7883          NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_913.93     1_607.83     4_521.76       0.7886          NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_913.93     2_340.44     5_254.37       0.7886          NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_913.93     7_769.55    10_683.48       0.7427          NaN         3.61
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.90     9_646.68     9_666.58       1.0000     0.000000        97.66
Exhaustive (self)                                         19.90    32_432.58    32_452.49       1.0000     0.000000        97.66
Exhaustive-PQ-m16 (query)                              1_178.36       673.15     1_851.51       0.1684          NaN         1.26
Exhaustive-PQ-m16 (self)                               1_178.36     2_244.90     3_423.26       0.1309          NaN         1.26
Exhaustive-PQ-m32 (query)                              2_309.69     1_493.06     3_802.75       0.2243          NaN         2.03
Exhaustive-PQ-m32 (self)                               2_309.69     4_967.62     7_277.31       0.1607          NaN         2.03
Exhaustive-PQ-m64 (query)                              2_517.75     3_973.08     6_490.83       0.3028          NaN         3.55
Exhaustive-PQ-m64 (self)                               2_517.75    13_159.05    15_676.80       0.2122          NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           4_049.88       389.38     4_439.26       0.2845          NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          4_049.88       649.06     4_698.94       0.2847          NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          4_049.88       903.00     4_952.88       0.2847          NaN         1.57
IVF-PQ-nl158-m16 (self)                                4_049.88     3_101.40     7_151.28       0.1816          NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_699.66       542.80     6_242.46       0.3771          NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_699.66       915.62     6_615.28       0.3779          NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_699.66     1_299.31     6_998.97       0.3779          NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_699.66     4_286.77     9_986.43       0.2659          NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_538.73       908.64     6_447.37       0.5291          NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_538.73     1_496.00     7_034.73       0.5310          NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_538.73     2_088.28     7_627.01       0.5310          NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_538.73     6_996.23    12_534.97       0.4586          NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_660.63       585.95     3_246.58       0.2858          NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_660.63       727.26     3_387.89       0.2859          NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_660.63     1_075.78     3_736.41       0.2859          NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_660.63     3_513.49     6_174.12       0.1814          NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_736.29       787.66     4_523.96       0.3813          NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_736.29     1_009.02     4_745.32       0.3813          NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_736.29     1_488.45     5_224.75       0.3813          NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_736.29     4_968.76     8_705.05       0.2651          NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_740.54     1_260.46     5_001.00       0.5317          NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_740.54     1_601.67     5_342.21       0.5319          NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_740.54     2_412.28     6_152.83       0.5319          NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_740.54     7_950.43    11_690.97       0.4588          NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_923.99       770.88     3_694.87       0.2844          NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_923.99       855.41     3_779.40       0.2844          NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_923.99     1_230.43     4_154.42       0.2844          NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_923.99     4_005.31     6_929.29       0.1787          NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_138.61     1_050.28     5_188.89       0.3797          NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_138.61     1_186.65     5_325.26       0.3797          NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_138.61     1_710.33     5_848.94       0.3797          NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_138.61     5_651.58     9_790.19       0.2603          NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          4_152.06     1_649.41     5_801.47       0.5348          NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          4_152.06     1_848.54     6_000.60       0.5350          NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          4_152.06     2_692.84     6_844.90       0.5350          NaN         4.17
IVF-PQ-nl316-m64 (self)                                4_152.06     9_030.27    13_182.33       0.4587          NaN         4.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.06     1_783.36     1_786.41       1.0000     0.000000        24.41
Exhaustive (self)                                          3.06     6_128.13     6_131.18       1.0000     0.000000        24.41
Exhaustive-PQ-m8 (query)                                 603.68       313.76       917.44       0.1240          NaN         0.51
Exhaustive-PQ-m8 (self)                                  603.68     1_052.70     1_656.38       0.2528          NaN         0.51
Exhaustive-PQ-m16 (query)                                670.12       675.74     1_345.87       0.1806          NaN         0.89
Exhaustive-PQ-m16 (self)                                 670.12     2_214.66     2_884.78       0.3284          NaN         0.89
IVF-PQ-nl158-m8-np7 (query)                            1_281.55       198.72     1_480.27       0.3196          NaN         0.59
IVF-PQ-nl158-m8-np12 (query)                           1_281.55       305.75     1_587.29       0.3197          NaN         0.59
IVF-PQ-nl158-m8-np17 (query)                           1_281.55       434.75     1_716.30       0.3197          NaN         0.59
IVF-PQ-nl158-m8 (self)                                 1_281.55     1_418.97     2_700.52       0.4617          NaN         0.59
IVF-PQ-nl158-m16-np7 (query)                           1_294.72       299.94     1_594.66       0.4309          NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_294.72       506.59     1_801.31       0.4309          NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_294.72       675.39     1_970.11       0.4310          NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_294.72     2_318.71     3_613.43       0.5727          NaN         0.97
IVF-PQ-nl223-m8-np11 (query)                           1_033.37       235.22     1_268.60       0.3666          NaN         0.62
IVF-PQ-nl223-m8-np14 (query)                           1_033.37       321.61     1_354.98       0.3666          NaN         0.62
IVF-PQ-nl223-m8-np21 (query)                           1_033.37       426.34     1_459.71       0.3666          NaN         0.62
IVF-PQ-nl223-m8 (self)                                 1_033.37     1_427.21     2_460.58       0.5113          NaN         0.62
IVF-PQ-nl223-m16-np11 (query)                            784.38       353.74     1_138.12       0.4801          NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            784.38       456.67     1_241.05       0.4801          NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            784.38       677.16     1_461.54       0.4802          NaN         1.00
IVF-PQ-nl223-m16 (self)                                  784.38     2_257.80     3_042.18       0.6156          NaN         1.00
IVF-PQ-nl316-m8-np15 (query)                             869.88       314.18     1_184.06       0.3764          NaN         0.66
IVF-PQ-nl316-m8-np17 (query)                             869.88       349.98     1_219.86       0.3764          NaN         0.66
IVF-PQ-nl316-m8-np25 (query)                             869.88       484.47     1_354.36       0.3764          NaN         0.66
IVF-PQ-nl316-m8 (self)                                   869.88     1_632.01     2_501.89       0.5170          NaN         0.66
IVF-PQ-nl316-m16-np15 (query)                            748.41       451.86     1_200.27       0.4907          NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                            748.41       513.07     1_261.48       0.4907          NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                            748.41       733.45     1_481.86       0.4907          NaN         1.05
IVF-PQ-nl316-m16 (self)                                  748.41     2_425.55     3_173.95       0.6197          NaN         1.05
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.80     4_284.70     4_291.51       1.0000     0.000000        48.83
Exhaustive (self)                                          6.80    14_425.81    14_432.61       1.0000     0.000000        48.83
Exhaustive-PQ-m16 (query)                              1_385.08       681.21     2_066.29       0.1170          NaN         1.01
Exhaustive-PQ-m16 (self)                               1_385.08     2_235.19     3_620.26       0.3086          NaN         1.01
Exhaustive-PQ-m32 (query)                              1_291.92     1_507.17     2_799.09       0.1723          NaN         1.78
Exhaustive-PQ-m32 (self)                               1_291.92     5_075.87     6_367.78       0.4012          NaN         1.78
Exhaustive-PQ-m64 (query)                              2_196.97     4_012.98     6_209.95       0.2817          NaN         3.30
Exhaustive-PQ-m64 (self)                               2_196.97    13_413.27    15_610.25       0.5337          NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_502.11       308.30     2_810.41       0.3362          NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_502.11       518.77     3_020.88       0.3363          NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_502.11       718.58     3_220.70       0.3363          NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_502.11     2_331.65     4_833.76       0.5860          NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_477.09       507.53     2_984.62       0.4530          NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_477.09       846.62     3_323.70       0.4534          NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_477.09     1_195.96     3_673.04       0.4534          NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_477.09     3_950.20     6_427.28       0.6918          NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_244.20       935.64     4_179.84       0.6565          NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_244.20     1_612.55     4_856.75       0.6573          NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_244.20     2_283.87     5_528.06       0.6574          NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_244.20     7_626.29    10_870.49       0.8277          NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_290.75       407.47     1_698.21       0.3576          NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_290.75       509.11     1_799.86       0.3577          NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_290.75       753.32     2_044.07       0.3578          NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_290.75     2_498.74     3_789.49       0.6017          NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_467.27       657.31     2_124.58       0.4765          NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_467.27       830.36     2_297.63       0.4767          NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_467.27     1_228.84     2_696.11       0.4768          NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_467.27     4_023.01     5_490.27       0.7014          NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_314.57     1_102.06     3_416.63       0.6772          NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_314.57     1_392.02     3_706.58       0.6775          NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_314.57     2_111.51     4_426.08       0.6777          NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_314.57     6_949.79     9_264.36       0.8351          NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_622.83       545.58     2_168.41       0.3667          NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_622.83       613.90     2_236.73       0.3667          NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_622.83       880.96     2_503.79       0.3668          NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_622.83     2_927.11     4_549.94       0.6029          NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_479.45       816.72     2_296.18       0.4863          NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_479.45       913.11     2_392.57       0.4864          NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_479.45     1_335.67     2_815.13       0.4865          NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_479.45     4_552.28     6_031.73       0.7000          NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_357.82     1_409.08     3_766.90       0.6836          NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_357.82     1_599.08     3_956.90       0.6837          NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_357.82     2_328.63     4_686.44       0.6839          NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_357.82     7_748.51    10_106.32       0.8354          NaN         3.61
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.18     9_912.07     9_932.25       1.0000     0.000000        97.66
Exhaustive (self)                                         20.18    34_513.26    34_533.44       1.0000     0.000000        97.66
Exhaustive-PQ-m16 (query)                              1_408.77       697.35     2_106.12       0.0871          NaN         1.26
Exhaustive-PQ-m16 (self)                               1_408.77     2_303.52     3_712.30       0.2928          NaN         1.26
Exhaustive-PQ-m32 (query)                              2_951.90     1_534.75     4_486.65       0.1140          NaN         2.03
Exhaustive-PQ-m32 (self)                               2_951.90     5_115.61     8_067.52       0.3714          NaN         2.03
Exhaustive-PQ-m64 (query)                              2_580.96     4_143.72     6_724.68       0.1674          NaN         3.55
Exhaustive-PQ-m64 (self)                               2_580.96    13_460.78    16_041.74       0.4781          NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_628.74       422.39     4_051.13       0.2565          NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_628.74       715.94     4_344.68       0.2566          NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_628.74       996.61     4_625.35       0.2566          NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_628.74     3_273.41     6_902.15       0.5497          NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_892.88       608.63     5_501.51       0.3199          NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_892.88     1_024.71     5_917.59       0.3202          NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_892.88     1_478.72     6_371.59       0.3202          NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_892.88     4_786.49     9_679.37       0.6347          NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_873.03       983.91     5_856.93       0.4381          NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_873.03     1_698.94     6_571.96       0.4387          NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_873.03     2_458.26     7_331.28       0.4387          NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_873.03     8_017.81    12_890.83       0.7341          NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_520.31       582.50     2_102.81       0.2752          NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_520.31       737.85     2_258.16       0.2753          NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_520.31     1_081.62     2_601.93       0.2754          NaN         1.70
IVF-PQ-nl223-m16 (self)                                1_520.31     3_547.85     5_068.15       0.5672          NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_751.95       792.22     3_544.17       0.3374          NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_751.95       997.08     3_749.03       0.3375          NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_751.95     1_481.58     4_233.53       0.3376          NaN         2.46
IVF-PQ-nl223-m32 (self)                                2_751.95     4_885.60     7_637.56       0.6430          NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_844.55     1_271.54     4_116.09       0.4532          NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_844.55     1_608.20     4_452.75       0.4536          NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_844.55     2_385.65     5_230.19       0.4538          NaN         3.99
IVF-PQ-nl223-m64 (self)                                2_844.55     7_981.72    10_826.27       0.7363          NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_465.16       785.41     2_250.57       0.2874          NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_465.16       866.88     2_332.04       0.2874          NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_465.16     1_250.53     2_715.69       0.2874          NaN         1.88
IVF-PQ-nl316-m16 (self)                                1_465.16     4_102.97     5_568.12       0.5775          NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_790.82     1_059.26     3_850.08       0.3516          NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_790.82     1_181.94     3_972.77       0.3516          NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_790.82     1_710.72     4_501.54       0.3516          NaN         2.65
IVF-PQ-nl316-m32 (self)                                2_790.82     5_877.29     8_668.11       0.6509          NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_759.44     1_643.89     4_403.32       0.4730          NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_759.44     1_859.88     4_619.32       0.4732          NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_759.44     2_709.45     5_468.88       0.4733          NaN         4.17
IVF-PQ-nl316-m64 (self)                                2_759.44     9_047.11    11_806.55       0.7432          NaN         4.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Especially for the data with more internal structure, we can appreciate that
the Recalls reach ≥0.7 while providing a massive reduction in memory
fingerprint.

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

#### Correlated data

<details>
<summary><b>Correlated data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.54     1_765.67     1_770.21       1.0000     0.000000        24.41
Exhaustive (self)                                          4.54     5_870.25     5_874.79       1.0000     0.000000        24.41
Exhaustive-OPQ-m8 (query)                              3_220.44       311.75     3_532.19       0.2775          NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_220.44     1_131.54     4_351.98       0.2079          NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_138.83       661.11     3_799.94       0.3906          NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_138.83     2_287.93     5_426.76       0.3044          NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_821.19       143.11     3_964.30       0.4470          NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_821.19       233.67     4_054.86       0.4482          NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_821.19       320.86     4_142.04       0.4483          NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_821.19     1_150.63     4_971.82       0.3499          NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_700.13       233.87     3_934.01       0.6082          NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_700.13       388.68     4_088.82       0.6108          NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_700.13       542.51     4_242.64       0.6110          NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_700.13     1_874.26     5_574.39       0.5263          NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_876.28       215.92     4_092.20       0.4543          NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_876.28       267.55     4_143.83       0.4542          NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_876.28       404.57     4_280.85       0.4542          NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_876.28     1_446.04     5_322.32       0.3553          NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_410.64       345.90     3_756.54       0.6150          NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_410.64       444.86     3_855.50       0.6152          NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_410.64       655.85     4_066.49       0.6152          NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_410.64     2_235.38     5_646.02       0.5294          NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_680.98       280.93     3_961.91       0.4582          NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_680.98       315.83     3_996.82       0.4582          NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_680.98       454.93     4_135.91       0.4582          NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_680.98     1_626.80     5_307.78       0.3600          NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_581.44       417.35     3_998.79       0.6158          NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_581.44       471.65     4_053.09       0.6160          NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_581.44       684.01     4_265.45       0.6161          NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_581.44     2_360.70     5_942.14       0.5319          NaN         1.11
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.03     4_427.73     4_437.76       1.0000     0.000000        48.83
Exhaustive (self)                                         10.03    14_817.27    14_827.30       1.0000     0.000000        48.83
Exhaustive-OPQ-m16 (query)                             6_619.06       671.84     7_290.89       0.2935          NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_619.06     2_543.64     9_162.70       0.2193          NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_604.48     1_500.57     8_105.06       0.4031          NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_604.48     5_306.81    11_911.29       0.3192          NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_460.63     4_088.45    14_549.08       0.5166          NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_460.63    13_986.69    24_447.32       0.4439          NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_996.95       278.97     8_275.93       0.4536          NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_996.95       472.29     8_469.24       0.4571          NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_996.95       652.40     8_649.36       0.4572          NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_996.95     2_528.42    10_525.37       0.3636          NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          8_001.06       428.87     8_429.93       0.6100          NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         8_001.06       727.07     8_728.13       0.6163          NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         8_001.06     1_022.34     9_023.40       0.6166          NaN         2.18
IVF-OPQ-nl158-m32 (self)                               8_001.06     3_757.92    11_758.98       0.5370          NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_668.09       809.98    12_478.07       0.7424          NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_668.09     1_350.25    13_018.35       0.7512          NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_668.09     1_915.22    13_583.32       0.7519          NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_668.09     7_033.51    18_701.61       0.7023          NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         7_158.90       400.30     7_559.21       0.4604          NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         7_158.90       511.59     7_670.50       0.4606          NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         7_158.90       761.36     7_920.26       0.4606          NaN         1.48
IVF-OPQ-nl223-m16 (self)                               7_158.90     2_889.12    10_048.02       0.3674          NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_133.40       671.99     7_805.39       0.6187          NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_133.40       814.09     7_947.50       0.6192          NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_133.40     1_213.86     8_347.26       0.6192          NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_133.40     4_388.38    11_521.78       0.5400          NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_764.09     1_184.31    11_948.40       0.7538          NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_764.09     1_426.23    12_190.32       0.7546          NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_764.09     2_148.18    12_912.27       0.7546          NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_764.09     7_561.27    18_325.36       0.7044          NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         7_329.53       563.97     7_893.51       0.4605          NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         7_329.53       627.57     7_957.10       0.4607          NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         7_329.53       911.29     8_240.82       0.4609          NaN         1.57
IVF-OPQ-nl316-m16 (self)                               7_329.53     3_412.23    10_741.77       0.3687          NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_388.52       831.03     8_219.55       0.6199          NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_388.52       948.70     8_337.22       0.6204          NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_388.52     1_377.87     8_766.39       0.6208          NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_388.52     4_959.93    12_348.45       0.5405          NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_921.06     1_449.48    12_370.54       0.7545          NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_921.06     1_651.82    12_572.88       0.7556          NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_921.06     2_428.17    13_349.23       0.7562          NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_921.06     8_490.30    19_411.37       0.7060          NaN         3.86
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.48     9_870.05     9_890.53       1.0000     0.000000        97.66
Exhaustive (self)                                         20.48    33_018.56    33_039.04       1.0000     0.000000        97.66
Exhaustive-OPQ-m16 (query)                             8_251.56       691.49     8_943.04       0.2181          NaN         2.26
Exhaustive-OPQ-m16 (self)                              8_251.56     3_706.97    11_958.52       0.1648          NaN         2.26
Exhaustive-OPQ-m32 (query)                            14_401.20     1_520.90    15_922.10       0.2950          NaN         3.03
Exhaustive-OPQ-m32 (self)                             14_401.20     6_454.84    20_856.04       0.2217          NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_345.80     4_123.61    17_469.42       0.4084          NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_345.80    14_984.31    28_330.11       0.3278          NaN         4.55
IVF-OPQ-nl158-m16-np7 (query)                         11_188.64       396.01    11_584.66       0.3235          NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        11_188.64       665.00    11_853.64       0.3255          NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        11_188.64       935.71    12_124.36       0.3256          NaN         2.57
IVF-OPQ-nl158-m16 (self)                              11_188.64     4_581.64    15_770.28       0.2416          NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         16_723.29       556.96    17_280.26       0.4579          NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        16_723.29       924.48    17_647.78       0.4626          NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        16_723.29     1_324.42    18_047.71       0.4628          NaN         3.34
IVF-OPQ-nl158-m32 (self)                              16_723.29     5_710.21    22_433.50       0.3725          NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_025.61       917.85    16_943.46       0.6095          NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_025.61     1_512.27    17_537.88       0.6175          NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_025.61     2_103.78    18_129.39       0.6181          NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_025.61     8_441.26    24_466.88       0.5441          NaN         4.86
IVF-OPQ-nl223-m16-np11 (query)                         8_985.11       572.50     9_557.61       0.3281          NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_985.11       734.16     9_719.27       0.3281          NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_985.11     1_084.47    10_069.58       0.3281          NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_985.11     4_998.51    13_983.62       0.2427          NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        15_027.38       790.59    15_817.97       0.4644          NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        15_027.38     1_012.69    16_040.08       0.4645          NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        15_027.38     1_493.41    16_520.79       0.4645          NaN         3.46
IVF-OPQ-nl223-m32 (self)                              15_027.38     6_427.58    21_454.96       0.3750          NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_380.81     1_322.40    15_703.21       0.6206          NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_380.81     1_657.87    16_038.68       0.6209          NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_380.81     2_491.83    16_872.64       0.6209          NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_380.81     9_759.37    24_140.18       0.5450          NaN         4.99
IVF-OPQ-nl316-m16-np15 (query)                         9_374.49       792.87    10_167.36       0.3315          NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_374.49       867.84    10_242.33       0.3315          NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_374.49     1_248.44    10_622.92       0.3315          NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_374.49     5_626.08    15_000.57       0.2453          NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        15_352.49     1_049.34    16_401.82       0.4680          NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        15_352.49     1_182.74    16_535.23       0.4681          NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        15_352.49     1_730.99    17_083.47       0.4682          NaN         3.65
IVF-OPQ-nl316-m32 (self)                              15_352.49     7_194.75    22_547.24       0.3780          NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_783.94     1_671.72    16_455.67       0.6218          NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_783.94     1_928.58    16_712.52       0.6221          NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_783.94     2_781.14    17_565.08       0.6222          NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_783.94    10_768.56    25_552.50       0.5477          NaN         5.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.61     1_774.49     1_779.10       1.0000     0.000000        24.41
Exhaustive (self)                                          4.61     6_135.79     6_140.40       1.0000     0.000000        24.41
Exhaustive-OPQ-m8 (query)                              3_365.85       309.26     3_675.11       0.2340          NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_365.85     1_133.83     4_499.68       0.1728          NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_156.45       657.45     3_813.89       0.3327          NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_156.45     2_265.46     5_421.90       0.2265          NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_908.62       141.53     4_050.15       0.5211          NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_908.62       239.17     4_147.79       0.5213          NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_908.62       353.92     4_262.53       0.5213          NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_908.62     1_185.78     5_094.40       0.4274          NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_729.60       219.19     3_948.80       0.6811          NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_729.60       374.97     4_104.57       0.6814          NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_729.60       521.81     4_251.41       0.6814          NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_729.60     1_816.28     5_545.88       0.6004          NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_587.73       228.65     3_816.38       0.5276          NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_587.73       277.34     3_865.07       0.5276          NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_587.73       416.00     4_003.73       0.5276          NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_587.73     1_436.66     5_024.39       0.4328          NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_445.72       333.03     3_778.74       0.6827          NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_445.72       416.41     3_862.13       0.6829          NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_445.72       621.33     4_067.04       0.6829          NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_445.72     2_167.83     5_613.54       0.6044          NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_687.17       284.92     3_972.09       0.5317          NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_687.17       324.73     4_011.89       0.5318          NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_687.17       469.16     4_156.33       0.5318          NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_687.17     1_645.19     5_332.36       0.4371          NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_647.88       458.42     4_106.29       0.6849          NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_647.88       523.85     4_171.73       0.6849          NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_647.88       758.81     4_406.69       0.6850          NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_647.88     2_618.50     6_266.38       0.6065          NaN         1.11
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.50     4_524.12     4_534.62       1.0000     0.000000        48.83
Exhaustive (self)                                         10.50    14_341.53    14_352.03       1.0000     0.000000        48.83
Exhaustive-OPQ-m16 (query)                             5_830.84       662.39     6_493.23       0.2268          NaN         1.26
Exhaustive-OPQ-m16 (self)                              5_830.84     2_508.86     8_339.70       0.1687          NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_413.97     1_465.95     7_879.93       0.3290          NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_413.97     5_189.83    11_603.80       0.2357          NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_845.57     3_996.63    13_842.20       0.4681          NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_845.57    13_643.49    23_489.06       0.3619          NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_092.48       269.56     7_362.04       0.4875          NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_092.48       451.29     7_543.78       0.4880          NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_092.48       631.07     7_723.55       0.4880          NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_092.48     2_425.62     9_518.11       0.3969          NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_700.56       422.63     8_123.19       0.6543          NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_700.56       717.72     8_418.28       0.6552          NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_700.56     1_008.38     8_708.94       0.6552          NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_700.56     3_692.89    11_393.45       0.5761          NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         10_979.19       756.29    11_735.48       0.7892          NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        10_979.19     1_304.40    12_283.59       0.7906          NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        10_979.19     1_840.98    12_820.17       0.7906          NaN         3.71
IVF-OPQ-nl158-m64 (self)                              10_979.19     6_519.29    17_498.47       0.7457          NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_380.26       406.96     6_787.22       0.4891          NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_380.26       517.86     6_898.12       0.4891          NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_380.26       764.25     7_144.51       0.4891          NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_380.26     2_891.61     9_271.86       0.3985          NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_969.66       625.25     7_594.91       0.6567          NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_969.66       796.69     7_766.35       0.6568          NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_969.66     1_173.89     8_143.56       0.6568          NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_969.66     4_207.94    11_177.60       0.5795          NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_395.72     1_100.40    11_496.12       0.7921          NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_395.72     1_398.75    11_794.47       0.7924          NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_395.72     2_083.75    12_479.47       0.7924          NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_395.72     7_324.22    17_719.94       0.7469          NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_613.40       541.02     7_154.42       0.4903          NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_613.40       613.80     7_227.20       0.4903          NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_613.40       890.01     7_503.41       0.4903          NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_613.40     3_314.71     9_928.11       0.4006          NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_159.54       813.75     7_973.29       0.6572          NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_159.54       923.25     8_082.78       0.6573          NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_159.54     1_346.36     8_505.90       0.6574          NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_159.54     4_799.64    11_959.18       0.5810          NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_687.27     1_494.79    12_182.07       0.7942          NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_687.27     1_615.67    12_302.95       0.7945          NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_687.27     2_368.25    13_055.53       0.7945          NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_687.27     8_381.54    19_068.81       0.7484          NaN         3.86
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.07     9_893.16     9_913.24       1.0000     0.000000        97.66
Exhaustive (self)                                         20.07    33_190.77    33_210.85       1.0000     0.000000        97.66
Exhaustive-OPQ-m16 (query)                             8_189.07       683.33     8_872.40       0.1653          NaN         2.26
Exhaustive-OPQ-m16 (self)                              8_189.07     3_722.48    11_911.55       0.1316          NaN         2.26
Exhaustive-OPQ-m32 (query)                            14_638.00     1_524.57    16_162.56       0.2236          NaN         3.03
Exhaustive-OPQ-m32 (self)                             14_638.00     6_435.95    21_073.94       0.1684          NaN         3.03
Exhaustive-OPQ-m64 (query)                            14_251.28     4_105.86    18_357.15       0.3280          NaN         4.55
Exhaustive-OPQ-m64 (self)                             14_251.28    14_959.22    29_210.51       0.2449          NaN         4.55
IVF-OPQ-nl158-m16-np7 (query)                         11_289.44       381.51    11_670.94       0.3110          NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        11_289.44       644.60    11_934.04       0.3115          NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        11_289.44       901.04    12_190.48       0.3115          NaN         2.57
IVF-OPQ-nl158-m16 (self)                              11_289.44     4_483.16    15_772.60       0.2309          NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         18_733.15       539.76    19_272.91       0.4609          NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        18_733.15       913.66    19_646.80       0.4622          NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        18_733.15     1_298.81    20_031.96       0.4622          NaN         3.34
IVF-OPQ-nl158-m32 (self)                              18_733.15     5_734.59    24_467.74       0.3789          NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_833.88       903.93    17_737.81       0.6225          NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_833.88     1_538.03    18_371.91       0.6251          NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_833.88     2_188.84    19_022.72       0.6251          NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_833.88     8_604.94    25_438.82       0.5501          NaN         4.86
IVF-OPQ-nl223-m16-np11 (query)                         9_366.33       569.49     9_935.81       0.3131          NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_366.33       747.66    10_113.99       0.3132          NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_366.33     1_075.89    10_442.22       0.3132          NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_366.33     5_030.53    14_396.85       0.2327          NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        15_992.31       796.24    16_788.55       0.4634          NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        15_992.31     1_011.19    17_003.50       0.4635          NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        15_992.31     1_507.90    17_500.21       0.4635          NaN         3.46
IVF-OPQ-nl223-m32 (self)                              15_992.31     6_417.26    22_409.57       0.3795          NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        15_266.40     1_293.26    16_559.67       0.6265          NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        15_266.40     1_654.38    16_920.78       0.6267          NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        15_266.40     2_487.35    17_753.75       0.6267          NaN         4.99
IVF-OPQ-nl223-m64 (self)                              15_266.40     9_777.81    25_044.22       0.5517          NaN         4.99
IVF-OPQ-nl316-m16-np15 (query)                         9_717.80       783.78    10_501.58       0.3146          NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_717.80       898.85    10_616.65       0.3146          NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_717.80     1_272.92    10_990.72       0.3146          NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_717.80     5_553.87    15_271.67       0.2326          NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        16_018.57     1_053.67    17_072.24       0.4624          NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        16_018.57     1_186.76    17_205.33       0.4625          NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        16_018.57     1_748.35    17_766.92       0.4625          NaN         3.65
IVF-OPQ-nl316-m32 (self)                              16_018.57     7_275.37    23_293.94       0.3793          NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        15_733.53     1_786.89    17_520.42       0.6271          NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        15_733.53     1_910.77    17_644.30       0.6272          NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        15_733.53     2_845.25    18_578.78       0.6272          NaN         5.17
IVF-OPQ-nl316-m64 (self)                              15_733.53    10_875.03    26_608.56       0.5535          NaN         5.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 128 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.22     1_932.43     1_935.65       1.0000     0.000000        24.41
Exhaustive (self)                                          3.22     6_323.49     6_326.70       1.0000     0.000000        24.41
Exhaustive-OPQ-m8 (query)                              3_281.32       313.19     3_594.51       0.1253          NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_281.32     1_136.93     4_418.25       0.2565          NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_125.22       667.82     3_793.04       0.1716          NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_125.22     2_285.15     5_410.37       0.3354          NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_787.54       166.04     3_953.58       0.3011          NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_787.54       300.00     4_087.54       0.3010          NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_787.54       397.59     4_185.12       0.3010          NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_787.54     1_412.36     5_199.89       0.5401          NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_708.31       275.55     3_983.86       0.3990          NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_708.31       458.63     4_166.94       0.3990          NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_708.31       636.62     4_344.93       0.3990          NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_708.31     2_216.50     5_924.82       0.6569          NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_859.55       215.98     4_075.53       0.3275          NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_859.55       280.53     4_140.08       0.3275          NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_859.55       412.51     4_272.07       0.3275          NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_859.55     1_443.31     5_302.86       0.5946          NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_269.86       350.37     3_620.23       0.4273          NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_269.86       453.09     3_722.94       0.4273          NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_269.86       663.80     3_933.65       0.4273          NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_269.86     2_297.93     5_567.78       0.6983          NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_477.97       279.60     3_757.57       0.3369          NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_477.97       322.67     3_800.64       0.3369          NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_477.97       469.69     3_947.66       0.3369          NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_477.97     1_644.26     5_122.23       0.6012          NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_270.10       462.57     3_732.68       0.4370          NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_270.10       545.18     3_815.28       0.4370          NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_270.10       782.12     4_052.23       0.4370          NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_270.10     2_675.50     5_945.60       0.7007          NaN         1.11
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.91     4_143.70     4_150.61       1.0000     0.000000        48.83
Exhaustive (self)                                          6.91    14_117.00    14_123.91       1.0000     0.000000        48.83
Exhaustive-OPQ-m16 (query)                             5_784.31       659.41     6_443.72       0.1141          NaN         1.26
Exhaustive-OPQ-m16 (self)                              5_784.31     2_505.72     8_290.03       0.3027          NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_419.01     1_475.82     7_894.83       0.1529          NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_419.01     5_267.26    11_686.27       0.4066          NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_872.58     3_991.88    13_864.46       0.2357          NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_872.58    13_628.27    23_500.85       0.5577          NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          6_813.16       295.33     7_108.49       0.2893          NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         6_813.16       509.23     7_322.39       0.2895          NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         6_813.16       707.12     7_520.28       0.2895          NaN         1.42
IVF-OPQ-nl158-m16 (self)                               6_813.16     2_718.52     9_531.68       0.6663          NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_704.85       512.49     8_217.34       0.3951          NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_704.85       853.77     8_558.62       0.3953          NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_704.85     1_209.87     8_914.72       0.3953          NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_704.85     4_396.46    12_101.31       0.7601          NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         12_157.25       953.01    13_110.25       0.6249          NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        12_157.25     1_654.99    13_812.24       0.6257          NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        12_157.25     2_350.93    14_508.18       0.6258          NaN         3.71
IVF-OPQ-nl158-m64 (self)                              12_157.25     8_258.77    20_416.02       0.8432          NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_822.70       408.80     7_231.50       0.3024          NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_822.70       529.17     7_351.86       0.3024          NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_822.70       771.85     7_594.55       0.3024          NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_822.70     2_902.91     9_725.60       0.6832          NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_866.96       646.81     7_513.77       0.4094          NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_866.96       825.60     7_692.56       0.4095          NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_866.96     1_226.35     8_093.31       0.4096          NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_866.96     4_486.95    11_353.91       0.7704          NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_472.61     1_120.75    11_593.36       0.6372          NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_472.61     1_439.35    11_911.95       0.6375          NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_472.61     2_143.96    12_616.57       0.6376          NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_472.61     7_583.55    18_056.15       0.8497          NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_952.59       564.72     7_517.31       0.3114          NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_952.59       628.94     7_581.53       0.3114          NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_952.59       924.76     7_877.35       0.3114          NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_952.59     3_403.55    10_356.14       0.6864          NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         6_843.04       844.07     7_687.11       0.4173          NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         6_843.04       948.89     7_791.93       0.4173          NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         6_843.04     1_390.22     8_233.26       0.4174          NaN         2.34
IVF-OPQ-nl316-m32 (self)                               6_843.04     4_969.54    11_812.58       0.7705          NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_431.96     1_451.18    11_883.13       0.6493          NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_431.96     1_654.44    12_086.40       0.6495          NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_431.96     2_426.98    12_858.94       0.6496          NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_431.96     8_473.14    18_905.10       0.8495          NaN         3.86
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.98     9_844.57     9_864.55       1.0000     0.000000        97.66
Exhaustive (self)                                         19.98    33_263.55    33_283.53       1.0000     0.000000        97.66
Exhaustive-OPQ-m16 (query)                             7_903.74       685.64     8_589.38       0.0880          NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_903.74     3_715.52    11_619.25       0.2699          NaN         2.26
Exhaustive-OPQ-m32 (query)                            14_077.46     1_520.44    15_597.89       0.1079          NaN         3.03
Exhaustive-OPQ-m32 (self)                             14_077.46     6_430.23    20_507.69       0.3567          NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_976.10     4_113.00    18_089.10       0.1462          NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_976.10    14_979.75    28_955.86       0.4750          NaN         4.55
IVF-OPQ-nl158-m16-np7 (query)                         10_603.31       391.52    10_994.83       0.2178          NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_603.31       659.33    11_262.64       0.2178          NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_603.31       931.89    11_535.20       0.2178          NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_603.31     4_583.25    15_186.55       0.6326          NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         16_230.81       616.33    16_847.14       0.2705          NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        16_230.81     1_001.98    17_232.79       0.2707          NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        16_230.81     1_410.33    17_641.14       0.2707          NaN         3.34
IVF-OPQ-nl158-m32 (self)                              16_230.81     6_093.43    22_324.24       0.7168          NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_937.45       984.46    16_921.92       0.3706          NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_937.45     1_698.60    17_636.05       0.3710          NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_937.45     2_410.33    18_347.79       0.3709          NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_937.45     9_467.27    25_404.72       0.8013          NaN         4.86
IVF-OPQ-nl223-m16-np11 (query)                         7_456.94       564.67     8_021.61       0.2313          NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         7_456.94       718.20     8_175.14       0.2313          NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         7_456.94     1_056.34     8_513.28       0.2313          NaN         2.70
IVF-OPQ-nl223-m16 (self)                               7_456.94     4_944.54    12_401.48       0.6444          NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        12_411.08       775.05    13_186.13       0.2807          NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        12_411.08       987.29    13_398.37       0.2807          NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        12_411.08     1_457.04    13_868.12       0.2807          NaN         3.46
IVF-OPQ-nl223-m32 (self)                              12_411.08     6_236.13    18_647.21       0.7233          NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_011.71     1_252.09    14_263.80       0.3785          NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_011.71     1_595.99    14_607.70       0.3787          NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_011.71     2_381.39    15_393.10       0.3787          NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_011.71     9_432.61    22_444.32       0.8020          NaN         4.99
IVF-OPQ-nl316-m16-np15 (query)                         7_556.76       763.08     8_319.84       0.2416          NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         7_556.76       858.76     8_415.52       0.2415          NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         7_556.76     1_235.32     8_792.08       0.2415          NaN         2.88
IVF-OPQ-nl316-m16 (self)                               7_556.76     5_520.06    13_076.82       0.6599          NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        12_467.79     1_030.60    13_498.39       0.2952          NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        12_467.79     1_160.25    13_628.04       0.2952          NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        12_467.79     1_687.19    14_154.98       0.2951          NaN         3.65
IVF-OPQ-nl316-m32 (self)                              12_467.79     7_037.90    19_505.69       0.7381          NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        12_985.45     1_636.50    14_621.95       0.3962          NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        12_985.45     1_860.50    14_845.94       0.3964          NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        12_985.45     2_726.20    15_711.65       0.3963          NaN         5.17
IVF-OPQ-nl316-m64 (self)                              12_985.45    10_575.21    23_560.66       0.8135          NaN         5.17
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

## Conclusions

The crate offers various quantisations that can reduce the memory fingerprint
of the respective index quite substantially (usually at the cost of precision).
Generally speaking, the quantisations are performing worse at small dimensions
and become better and more accurate at large dimensions – exactly the situation
you should be using them.

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*Last update: 2026/03/15 with version **0.2.5***
