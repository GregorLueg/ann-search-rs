## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger
print. If you wish to run on the examples, you can do so via:

```bash
cargo run --example gridsearch_sq8 --release --features quantised
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided. Compared to other
benchmarks, we will use the `"correlated"`, `"lowrank"` and `"embedding"`
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
higher for Cosine compared to Euclidean distance.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

<details>
<summary><b>BF16 quantisations - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.28     1_456.96     1_460.24       1.0000          1.0000        18.31
Exhaustive (self)                                          3.28    15_493.75    15_497.03       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.25     1_229.36     1_234.61       0.9828          1.0000         9.16
Exhaustive-BF16 (self)                                     5.25    16_315.40    16_320.65       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              389.96        90.45       480.41       0.9806          1.0003         9.19
IVF-BF16-nl273-np16 (query)                              389.96       104.01       493.97       0.9825          1.0001         9.19
IVF-BF16-nl273-np23 (query)                              389.96       141.81       531.77       0.9828          1.0000         9.19
IVF-BF16-nl273 (self)                                    389.96     1_409.90     1_799.86       0.9798          1.0001         9.19
IVF-BF16-nl387-np19 (query)                              750.95        93.31       844.25       0.9821          1.0001         9.21
IVF-BF16-nl387-np27 (query)                              750.95       123.49       874.44       0.9828          1.0000         9.21
IVF-BF16-nl387 (self)                                    750.95     1_220.23     1_971.18       0.9798          1.0001         9.21
IVF-BF16-nl547-np23 (query)                            1_451.76        86.36     1_538.12       0.9772          1.0005         9.23
IVF-BF16-nl547-np27 (query)                            1_451.76        97.61     1_549.38       0.9815          1.0001         9.23
IVF-BF16-nl547-np33 (query)                            1_451.76       113.69     1_565.46       0.9828          1.0000         9.23
IVF-BF16-nl547 (self)                                  1_451.76     1_143.65     2_595.42       0.9798          1.0001         9.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.37     1_582.68     1_587.06       1.0000          1.0000        18.88
Exhaustive (self)                                          4.37    15_974.38    15_978.75       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                    5.83     1_230.59     1_236.42       0.8870          0.9927         9.44
Exhaustive-BF16 (self)                                     5.83    15_732.96    15_738.79       1.0000          1.0000         9.44
IVF-BF16-nl273-np13 (query)                              375.28        92.23       467.51       0.8860          0.9929         9.48
IVF-BF16-nl273-np16 (query)                              375.28       112.96       488.23       0.8870          0.9927         9.48
IVF-BF16-nl273-np23 (query)                              375.28       149.83       525.10       0.8870          0.9927         9.48
IVF-BF16-nl273 (self)                                    375.28     1_513.44     1_888.71       0.8852          0.9925         9.48
IVF-BF16-nl387-np19 (query)                              713.94        97.36       811.30       0.8867          0.9928         9.49
IVF-BF16-nl387-np27 (query)                              713.94       128.18       842.12       0.8870          0.9927         9.49
IVF-BF16-nl387 (self)                                    713.94     1_290.70     2_004.65       0.8852          0.9925         9.49
IVF-BF16-nl547-np23 (query)                            1_399.52        90.33     1_489.85       0.8849          0.9931         9.51
IVF-BF16-nl547-np27 (query)                            1_399.52       100.27     1_499.79       0.8866          0.9928         9.51
IVF-BF16-nl547-np33 (query)                            1_399.52       119.27     1_518.80       0.8870          0.9927         9.51
IVF-BF16-nl547 (self)                                  1_399.52     1_200.45     2_599.98       0.8852          0.9925         9.51
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.02     1_499.71     1_502.73       1.0000          1.0000        18.31
Exhaustive (self)                                          3.02    15_151.75    15_154.78       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.89     1_161.81     1_166.71       0.9223          1.0022         9.16
Exhaustive-BF16 (self)                                     4.89    15_263.75    15_268.64       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              374.93        89.71       464.64       0.9223          1.0022         9.19
IVF-BF16-nl273-np16 (query)                              374.93       106.43       481.35       0.9223          1.0022         9.19
IVF-BF16-nl273-np23 (query)                              374.93       147.97       522.89       0.9223          1.0022         9.19
IVF-BF16-nl273 (self)                                    374.93     1_487.50     1_862.42       0.9031          1.0049         9.19
IVF-BF16-nl387-np19 (query)                              717.82        94.98       812.81       0.9223          1.0022         9.21
IVF-BF16-nl387-np27 (query)                              717.82       123.47       841.29       0.9223          1.0022         9.21
IVF-BF16-nl387 (self)                                    717.82     1_228.84     1_946.66       0.9031          1.0049         9.21
IVF-BF16-nl547-np23 (query)                            1_412.36        83.14     1_495.50       0.9223          1.0022         9.23
IVF-BF16-nl547-np27 (query)                            1_412.36        92.88     1_505.24       0.9223          1.0022         9.23
IVF-BF16-nl547-np33 (query)                            1_412.36       108.52     1_520.88       0.9223          1.0022         9.23
IVF-BF16-nl547 (self)                                  1_412.36     1_090.09     2_502.45       0.9031          1.0049         9.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.05     1_474.67     1_477.72       1.0000          1.0000        18.31
Exhaustive (self)                                          3.05    15_281.07    15_284.13       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.14     1_165.02     1_170.16       0.9516          1.0013         9.16
Exhaustive-BF16 (self)                                     5.14    15_294.27    15_299.41       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              377.12        79.85       456.97       0.9516          1.0013         9.19
IVF-BF16-nl273-np16 (query)                              377.12        90.23       467.35       0.9516          1.0013         9.19
IVF-BF16-nl273-np23 (query)                              377.12       123.92       501.04       0.9516          1.0013         9.19
IVF-BF16-nl273 (self)                                    377.12     1_213.65     1_590.77       0.9405          1.0030         9.19
IVF-BF16-nl387-np19 (query)                              717.87        82.34       800.21       0.9516          1.0013         9.21
IVF-BF16-nl387-np27 (query)                              717.87       106.47       824.34       0.9516          1.0013         9.21
IVF-BF16-nl387 (self)                                    717.87     1_061.29     1_779.16       0.9405          1.0030         9.21
IVF-BF16-nl547-np23 (query)                            1_380.23        78.88     1_459.11       0.9516          1.0013         9.23
IVF-BF16-nl547-np27 (query)                            1_380.23        86.37     1_466.59       0.9516          1.0013         9.23
IVF-BF16-nl547-np33 (query)                            1_380.23        98.11     1_478.34       0.9516          1.0013         9.23
IVF-BF16-nl547 (self)                                  1_380.23       984.93     2_365.16       0.9405          1.0030         9.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.09     5_668.68     5_682.78       1.0000          1.0000        73.24
Exhaustive (self)                                         14.09    58_086.31    58_100.40       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   22.50     4_961.02     4_983.52       0.9717          1.0016        36.62
Exhaustive-BF16 (self)                                    22.50    58_667.76    58_690.26       1.0000          1.0000        36.62
IVF-BF16-nl273-np13 (query)                              636.74       262.06       898.80       0.9717          1.0016        36.76
IVF-BF16-nl273-np16 (query)                              636.74       298.59       935.33       0.9717          1.0016        36.76
IVF-BF16-nl273-np23 (query)                              636.74       416.72     1_053.46       0.9717          1.0016        36.76
IVF-BF16-nl273 (self)                                    636.74     4_260.79     4_897.53       0.9674          1.0044        36.76
IVF-BF16-nl387-np19 (query)                            1_220.04       278.62     1_498.66       0.9717          1.0016        36.81
IVF-BF16-nl387-np27 (query)                            1_220.04       358.39     1_578.43       0.9717          1.0016        36.81
IVF-BF16-nl387 (self)                                  1_220.04     3_653.14     4_873.17       0.9674          1.0044        36.81
IVF-BF16-nl547-np23 (query)                            2_452.83       260.06     2_712.89       0.9717          1.0016        36.89
IVF-BF16-nl547-np27 (query)                            2_452.83       285.64     2_738.47       0.9717          1.0016        36.89
IVF-BF16-nl547-np33 (query)                            2_452.83       335.30     2_788.13       0.9717          1.0016        36.89
IVF-BF16-nl547 (self)                                  2_452.83     3_312.43     5_765.27       0.9674          1.0044        36.89
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.24     1_467.22     1_470.46       1.0000          1.0000        18.31
Exhaustive (self)                                          3.24    14_699.11    14_702.35       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.92       724.35       731.27       0.7939             NaN         4.58
Exhaustive-SQ8 (self)                                      6.92     7_370.60     7_377.52       0.7931             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               423.97        50.39       474.36       0.7862             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               423.97        58.48       482.45       0.7871             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               423.97        77.41       501.38       0.7872             NaN         4.61
IVF-SQ8-nl273 (self)                                     423.97       770.57     1_194.54       0.7862             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               720.21        53.12       773.33       0.7965             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               720.21        68.95       789.16       0.7968             NaN         4.63
IVF-SQ8-nl387 (self)                                     720.21       684.17     1_404.38       0.7961             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_388.99        52.02     1_441.02       0.7919             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_388.99        56.28     1_445.27       0.7936             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_388.99        64.49     1_453.48       0.7940             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_388.99       648.59     2_037.58       0.7931             NaN         4.65
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.97     1_496.03     1_500.00       1.0000          1.0000        18.88
Exhaustive (self)                                          3.97    15_364.01    15_367.98       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                     7.19       978.59       985.78       0.8273             NaN         5.15
Exhaustive-SQ8 (self)                                      7.19     9_801.27     9_808.47       0.8260             NaN         5.15
IVF-SQ8-nl273-np13 (query)                               367.10        62.68       429.77       0.8256             NaN         5.19
IVF-SQ8-nl273-np16 (query)                               367.10        72.41       439.50       0.8262             NaN         5.19
IVF-SQ8-nl273-np23 (query)                               367.10        99.21       466.31       0.8263             NaN         5.19
IVF-SQ8-nl273 (self)                                     367.10       982.27     1_349.37       0.8252             NaN         5.19
IVF-SQ8-nl387-np19 (query)                               703.55        66.18       769.73       0.8269             NaN         5.20
IVF-SQ8-nl387-np27 (query)                               703.55        86.00       789.54       0.8272             NaN         5.20
IVF-SQ8-nl387 (self)                                     703.55       855.23     1_558.78       0.8264             NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_363.34        62.92     1_426.26       0.8259             NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_363.34        68.86     1_432.20       0.8275             NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_363.34        80.41     1_443.75       0.8278             NaN         5.22
IVF-SQ8-nl547 (self)                                   1_363.34       802.00     2_165.34       0.8267             NaN         5.22
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.10     1_460.25     1_463.35       1.0000          1.0000        18.31
Exhaustive (self)                                          3.10    14_916.25    14_919.35       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.92       708.13       715.05       0.7705             NaN         4.58
Exhaustive-SQ8 (self)                                      6.92     7_309.72     7_316.64       0.7670             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               368.48        51.48       419.96       0.7717             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               368.48        60.74       429.22       0.7716             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               368.48        82.85       451.33       0.7716             NaN         4.61
IVF-SQ8-nl273 (self)                                     368.48       836.22     1_204.70       0.7688             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               705.26        53.83       759.09       0.7715             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               705.26        69.90       775.16       0.7715             NaN         4.63
IVF-SQ8-nl387 (self)                                     705.26       706.56     1_411.82       0.7684             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_372.62        50.16     1_422.78       0.7709             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_372.62        55.28     1_427.90       0.7709             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_372.62        63.38     1_436.00       0.7709             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_372.62       638.44     2_011.06       0.7672             NaN         4.65
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.13     1_460.74     1_463.87       1.0000          1.0000        18.31
Exhaustive (self)                                          3.13    14_807.39    14_810.52       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.72       709.97       716.69       0.7050             NaN         4.58
Exhaustive-SQ8 (self)                                      6.72     7_347.19     7_353.91       0.7116             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               367.76        46.47       414.23       0.7055             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               367.76        54.27       422.03       0.7056             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               367.76        75.63       443.39       0.7055             NaN         4.61
IVF-SQ8-nl273 (self)                                     367.76       697.59     1_065.35       0.7124             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               707.98        48.40       756.37       0.7057             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               707.98        61.49       769.47       0.7056             NaN         4.63
IVF-SQ8-nl387 (self)                                     707.98       616.03     1_324.01       0.7122             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_378.90        47.20     1_426.11       0.7050             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_378.90        50.94     1_429.85       0.7050             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_378.90        57.56     1_436.47       0.7049             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_378.90       578.71     1_957.61       0.7117             NaN         4.65
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### More dimensions

With higher dimensions, the Recall in more structured, correlated data does
become better again.

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.28     5_599.37     5_614.65       1.0000          1.0000        73.24
Exhaustive (self)                                         15.28    57_451.60    57_466.88       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    37.86     1_658.80     1_696.66       0.7859             NaN        18.31
Exhaustive-SQ8 (self)                                     37.86    16_985.61    17_023.47       0.8138             NaN        18.31
IVF-SQ8-nl273-np13 (query)                               651.64        95.74       747.38       0.7872             NaN        18.45
IVF-SQ8-nl273-np16 (query)                               651.64       104.39       756.03       0.7872             NaN        18.45
IVF-SQ8-nl273-np23 (query)                               651.64       141.90       793.54       0.7872             NaN        18.45
IVF-SQ8-nl273 (self)                                     651.64     1_342.71     1_994.35       0.8139             NaN        18.45
IVF-SQ8-nl387-np19 (query)                             1_209.19        99.21     1_308.40       0.7873             NaN        18.50
IVF-SQ8-nl387-np27 (query)                             1_209.19       125.77     1_334.96       0.7873             NaN        18.50
IVF-SQ8-nl387 (self)                                   1_209.19     1_189.00     2_398.19       0.8140             NaN        18.50
IVF-SQ8-nl547-np23 (query)                             2_498.07       101.81     2_599.88       0.7864             NaN        18.58
IVF-SQ8-nl547-np27 (query)                             2_498.07       107.56     2_605.62       0.7864             NaN        18.58
IVF-SQ8-nl547-np33 (query)                             2_498.07       127.65     2_625.72       0.7864             NaN        18.58
IVF-SQ8-nl547 (self)                                   2_498.07     1_130.84     3_628.91       0.8139             NaN        18.58
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### Product quantisations

Product quantisation methods (PQ, OPQ) compress vectors far more aggressively
than BF16 or SQ8 by dividing each vector into subvectors and encoding each with
a small codebook. Compared to the previous benchmarks, we use higher
dimensionality (128, 256, 512) with reduced sample counts (50k - for faster
bench marking) to reflect the regime where these methods are most relevant:
large vectors under memory pressure. We benchmark against three synthetic data
types of increasing difficulty:

- `"correlated"` data with subspace-clustered activation patterns.
- `"lowrank"` data embedded from a lower-dimensional manifold.
- Lastly, `"quantisation"` stress data that combines power-law spectral decay
  with norm-stratified clusters: specifically designed to expose failure modes
  of aggressive quantisation such as sign binarisation, axis-aligned sub-vector
  splits, and low-bit angular resolution loss.

#### Product quantisation (Exhaustive and IVF)

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

##### Correlated data

<details>
<summary><b>Correlated data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.50     1_734.32     1_738.82       1.0000          1.0000        24.41
Exhaustive (self)                                          4.50     5_717.87     5_722.37       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                622.83       640.86     1_263.69       0.2473             NaN         0.89
Exhaustive-PQ-m16 (self)                                 622.83     2_127.22     2_750.05       0.2042             NaN         0.89
Exhaustive-PQ-m32 (query)                                968.26     1_486.26     2_454.52       0.3277             NaN         1.65
Exhaustive-PQ-m32 (self)                                 968.26     4_911.21     5_879.48       0.2697             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_951.36     3_927.80     5_879.16       0.5384             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_951.36    13_037.90    14_989.26       0.4791             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_439.86       218.18     1_658.04       0.4653             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_439.86       355.02     1_794.88       0.4653             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_439.86       496.19     1_936.05       0.4653             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_439.86     1_617.23     3_057.09       0.3714             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_818.09       373.86     2_191.95       0.6961             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_818.09       669.30     2_487.39       0.6961             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_818.09       950.60     2_768.70       0.6961             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_818.09     2_908.74     4_726.83       0.6342             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_592.70       799.85     3_392.55       0.8877             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_592.70     1_360.97     3_953.67       0.8877             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_592.70     1_930.30     4_523.00       0.8877             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_592.70     6_356.25     8_948.95       0.8687             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            955.45       299.71     1_255.16       0.4695             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            955.45       379.52     1_334.97       0.4695             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            955.45       565.43     1_520.87       0.4695             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  955.45     1_845.66     2_801.10       0.3763             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_305.15       510.27     1_815.42       0.7004             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_305.15       653.56     1_958.71       0.7004             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_305.15       983.95     2_289.11       0.7004             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_305.15     3_231.24     4_536.40       0.6390             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_314.69     1_158.12     3_472.81       0.8909             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_314.69     1_467.13     3_781.82       0.8909             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_314.69     2_196.40     4_511.09       0.8909             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_314.69     7_340.60     9_655.29       0.8701             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_066.25       400.14     1_466.39       0.4792             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_066.25       434.06     1_500.32       0.4792             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_066.25       633.86     1_700.12       0.4792             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_066.25     2_093.91     3_160.17       0.3830             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_444.93       664.89     2_109.82       0.7051             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_444.93       750.90     2_195.83       0.7051             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_444.93     1_090.67     2_535.59       0.7051             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_444.93     3_628.19     5_073.12       0.6426             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_427.29     1_531.32     3_958.62       0.8909             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_427.29     1_700.23     4_127.52       0.8909             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_427.29     2_495.85     4_923.14       0.8909             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_427.29     8_361.62    10_788.92       0.8710             NaN         3.34
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.88     4_138.70     4_148.58       1.0000          1.0000        48.83
Exhaustive (self)                                          9.88    13_653.40    13_663.28       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_702.22       713.57     2_415.79       0.1810             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_702.22     2_164.45     3_866.67       0.1575             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_198.31     1_482.10     2_680.41       0.2138             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_198.31     4_946.19     6_144.51       0.1773             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_931.77     3_895.14     5_826.90       0.2946             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_931.77    12_991.80    14_923.56       0.2455             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_787.43       242.35     3_029.77       0.3002             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_787.43       403.64     3_191.06       0.3003             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_787.43       562.33     3_349.76       0.3003             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_787.43     1_892.45     4_679.88       0.2095             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_369.26       394.93     2_764.18       0.4314             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_369.26       659.60     3_028.86       0.4316             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_369.26       935.43     3_304.69       0.4316             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_369.26     3_114.78     5_484.03       0.3423             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_099.53       721.72     3_821.25       0.6710             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_099.53     1_212.00     4_311.53       0.6714             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_099.53     1_717.47     4_817.00       0.6714             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_099.53     5_825.64     8_925.17       0.6119             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_190.11       374.89     2_565.01       0.3050             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_190.11       478.96     2_669.07       0.3050             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_190.11       697.72     2_887.84       0.3050             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_190.11     2_321.25     4_511.37       0.2137             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_739.05       594.63     2_333.68       0.4357             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_739.05       751.20     2_490.26       0.4357             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_739.05     1_119.76     2_858.81       0.4357             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_739.05     3_725.75     5_464.81       0.3448             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_463.03     1_050.24     3_513.27       0.6756             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_463.03     1_331.52     3_794.55       0.6756             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_463.03     2_008.80     4_471.83       0.6756             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_463.03     7_208.01     9_671.04       0.6158             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_490.92       513.11     3_004.03       0.3102             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_490.92       542.99     3_033.91       0.3102             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_490.92       791.26     3_282.19       0.3102             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_490.92     2_683.78     5_174.71       0.2160             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_008.76       771.04     2_779.80       0.4406             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_008.76       862.97     2_871.74       0.4406             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_008.76     1_262.95     3_271.71       0.4406             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_008.76     4_221.91     6_230.68       0.3485             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_733.21     1_351.64     4_084.85       0.6786             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_733.21     1_522.86     4_256.07       0.6786             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_733.21     2_243.23     4_976.44       0.6786             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_733.21     7_450.59    10_183.79       0.6183             NaN         3.61
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.42     9_510.86     9_531.28       1.0000          1.0000        97.66
Exhaustive (self)                                         20.42    31_990.58    32_011.00       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_156.85       661.10     1_817.95       0.1437             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_156.85     2_198.07     3_354.92       0.1309             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_339.36     1_505.79     4_845.15       0.1563             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_339.36     5_021.90     8_361.27       0.1378             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_429.94     3_900.98     6_330.92       0.1894             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_429.94    13_246.84    15_676.78       0.1563             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_711.59       354.72     4_066.31       0.2147             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_711.59       586.77     4_298.36       0.2147             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_711.59       846.97     4_558.56       0.2147             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_711.59     2_749.05     6_460.64       0.1475             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_868.05       498.03     6_366.08       0.2772             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_868.05       844.74     6_712.79       0.2772             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_868.05     1_177.73     7_045.77       0.2772             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_868.05     3_957.23     9_825.28       0.1883             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_958.25       832.04     5_790.29       0.4068             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_958.25     1_394.62     6_352.88       0.4069             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_958.25     1_951.30     6_909.55       0.4069             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_958.25     6_530.15    11_488.40       0.3186             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_147.13       519.51     2_666.64       0.2184             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_147.13       648.49     2_795.63       0.2184             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_147.13       958.60     3_105.73       0.2184             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_147.13     3_221.64     5_368.78       0.1501             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_314.14       732.13     5_046.27       0.2817             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_314.14       928.19     5_242.32       0.2817             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_314.14     1_357.20     5_671.33       0.2817             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_314.14     4_545.11     8_859.25       0.1912             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_438.79     1_205.24     4_644.04       0.4136             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_438.79     1_508.92     4_947.71       0.4136             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_438.79     2_260.98     5_699.77       0.4136             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_438.79     7_494.93    10_933.73       0.3225             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_531.19       674.17     3_205.36       0.2211             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_531.19       763.64     3_294.83       0.2211             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_531.19     1_098.71     3_629.90       0.2211             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_531.19     3_643.24     6_174.43       0.1518             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_680.21       957.15     5_637.36       0.2852             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_680.21     1_068.54     5_748.75       0.2852             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_680.21     1_551.99     6_232.20       0.2852             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_680.21     5_221.51     9_901.72       0.1932             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_810.44     1_558.05     5_368.49       0.4162             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_810.44     1_761.82     5_572.27       0.4162             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_810.44     2_556.00     6_366.44       0.4162             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_810.44     9_174.41    12_984.85       0.3254             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Lowrank data

<details>
<summary><b>Lowrank data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.47     1_762.70     1_767.17       1.0000          1.0000        24.41
Exhaustive (self)                                          4.47     5_865.13     5_869.59       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                620.71       639.84     1_260.55       0.4344             NaN         0.89
Exhaustive-PQ-m16 (self)                                 620.71     2_141.29     2_762.00       0.3488             NaN         0.89
Exhaustive-PQ-m32 (query)                                976.55     1_477.91     2_454.46       0.5729             NaN         1.65
Exhaustive-PQ-m32 (self)                                 976.55     4_925.47     5_902.02       0.4904             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_948.19     3_900.80     5_848.98       0.7600             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_948.19    13_083.04    15_031.22       0.7064             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_208.47       193.86     1_402.33       0.7102             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_208.47       315.05     1_523.52       0.7102             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_208.47       434.68     1_643.15       0.7102             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_208.47     1_438.24     2_646.71       0.6382             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_561.05       341.23     1_902.28       0.8425             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_561.05       550.21     2_111.26       0.8425             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_561.05       781.34     2_342.39       0.8425             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_561.05     2_549.67     4_110.72       0.8047             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_562.37       775.98     3_338.35       0.9474             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_562.37     1_248.93     3_811.30       0.9474             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_562.37     1_741.19     4_303.56       0.9474             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_562.37     5_813.22     8_375.59       0.9357             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                          1_048.62       290.12     1_338.74       0.7169             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                          1_048.62       383.04     1_431.66       0.7170             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                          1_048.62       532.98     1_581.60       0.7170             NaN         1.00
IVF-PQ-nl223-m16 (self)                                1_048.62     1_767.85     2_816.47       0.6405             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_413.33       503.17     1_916.50       0.8466             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_413.33       654.21     2_067.54       0.8467             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_413.33       941.28     2_354.61       0.8467             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_413.33     3_150.80     4_564.13       0.8093             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_391.39     1_141.01     3_532.41       0.9484             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_391.39     1_426.82     3_818.22       0.9486             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_391.39     2_233.01     4_624.40       0.9486             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_391.39     7_090.06     9_481.45       0.9370             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_162.22       405.14     1_567.36       0.7211             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_162.22       436.40     1_598.63       0.7211             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_162.22       658.67     1_820.89       0.7211             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_162.22     2_124.31     3_286.53       0.6396             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_515.98       662.13     2_178.11       0.8502             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_515.98       751.00     2_266.98       0.8503             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_515.98     1_086.55     2_602.53       0.8503             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_515.98     3_652.20     5_168.18       0.8116             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_507.80     1_490.61     3_998.41       0.9495             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_507.80     1_677.78     4_185.58       0.9496             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_507.80     2_450.50     4_958.30       0.9496             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_507.80     8_176.24    10_684.04       0.9384             NaN         3.34
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.84     4_113.80     4_123.64       1.0000          1.0000        48.83
Exhaustive (self)                                          9.84    13_797.06    13_806.90       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_645.12       655.24     2_300.36       0.2962             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_645.12     2_165.16     3_810.27       0.2324             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_203.11     1_480.06     2_683.17       0.4045             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_203.11     4_930.33     6_133.44       0.3204             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_924.91     4_172.55     6_097.45       0.5368             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_924.91    12_978.54    14_903.44       0.4607             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_715.73       244.00     2_959.73       0.5293             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_715.73       390.20     3_105.93       0.5293             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_715.73       535.79     3_251.52       0.5293             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_715.73     1_796.61     4_512.34       0.4281             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_274.48       390.54     2_665.02       0.6699             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_274.48       631.10     2_905.58       0.6699             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_274.48       882.76     3_157.25       0.6699             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_274.48     2_872.23     5_146.71       0.6073             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_010.08       697.37     3_707.45       0.8316             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_010.08     1_120.14     4_130.22       0.8316             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_010.08     1_558.69     4_568.78       0.8316             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_010.08     5_183.10     8_193.18       0.7984             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_318.45       369.19     2_687.64       0.5316             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_318.45       460.42     2_778.87       0.5317             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_318.45       679.05     2_997.50       0.5317             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_318.45     2_250.35     4_568.79       0.4205             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_878.90       593.21     2_472.12       0.6716             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_878.90       726.33     2_605.23       0.6718             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_878.90     1_086.34     2_965.24       0.6718             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_878.90     3_602.50     5_481.40       0.6048             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_688.71     1_026.02     3_714.73       0.8350             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_688.71     1_282.08     3_970.80       0.8354             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_688.71     1_903.95     4_592.66       0.8354             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_688.71     6_368.80     9_057.51       0.8013             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_545.63       485.99     3_031.62       0.5298             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_545.63       542.26     3_087.90       0.5298             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_545.63       790.03     3_335.66       0.5298             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_545.63     2_587.34     5_132.97       0.4120             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_103.53       770.08     2_873.60       0.6736             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_103.53       859.89     2_963.42       0.6737             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_103.53     1_243.78     3_347.31       0.6738             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_103.53     4_149.19     6_252.72       0.6015             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_844.68     1_330.26     4_174.94       0.8371             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_844.68     1_491.08     4_335.76       0.8373             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_844.68     2_160.45     5_005.14       0.8373             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_844.68     7_199.61    10_044.29       0.8028             NaN         3.61
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.12     9_639.80     9_660.92       1.0000          1.0000        97.66
Exhaustive (self)                                         21.12    32_153.06    32_174.18       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_160.22       660.91     1_821.13       0.2164             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_160.22     2_190.75     3_350.97       0.1776             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_316.03     1_498.39     4_814.42       0.2862             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_316.03     4_997.18     8_313.21       0.2244             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_431.67     3_922.40     6_354.08       0.3812             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_431.67    13_083.91    15_515.58       0.3011             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_400.31       344.08     3_744.39       0.3740             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_400.31       570.23     3_970.54       0.3740             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_400.31       784.81     4_185.12       0.3740             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_400.31     2_641.23     6_041.54       0.2675             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_529.14       488.68     6_017.82       0.4850             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_529.14       795.98     6_325.12       0.4850             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_529.14     1_107.29     6_636.43       0.4850             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_529.14     3_690.36     9_219.50       0.3901             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_683.93       805.77     5_489.71       0.6268             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_683.93     1_287.58     5_971.51       0.6268             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_683.93     1_791.29     6_475.23       0.6268             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_683.93     5_939.30    10_623.23       0.5760             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_330.12       510.93     2_841.04       0.3699             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_330.12       650.09     2_980.20       0.3699             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_330.12       949.49     3_279.61       0.3699             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_330.12     3_145.87     5_475.99       0.2575             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_830.92       718.46     5_549.38       0.4847             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_830.92       906.90     5_737.81       0.4847             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_830.92     1_330.21     6_161.13       0.4847             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_830.92     4_421.88     9_252.80       0.3781             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_562.27     1_163.37     4_725.64       0.6276             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_562.27     1_447.02     5_009.30       0.6276             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_562.27     2_127.27     5_689.54       0.6276             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_562.27     7_087.66    10_649.94       0.5701             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_690.30       669.42     3_359.71       0.3692             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_690.30       757.56     3_447.86       0.3692             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_690.30     1_082.57     3_772.87       0.3692             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_690.30     3_640.50     6_330.79       0.2524             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_921.03       963.04     5_884.07       0.4838             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_921.03     1_058.11     5_979.14       0.4838             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_921.03     1_518.36     6_439.39       0.4838             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_921.03     5_081.96    10_002.99       0.3703             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_966.99     1_529.69     5_496.69       0.6309             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_966.99     1_714.91     5_681.90       0.6309             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_966.99     2_458.34     6_425.33       0.6309             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_966.99     8_213.80    12_180.79       0.5684             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Quantisation (stress) data

<details>
<summary><b>Cell embedding data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.62     1_752.58     1_757.20       1.0000          1.0000        24.41
Exhaustive (self)                                          4.62     5_855.31     5_859.93       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                614.49       642.97     1_257.46       0.8085             NaN         0.89
Exhaustive-PQ-m16 (self)                                 614.49     2_129.89     2_744.38       0.7445             NaN         0.89
Exhaustive-PQ-m32 (query)                                961.90     1_490.08     2_451.98       0.8682             NaN         1.65
Exhaustive-PQ-m32 (self)                                 961.90     4_951.59     5_913.49       0.8230             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_941.98     3_900.18     5_842.16       0.9196             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_941.98    13_071.82    15_013.81       0.8923             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_231.41       211.33     1_442.74       0.8726             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_231.41       370.17     1_601.58       0.8732             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_231.41       493.08     1_724.49       0.8733             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_231.41     1_609.02     2_840.43       0.8231             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_595.87       376.75     1_972.63       0.9185             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_595.87       635.89     2_231.77       0.9194             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_595.87       900.30     2_496.17       0.9195             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_595.87     2_955.00     4_550.88       0.8885             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_572.51       856.71     3_429.22       0.9585             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_572.51     1_457.67     4_030.18       0.9595             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_572.51     2_061.40     4_633.91       0.9597             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_572.51     6_858.16     9_430.67       0.9445             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            950.72       300.09     1_250.81       0.8772             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            950.72       383.07     1_333.79       0.8773             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            950.72       585.67     1_536.39       0.8773             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  950.72     1_876.67     2_827.39       0.8286             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_307.93       523.77     1_831.70       0.9226             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_307.93       668.39     1_976.32       0.9227             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_307.93       998.29     2_306.22       0.9227             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_307.93     3_330.93     4_638.86       0.8928             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_312.32     1_198.96     3_511.28       0.9612             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_312.32     1_536.88     3_849.19       0.9613             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_312.32     2_267.74     4_580.06       0.9614             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_312.32     7_567.01     9_879.33       0.9471             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_116.00       392.39     1_508.39       0.8790             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_116.00       445.71     1_561.71       0.8791             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_116.00       649.80     1_765.80       0.8791             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_116.00     2_152.80     3_268.80       0.8308             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_481.95       689.83     2_171.78       0.9230             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_481.95       773.93     2_255.87       0.9230             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_481.95     1_125.73     2_607.68       0.9231             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_481.95     3_751.33     5_233.27       0.8934             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_447.86     1_551.99     3_999.85       0.9623             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_447.86     1_747.68     4_195.54       0.9623             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_447.86     2_553.40     5_001.26       0.9624             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_447.86     8_520.47    10_968.33       0.9485             NaN         3.34
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.27     4_187.58     4_197.86       1.0000          1.0000        48.83
Exhaustive (self)                                         10.27    13_709.38    13_719.65       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_646.88       649.74     2_296.62       0.7118             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_646.88     2_157.47     3_804.35       0.6210             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_200.87     1_481.29     2_682.16       0.7717             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_200.87     4_933.47     6_134.34       0.6993             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_917.80     3_887.18     5_804.98       0.8251             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_917.80    13_025.89    14_943.69       0.7675             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_830.95       255.35     3_086.30       0.8272             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_830.95       419.71     3_250.66       0.8277             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_830.95       585.39     3_416.34       0.8277             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_830.95     1_963.12     4_794.07       0.7669             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_390.60       422.79     2_813.39       0.8746             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_390.60       759.00     3_149.60       0.8752             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_390.60       994.99     3_385.59       0.8752             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_390.60     3_324.92     5_715.52       0.8288             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_163.32       782.15     3_945.47       0.9048             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_163.32     1_330.14     4_493.46       0.9056             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_163.32     1_863.99     5_027.31       0.9056             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_163.32     6_238.76     9_402.08       0.8704             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_186.04       375.72     2_561.76       0.8424             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_186.04       478.18     2_664.23       0.8424             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_186.04       702.97     2_889.01       0.8424             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_186.04     2_398.69     4_584.73       0.7839             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_735.96       604.20     2_340.16       0.8833             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_735.96       770.25     2_506.21       0.8834             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_735.96     1_133.04     2_869.00       0.8834             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_735.96     3_791.57     5_527.53       0.8398             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_472.74     1_085.09     3_557.84       0.9099             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_472.74     1_373.04     3_845.78       0.9101             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_472.74     2_051.03     4_523.77       0.9101             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_472.74     6_801.37     9_274.12       0.8765             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_418.25       486.13     2_904.38       0.8500             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_418.25       554.09     2_972.34       0.8500             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_418.25       803.95     3_222.20       0.8500             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_418.25     2_699.81     5_118.06       0.7913             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_954.23       777.78     2_732.01       0.8870             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_954.23       887.46     2_841.69       0.8871             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_954.23     1_374.44     3_328.66       0.8871             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_954.23     4_295.94     6_250.17       0.8430             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_696.89     1_380.03     4_076.93       0.9128             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_696.89     1_567.23     4_264.13       0.9128             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_696.89     2_277.43     4_974.32       0.9129             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_696.89     7_559.09    10_255.99       0.8801             NaN         3.61
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.29     9_597.62     9_617.91       1.0000          1.0000        97.66
Exhaustive (self)                                         20.29    31_878.34    31_898.63       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_146.17       666.65     1_812.82       0.6791             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_146.17     2_196.64     3_342.81       0.5853             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_302.29     1_525.08     4_827.36       0.7374             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_302.29     4_993.88     8_296.16       0.6552             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_423.86     3_909.85     6_333.70       0.7805             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_423.86    13_241.24    15_665.10       0.7136             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_729.26       354.65     4_083.91       0.8455             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_729.26       601.15     4_330.41       0.8458             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_729.26       839.67     4_568.92       0.8458             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_729.26     2_814.62     6_543.88       0.7844             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_881.58       519.55     6_401.14       0.8726             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_881.58       874.53     6_756.12       0.8731             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_881.58     1_228.94     7_110.52       0.8731             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_881.58     4_116.06     9_997.65       0.8208             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_000.08       881.85     5_881.93       0.8936             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_000.08     1_487.00     6_487.09       0.8941             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_000.08     2_113.81     7_113.89       0.8941             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_000.08     6_917.72    11_917.80       0.8494             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_022.11       528.06     2_550.17       0.8549             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_022.11       655.14     2_677.24       0.8550             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_022.11       964.80     2_986.90       0.8550             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_022.11     3_201.00     5_223.11       0.7968             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_195.13       736.35     4_931.47       0.8808             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_195.13       933.69     5_128.82       0.8809             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_195.13     1_377.82     5_572.94       0.8809             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_195.13     4_599.55     8_794.67       0.8303             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_339.96     1_252.42     4_592.39       0.8994             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_339.96     1_556.29     4_896.25       0.8994             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_339.96     2_324.40     5_664.36       0.8995             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_339.96     7_661.02    11_000.98       0.8567             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_282.38       683.12     2_965.50       0.8705             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_282.38       764.45     3_046.83       0.8705             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_282.38     1_111.33     3_393.71       0.8705             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_282.38     3_714.65     5_997.04       0.8156             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_456.36     1_070.04     5_526.40       0.8917             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_456.36     1_090.54     5_546.90       0.8917             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_456.36     1_574.77     6_031.13       0.8917             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_456.36     5_281.32     9_737.68       0.8455             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_562.68     1_583.30     5_145.99       0.9064             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_562.68     1_785.87     5_348.56       0.9064             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_562.68     2_592.50     6_155.18       0.9064             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_562.68     8_659.12    12_221.80       0.8665             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Especially for the data with more internal structure, we can appreciate that
the Recalls reach ≥0.7 while providing a massive reduction in memory
fingerprint.

#### Optimised product quantisation (Exhaustive and IVF)

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

##### Why IVF massively outperforms Exhaustive OPQ

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

##### Correlated data

<details>
<summary><b>Correlated data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.45     1_694.52     1_698.97       1.0000          1.0000        24.41
Exhaustive (self)                                          4.45     5_713.12     5_717.57       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              4_242.06       299.84     4_541.89       0.2064             NaN         0.57
Exhaustive-OPQ-m8 (self)                               4_242.06     1_074.16     5_316.22       0.1974             NaN         0.57
Exhaustive-OPQ-m16 (query)                             2_911.94       641.48     3_553.43       0.2583             NaN         0.95
Exhaustive-OPQ-m16 (self)                              2_911.94     2_222.10     5_134.04       0.2223             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           4_771.26       133.57     4_904.83       0.3360             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          4_771.26       209.60     4_980.87       0.3360             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          4_771.26       292.54     5_063.80       0.3360             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                4_771.26     1_019.58     5_790.85       0.2467             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_482.85       210.11     3_692.95       0.4643             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_482.85       340.62     3_823.46       0.4643             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_482.85       479.41     3_962.26       0.4643             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_482.85     1_622.78     5_105.63       0.3734             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          4_579.49       191.28     4_770.77       0.3427             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          4_579.49       240.01     4_819.50       0.3427             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          4_579.49       341.35     4_920.84       0.3427             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                4_579.49     1_202.35     5_781.83       0.2516             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_262.36       303.07     3_565.43       0.4694             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_262.36       389.98     3_652.35       0.4694             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_262.36       585.06     3_847.42       0.4694             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_262.36     1_897.98     5_160.35       0.3782             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          4_654.13       251.45     4_905.58       0.3487             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          4_654.13       287.90     4_942.03       0.3487             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          4_654.13       408.18     5_062.31       0.3487             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                4_654.13     1_403.78     6_057.91       0.2549             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_370.45       409.64     3_780.09       0.4788             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_370.45       451.93     3_822.39       0.4788             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_370.45       651.90     4_022.35       0.4788             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_370.45     2_199.64     5_570.09       0.3837             NaN         1.11
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.76     4_197.97     4_207.73       1.0000          1.0000        48.83
Exhaustive (self)                                          9.76    14_148.70    14_158.47       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             8_522.07       649.94     9_172.01       0.1934             NaN         1.26
Exhaustive-OPQ-m16 (self)                              8_522.07     2_501.89    11_023.96       0.1801             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_253.38     1_483.86     7_737.24       0.2388             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_253.38     5_303.21    11_556.59       0.1982             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_683.07     3_896.62    13_579.68       0.3150             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_683.07    13_392.67    23_075.74       0.2571             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          9_698.15       244.78     9_942.93       0.3023             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         9_698.15       415.94    10_114.08       0.3023             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         9_698.15       585.89    10_284.03       0.3023             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               9_698.15     2_312.83    12_010.97       0.2155             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_477.86       405.94     7_883.80       0.4320             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_477.86       686.15     8_164.01       0.4321             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_477.86       971.55     8_449.41       0.4321             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_477.86     3_588.53    11_066.38       0.3438             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         10_726.04       727.13    11_453.17       0.6704             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        10_726.04     1_241.94    11_967.98       0.6709             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        10_726.04     1_757.95    12_483.99       0.6709             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              10_726.04     6_227.73    16_953.77       0.6112             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         9_180.71       369.67     9_550.38       0.3078             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         9_180.71       474.00     9_654.70       0.3078             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         9_180.71       707.27     9_887.97       0.3078             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               9_180.71     2_677.56    11_858.27       0.2192             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_825.20       604.16     7_429.36       0.4361             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_825.20       765.96     7_591.16       0.4361             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_825.20     1_151.24     7_976.44       0.4361             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_825.20     4_157.31    10_982.50       0.3461             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_223.50     1_071.86    11_295.36       0.6765             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_223.50     1_367.50    11_591.00       0.6765             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_223.50     2_052.13    12_275.63       0.6765             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_223.50     7_211.99    17_435.49       0.6165             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         9_365.59       501.94     9_867.54       0.3115             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         9_365.59       569.67     9_935.26       0.3115             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         9_365.59       832.75    10_198.35       0.3115             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               9_365.59     3_067.96    12_433.55       0.2213             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_190.12       793.87     7_983.99       0.4405             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_190.12       890.23     8_080.36       0.4406             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_190.12     1_309.59     8_499.71       0.4406             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_190.12     4_756.94    11_947.06       0.3493             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_473.28     1_382.60    11_855.87       0.6803             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_473.28     1_570.33    12_043.61       0.6803             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_473.28     2_317.46    12_790.74       0.6803             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_473.28     8_104.32    18_577.60       0.6197             NaN         3.86
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.28     9_556.47     9_576.75       1.0000          1.0000        97.66
Exhaustive (self)                                         20.28    32_293.21    32_313.48       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_120.17       661.40     7_781.58       0.1536             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_120.17     3_617.88    10_738.06       0.1519             NaN         2.26
Exhaustive-OPQ-m32 (query)                            17_548.45     1_497.82    19_046.27       0.1737             NaN         3.03
Exhaustive-OPQ-m32 (self)                             17_548.45     6_398.03    23_946.48       0.1581             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_410.98     3_913.16    16_324.14       0.2173             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_410.98    14_418.97    26_829.95       0.1750             NaN         4.55
Exhaustive-OPQ-m128 (query)                           18_376.49     9_084.68    27_461.17       0.2888             NaN         7.61
Exhaustive-OPQ-m128 (self)                            18_376.49    31_707.75    50_084.24       0.2357             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                          9_956.48       340.05    10_296.53       0.2139             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                         9_956.48       587.59    10_544.07       0.2139             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                         9_956.48       811.32    10_767.81       0.2139             NaN         2.57
IVF-OPQ-nl158-m16 (self)                               9_956.48     4_124.88    14_081.37       0.1540             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         21_110.59       497.02    21_607.60       0.2762             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        21_110.59       828.43    21_939.02       0.2762             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        21_110.59     1_168.56    22_279.15       0.2762             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              21_110.59     5_366.85    26_477.44       0.1915             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_100.37       840.10    15_940.46       0.4052             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_100.37     1_426.97    16_527.33       0.4053             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_100.37     2_005.86    17_106.22       0.4053             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_100.37     8_113.08    23_213.45       0.3197             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        21_092.30     1_493.54    22_585.84       0.6564             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       21_092.30     2_524.80    23_617.09       0.6567             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       21_092.30     3_565.02    24_657.32       0.6567             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             21_092.30    13_328.11    34_420.40       0.5968             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_143.91       512.93     8_656.84       0.2171             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_143.91       654.97     8_798.88       0.2171             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_143.91       966.35     9_110.26       0.2171             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_143.91     4_638.14    12_782.05       0.1564             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        18_605.37       730.56    19_335.93       0.2808             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        18_605.37       913.60    19_518.96       0.2808             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        18_605.37     1_354.16    19_959.53       0.2808             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              18_605.37     5_904.46    24_509.82       0.1940             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_484.81     1_199.21    14_684.02       0.4113             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_484.81     1_530.05    15_014.86       0.4113             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_484.81     2_275.83    15_760.64       0.4113             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_484.81     9_053.43    22_538.24       0.3234             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       19_402.86     2_162.60    21_565.45       0.6608             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       19_402.86     2_770.67    22_173.52       0.6609             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       19_402.86     4_161.12    23_563.98       0.6609             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             19_402.86    15_585.15    34_988.01       0.6004             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_451.48       696.79     9_148.28       0.2190             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_451.48       789.84     9_241.32       0.2190             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_451.48     1_135.60     9_587.08       0.2190             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_451.48     5_155.94    13_607.42       0.1576             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        19_176.72       948.65    20_125.36       0.2838             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        19_176.72     1_073.51    20_250.23       0.2838             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        19_176.72     1_563.15    20_739.87       0.2838             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              19_176.72     6_649.22    25_825.94       0.1957             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        13_769.06     1_563.32    15_332.38       0.4141             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        13_769.06     1_766.57    15_535.63       0.4141             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        13_769.06     2_599.64    16_368.70       0.4141             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              13_769.06    10_099.32    23_868.38       0.3252             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       19_695.60     2_765.36    22_460.96       0.6599             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       19_695.60     3_148.44    22_844.04       0.6600             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       19_695.60     4_646.42    24_342.02       0.6600             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             19_695.60    16_995.99    36_691.59       0.5998             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Lowrank data

<details>
<summary><b>Lowrank data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.48     1_768.19     1_772.67       1.0000          1.0000        24.41
Exhaustive (self)                                          4.48     5_985.32     5_989.80       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              4_252.40       301.20     4_553.60       0.3126             NaN         0.57
Exhaustive-OPQ-m8 (self)                               4_252.40     1_080.57     5_332.97       0.2242             NaN         0.57
Exhaustive-OPQ-m16 (query)                             2_937.45       643.12     3_580.57       0.4349             NaN         0.95
Exhaustive-OPQ-m16 (self)                              2_937.45     2_211.31     5_148.77       0.3213             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           4_789.28       126.69     4_915.97       0.6476             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          4_789.28       206.04     4_995.31       0.6476             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          4_789.28       290.55     5_079.82       0.6476             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                4_789.28     1_017.15     5_806.42       0.5965             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_501.05       199.58     3_700.63       0.7561             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_501.05       330.30     3_831.34       0.7561             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_501.05       451.08     3_952.13       0.7561             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_501.05     1_567.09     5_068.14       0.7265             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          4_655.34       190.54     4_845.88       0.6527             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          4_655.34       243.67     4_899.01       0.6527             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          4_655.34       364.04     5_019.38       0.6527             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                4_655.34     1_230.36     5_885.69       0.6038             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_347.82       310.42     3_658.24       0.7565             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_347.82       380.55     3_728.37       0.7566             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_347.82       576.40     3_924.22       0.7566             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_347.82     1_925.99     5_273.81       0.7297             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          4_750.14       247.41     4_997.55       0.6572             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          4_750.14       286.28     5_036.43       0.6572             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          4_750.14       419.23     5_169.37       0.6572             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                4_750.14     1_467.35     6_217.49       0.6085             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_489.22       400.32     3_889.53       0.7591             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_489.22       456.57     3_945.78       0.7591             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_489.22       671.53     4_160.74       0.7591             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_489.22     2_227.08     5_716.30       0.7320             NaN         1.11
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.79     4_210.43     4_220.23       1.0000          1.0000        48.83
Exhaustive (self)                                          9.79    13_950.73    13_960.53       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             8_524.44       649.82     9_174.26       0.2878             NaN         1.26
Exhaustive-OPQ-m16 (self)                              8_524.44     2_490.21    11_014.65       0.2016             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_260.76     1_483.62     7_744.38       0.4077             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_260.76     5_257.60    11_518.36       0.2913             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_672.20     3_958.49    13_630.70       0.5546             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_672.20    13_368.73    23_040.94       0.4616             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          9_600.54       237.85     9_838.40       0.6092             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         9_600.54       395.67     9_996.22       0.6092             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         9_600.54       551.52    10_152.07       0.6092             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               9_600.54     2_196.68    11_797.22       0.5571             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_350.11       394.20     7_744.31       0.7378             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_350.11       658.66     8_008.77       0.7379             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_350.11       897.19     8_247.30       0.7379             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_350.11     3_387.69    10_737.80       0.7040             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_451.53       714.42    12_165.95       0.8487             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_451.53     1_175.26    12_626.80       0.8488             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_451.53     1_609.44    13_060.98       0.8488             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_451.53     5_751.77    17_203.30       0.8226             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         9_344.00       365.23     9_709.22       0.6133             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         9_344.00       467.27     9_811.27       0.6134             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         9_344.00       695.37    10_039.37       0.6134             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               9_344.00     2_673.44    12_017.44       0.5650             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_945.44       596.13     7_541.58       0.7368             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_945.44       744.43     7_689.88       0.7370             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_945.44     1_108.96     8_054.40       0.7370             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_945.44     4_042.06    10_987.50       0.7065             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_357.50     1_042.25    11_399.75       0.8500             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_357.50     1_311.92    11_669.42       0.8503             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_357.50     1_957.32    12_314.82       0.8503             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_357.50     6_941.40    17_298.90       0.8258             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         9_543.40       484.97    10_028.37       0.6163             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         9_543.40       556.91    10_100.31       0.6163             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         9_543.40       810.71    10_354.11       0.6163             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               9_543.40     3_054.19    12_597.59       0.5677             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_209.60       790.92     8_000.52       0.7389             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_209.60       887.64     8_097.24       0.7390             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_209.60     1_287.06     8_496.66       0.7390             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_209.60     4_680.85    11_890.46       0.7090             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_579.01     1_374.05    11_953.06       0.8511             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_579.01     1_552.22    12_131.23       0.8513             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_579.01     2_254.94    12_833.95       0.8513             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_579.01     7_888.92    18_467.93       0.8270             NaN         3.86
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.96     9_503.16     9_523.12       1.0000          1.0000        97.66
Exhaustive (self)                                         19.96    32_733.82    32_753.78       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_603.97       663.63     8_267.60       0.1854             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_603.97     3_627.07    11_231.04       0.1630             NaN         2.26
Exhaustive-OPQ-m32 (query)                            17_610.89     1_509.31    19_120.20       0.2715             NaN         3.03
Exhaustive-OPQ-m32 (self)                             17_610.89     6_401.49    24_012.38       0.2139             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_577.84     3_913.37    16_491.21       0.3928             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_577.84    14_430.10    27_007.94       0.3001             NaN         4.55
Exhaustive-OPQ-m128 (query)                           18_472.65     9_096.04    27_568.69       0.5561             NaN         7.61
Exhaustive-OPQ-m128 (self)                            18_472.65    31_730.39    50_203.03       0.4604             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                          9_674.44       337.53    10_011.96       0.4397             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                         9_674.44       572.30    10_246.74       0.4397             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                         9_674.44       790.90    10_465.34       0.4397             NaN         2.57
IVF-OPQ-nl158-m16 (self)                               9_674.44     4_057.46    13_731.90       0.3671             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         20_088.12       480.09    20_568.21       0.5879             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        20_088.12       797.40    20_885.52       0.5879             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        20_088.12     1_106.83    21_194.95       0.5879             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              20_088.12     5_089.53    25_177.65       0.5307             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         14_783.44       807.73    15_591.17       0.7237             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        14_783.44     1_303.04    16_086.48       0.7237             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        14_783.44     1_813.98    16_597.42       0.7237             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              14_783.44     7_502.93    22_286.37       0.6826             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        20_599.21     1_441.51    22_040.71       0.8341             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       20_599.21     2_286.78    22_885.99       0.8341             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       20_599.21     3_175.28    23_774.49       0.8341             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             20_599.21    11_965.21    32_564.42       0.8069             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_293.74       519.14     8_812.88       0.4461             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_293.74       655.14     8_948.88       0.4461             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_293.74       960.52     9_254.26       0.4461             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_293.74     4_572.21    12_865.95       0.3745             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        18_810.16       715.58    19_525.74       0.5906             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        18_810.16       896.45    19_706.61       0.5906             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        18_810.16     1_342.14    20_152.30       0.5906             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              18_810.16     5_776.44    24_586.60       0.5338             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_636.73     1_167.26    14_803.99       0.7242             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_636.73     1_468.58    15_105.31       0.7242             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_636.73     2_145.21    15_781.94       0.7242             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_636.73     8_760.55    22_397.28       0.6856             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       19_547.85     2_093.27    21_641.12       0.8358             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       19_547.85     2_620.52    22_168.37       0.8358             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       19_547.85     3_864.07    23_411.92       0.8358             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             19_547.85    14_342.49    33_890.35       0.8104             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_674.96       684.44     9_359.40       0.4478             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_674.96       829.50     9_504.46       0.4478             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_674.96     1_117.92     9_792.88       0.4478             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_674.96     5_113.49    13_788.44       0.3767             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        19_167.17       946.13    20_113.30       0.5910             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        19_167.17     1_063.07    20_230.25       0.5910             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        19_167.17     1_529.40    20_696.58       0.5910             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              19_167.17     6_524.78    25_691.95       0.5372             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        13_997.26     1_538.43    15_535.69       0.7247             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        13_997.26     1_734.46    15_731.72       0.7247             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        13_997.26     2_504.93    16_502.19       0.7247             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              13_997.26     9_837.04    23_834.30       0.6865             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       19_875.57     2_734.12    22_609.69       0.8377             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       19_875.57     3_088.37    22_963.94       0.8377             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       19_875.57     4_473.86    24_349.43       0.8377             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             19_875.57    16_338.90    36_214.47       0.8114             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Quantisation (stress) data

<details>
<summary><b>Cell embedding data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.60     1_797.21     1_801.82       1.0000          1.0000        24.41
Exhaustive (self)                                          4.60     6_244.26     6_248.87       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              4_263.26       301.82     4_565.07       0.7867             NaN         0.57
Exhaustive-OPQ-m8 (self)                               4_263.26     1_087.64     5_350.90       0.7296             NaN         0.57
Exhaustive-OPQ-m16 (query)                             2_912.30       645.79     3_558.09       0.8536             NaN         0.95
Exhaustive-OPQ-m16 (self)                              2_912.30     2_211.99     5_124.28       0.8104             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           4_897.15       133.15     5_030.30       0.8495             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          4_897.15       217.49     5_114.64       0.8501             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          4_897.15       319.32     5_216.47       0.8501             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                4_897.15     1_092.59     5_989.74       0.8043             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_550.09       211.60     3_761.70       0.9058             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_550.09       362.45     3_912.54       0.9065             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_550.09       509.40     4_059.49       0.9066             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_550.09     1_738.99     5_289.08       0.8757             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          4_585.04       191.96     4_777.00       0.8531             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          4_585.04       250.56     4_835.60       0.8532             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          4_585.04       366.70     4_951.74       0.8532             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                4_585.04     1_271.72     5_856.76       0.8084             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_255.03       307.25     3_562.28       0.9088             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_255.03       395.93     3_650.96       0.9089             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_255.03       584.46     3_839.50       0.9089             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_255.03     2_004.83     5_259.87       0.8793             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          4_696.78       263.96     4_960.74       0.8564             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          4_696.78       296.68     4_993.46       0.8564             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          4_696.78       426.14     5_122.93       0.8565             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                4_696.78     1_463.67     6_160.45       0.8104             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_415.70       415.70     3_831.40       0.9089             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_415.70       469.08     3_884.78       0.9089             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_415.70       683.55     4_099.25       0.9090             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_415.70     2_299.14     5_714.84       0.8789             NaN         1.11
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.76     4_159.97     4_169.73       1.0000          1.0000        48.83
Exhaustive (self)                                          9.76    14_025.00    14_034.75       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             8_533.05       651.86     9_184.91       0.7808             NaN         1.26
Exhaustive-OPQ-m16 (self)                              8_533.05     2_500.16    11_033.22       0.7140             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_282.08     1_485.60     7_767.68       0.8259             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_282.08     5_270.48    11_552.56       0.7725             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_693.93     3_892.18    13_586.11       0.8530             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_693.93    13_347.81    23_041.74       0.8066             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          9_740.22       249.82     9_990.04       0.8764             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         9_740.22       427.79    10_168.01       0.8770             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         9_740.22       609.05    10_349.27       0.8770             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               9_740.22     2_360.66    12_100.88       0.8370             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_478.23       428.34     7_906.56       0.9041             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_478.23       725.30     8_203.53       0.9049             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_478.23     1_021.76     8_499.99       0.9049             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_478.23     3_765.09    11_243.31       0.8721             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         10_783.71       845.83    11_629.54       0.9212             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        10_783.71     1_358.37    12_142.09       0.9220             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        10_783.71     1_935.09    12_718.80       0.9220             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              10_783.71     6_808.00    17_591.71       0.8939             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         9_179.93       378.44     9_558.37       0.8776             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         9_179.93       485.42     9_665.34       0.8777             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         9_179.93       721.81     9_901.74       0.8777             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               9_179.93     2_775.96    11_955.89       0.8421             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_829.21       614.98     7_444.19       0.9034             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_829.21       781.93     7_611.14       0.9035             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_829.21     1_177.13     8_006.34       0.9036             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_829.21     4_303.17    11_132.38       0.8741             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_240.82     1_103.43    11_344.25       0.9225             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_240.82     1_407.77    11_648.59       0.9227             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_240.82     2_105.09    12_345.91       0.9227             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_240.82     7_509.26    17_750.08       0.8958             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         9_400.47       511.33     9_911.80       0.8759             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         9_400.47       569.85     9_970.32       0.8759             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         9_400.47       826.53    10_227.00       0.8760             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               9_400.47     3_105.17    12_505.64       0.8435             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_013.04       792.56     7_805.60       0.9014             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_013.04       905.12     7_918.17       0.9015             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_013.04     1_317.05     8_330.09       0.9015             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_013.04     4_795.65    11_808.69       0.8725             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_568.45     1_415.81    11_984.26       0.9220             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_568.45     1_608.23    12_176.68       0.9220             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_568.45     2_362.75    12_931.20       0.9221             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_568.45     8_252.70    18_821.14       0.8966             NaN         3.86
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.70     9_572.43     9_593.13       1.0000          1.0000        97.66
Exhaustive (self)                                         20.70    31_885.30    31_906.00       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_115.02       660.67     7_775.70       0.7453             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_115.02     3_914.01    11_029.04       0.6701             NaN         2.26
Exhaustive-OPQ-m32 (query)                            17_510.57     1_504.73    19_015.30       0.7984             NaN         3.03
Exhaustive-OPQ-m32 (self)                             17_510.57     6_426.00    23_936.57       0.7383             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_454.20     3_900.70    16_354.90       0.8363             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_454.20    14_414.31    26_868.52       0.7874             NaN         4.55
Exhaustive-OPQ-m128 (query)                           18_346.78     9_129.65    27_476.43       0.9166             NaN         7.61
Exhaustive-OPQ-m128 (self)                            18_346.78    31_671.22    50_018.00       0.8913             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_000.51       357.36    10_357.87       0.8653             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_000.51       603.77    10_604.28       0.8656             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_000.51       838.70    10_839.21       0.8656             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_000.51     4_211.74    14_212.25       0.8283             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         20_178.21       527.18    20_705.39       0.8826             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        20_178.21       868.33    21_046.53       0.8831             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        20_178.21     1_215.67    21_393.88       0.8831             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              20_178.21     5_500.64    25_678.84       0.8488             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_164.33       887.94    16_052.27       0.8944             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_164.33     1_507.65    16_671.98       0.8948             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_164.33     2_123.02    17_287.34       0.8949             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_164.33     8_536.95    23_701.27       0.8640             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        20_925.30     1_604.37    22_529.67       0.9512             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       20_925.30     2_737.43    23_662.73       0.9518             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       20_925.30     3_845.68    24_770.98       0.9518             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             20_925.30    14_353.93    35_279.23       0.9376             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         7_998.34       543.90     8_542.23       0.8746             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         7_998.34       653.42     8_651.75       0.8746             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         7_998.34       970.31     8_968.64       0.8746             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               7_998.34     4_669.27    12_667.61       0.8387             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        18_512.49       737.07    19_249.56       0.8899             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        18_512.49       921.64    19_434.12       0.8899             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        18_512.49     1_370.25    19_882.74       0.8900             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              18_512.49     5_981.76    24_494.24       0.8570             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_366.40     1_230.63    14_597.04       0.9024             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_366.40     1_570.28    14_936.68       0.9024             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_366.40     2_337.64    15_704.04       0.9025             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_366.40     9_281.46    22_647.86       0.8725             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       19_275.94     2_239.00    21_514.95       0.9545             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       19_275.94     2_852.89    22_128.83       0.9546             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       19_275.94     4_270.84    23_546.79       0.9546             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             19_275.94    15_701.14    34_977.08       0.9406             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_232.54       701.26     8_933.80       0.8873             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_232.54       801.77     9_034.31       0.8874             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_232.54     1_145.09     9_377.63       0.8874             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_232.54     5_200.49    13_433.04       0.8516             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        20_370.52       963.38    21_333.90       0.9005             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        20_370.52     1_090.04    21_460.56       0.9005             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        20_370.52     1_584.79    21_955.31       0.9005             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              20_370.52     6_684.84    27_055.36       0.8675             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        13_628.31     1_597.06    15_225.37       0.9098             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        13_628.31     1_822.29    15_450.60       0.9098             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        13_628.31     2_646.86    16_275.17       0.9098             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              13_628.31    10_307.82    23_936.12       0.8800             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       19_661.35     2_846.69    22_508.04       0.9594             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       19_661.35     3_250.74    22_912.09       0.9594             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       19_661.35     4_817.65    24_479.01       0.9594             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             19_661.35    17_396.91    37_058.27       0.9451             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

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
