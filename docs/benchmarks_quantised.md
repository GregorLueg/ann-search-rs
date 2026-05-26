## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger
print. If you wish to run the examples, below is how you launch it.

**BF16:**

```bash
cargo run --example gridsearch_bf16 --release --features quantised
```

**SQ8:**

```bash
cargo run --example gridsearch_sq8 --release --features quantised
```

**Product quantisation (PQ):**

```bash
cargo run --example gridsearch_pq --release --features quantised -- --dim 512 --data embedding --n-samples 50000
```

**Optimised product quantisation (OPQ):**

```bash
cargo run --example gridsearch_opq --release --features quantised -- --dim 512 --data embedding --n-samples 50000
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
Exhaustive (query)                                         3.09     1_538.79     1_541.88       1.0000          1.0000        18.31
Exhaustive (self)                                          3.09    16_224.48    16_227.56       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.13     1_248.62     1_253.76       0.9828             NaN         9.16
Exhaustive-BF16 (self)                                     5.13    16_698.98    16_704.11       1.0000             NaN         9.16
IVF-BF16-nl273-np13 (query)                              392.28        89.18       481.45       0.9806             NaN         9.19
IVF-BF16-nl273-np16 (query)                              392.28       103.60       495.87       0.9825             NaN         9.19
IVF-BF16-nl273-np23 (query)                              392.28       141.90       534.17       0.9828             NaN         9.19
IVF-BF16-nl273 (self)                                    392.28     1_407.89     1_800.16       0.9798             NaN         9.19
IVF-BF16-nl387-np19 (query)                              749.22        94.04       843.27       0.9821             NaN         9.21
IVF-BF16-nl387-np27 (query)                              749.22       121.59       870.82       0.9828             NaN         9.21
IVF-BF16-nl387 (self)                                    749.22     1_220.58     1_969.81       0.9798             NaN         9.21
IVF-BF16-nl547-np23 (query)                            1_449.97       101.53     1_551.50       0.9772             NaN         9.23
IVF-BF16-nl547-np27 (query)                            1_449.97        96.49     1_546.46       0.9815             NaN         9.23
IVF-BF16-nl547-np33 (query)                            1_449.97       112.71     1_562.67       0.9828             NaN         9.23
IVF-BF16-nl547 (self)                                  1_449.97     1_141.70     2_591.67       0.9798             NaN         9.23
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
Exhaustive (query)                                         3.86     1_570.50     1_574.36       1.0000          1.0000        18.88
Exhaustive (self)                                          3.86    15_889.32    15_893.18       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                    5.73     1_217.23     1_222.96       0.8870             NaN         9.44
Exhaustive-BF16 (self)                                     5.73    15_556.46    15_562.20       1.0000             NaN         9.44
IVF-BF16-nl273-np13 (query)                              371.99        93.38       465.37       0.8860             NaN         9.48
IVF-BF16-nl273-np16 (query)                              371.99       108.50       480.49       0.8870             NaN         9.48
IVF-BF16-nl273-np23 (query)                              371.99       156.99       528.98       0.8870             NaN         9.48
IVF-BF16-nl273 (self)                                    371.99     1_520.14     1_892.13       0.8852             NaN         9.48
IVF-BF16-nl387-np19 (query)                              715.72        97.18       812.90       0.8867             NaN         9.49
IVF-BF16-nl387-np27 (query)                              715.72       128.21       843.94       0.8870             NaN         9.49
IVF-BF16-nl387 (self)                                    715.72     1_295.74     2_011.47       0.8852             NaN         9.49
IVF-BF16-nl547-np23 (query)                            1_383.88        88.89     1_472.77       0.8849             NaN         9.51
IVF-BF16-nl547-np27 (query)                            1_383.88       101.30     1_485.18       0.8866             NaN         9.51
IVF-BF16-nl547-np33 (query)                            1_383.88       118.78     1_502.66       0.8870             NaN         9.51
IVF-BF16-nl547 (self)                                  1_383.88     1_197.80     2_581.68       0.8852             NaN         9.51
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
Exhaustive (query)                                         3.15     1_537.76     1_540.91       1.0000          1.0000        18.31
Exhaustive (self)                                          3.15    15_553.45    15_556.60       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.17     1_176.06     1_181.23       0.9223             NaN         9.16
Exhaustive-BF16 (self)                                     5.17    15_576.99    15_582.16       1.0000             NaN         9.16
IVF-BF16-nl273-np13 (query)                              371.50        91.33       462.83       0.9223             NaN         9.19
IVF-BF16-nl273-np16 (query)                              371.50       107.12       478.62       0.9223             NaN         9.19
IVF-BF16-nl273-np23 (query)                              371.50       148.73       520.23       0.9223             NaN         9.19
IVF-BF16-nl273 (self)                                    371.50     1_477.99     1_849.49       0.9031             NaN         9.19
IVF-BF16-nl387-np19 (query)                              715.85        92.38       808.23       0.9223             NaN         9.21
IVF-BF16-nl387-np27 (query)                              715.85       125.92       841.78       0.9223             NaN         9.21
IVF-BF16-nl387 (self)                                    715.85     1_226.70     1_942.56       0.9031             NaN         9.21
IVF-BF16-nl547-np23 (query)                            1_379.51        83.14     1_462.65       0.9223             NaN         9.23
IVF-BF16-nl547-np27 (query)                            1_379.51        95.22     1_474.73       0.9223             NaN         9.23
IVF-BF16-nl547-np33 (query)                            1_379.51       111.64     1_491.15       0.9223             NaN         9.23
IVF-BF16-nl547 (self)                                  1_379.51     1_091.81     2_471.32       0.9031             NaN         9.23
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
Exhaustive (query)                                         3.02     1_523.30     1_526.32       1.0000          1.0000        18.31
Exhaustive (self)                                          3.02    15_832.57    15_835.60       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.54     1_250.01     1_254.55       0.9516             NaN         9.16
Exhaustive-BF16 (self)                                     4.54    16_347.73    16_352.27       1.0000             NaN         9.16
IVF-BF16-nl273-np13 (query)                              509.58        96.97       606.55       0.9516             NaN         9.19
IVF-BF16-nl273-np16 (query)                              509.58        93.25       602.83       0.9516             NaN         9.19
IVF-BF16-nl273-np23 (query)                              509.58       138.94       648.52       0.9516             NaN         9.19
IVF-BF16-nl273 (self)                                    509.58     1_250.05     1_759.63       0.9405             NaN         9.19
IVF-BF16-nl387-np19 (query)                              724.07        82.77       806.84       0.9516             NaN         9.21
IVF-BF16-nl387-np27 (query)                              724.07       106.64       830.72       0.9516             NaN         9.21
IVF-BF16-nl387 (self)                                    724.07     1_062.40     1_786.47       0.9405             NaN         9.21
IVF-BF16-nl547-np23 (query)                            1_383.03        80.01     1_463.04       0.9516             NaN         9.23
IVF-BF16-nl547-np27 (query)                            1_383.03        86.05     1_469.08       0.9516             NaN         9.23
IVF-BF16-nl547-np33 (query)                            1_383.03        97.99     1_481.02       0.9516             NaN         9.23
IVF-BF16-nl547 (self)                                  1_383.03       975.82     2_358.85       0.9405             NaN         9.23
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
Exhaustive (query)                                        14.25     5_841.48     5_855.73       1.0000          1.0000        73.24
Exhaustive (self)                                         14.25    57_968.84    57_983.09       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   22.15     4_977.16     4_999.30       0.9717             NaN        36.62
Exhaustive-BF16 (self)                                    22.15    58_113.37    58_135.51       1.0000             NaN        36.62
IVF-BF16-nl273-np13 (query)                              679.19       261.96       941.15       0.9717             NaN        36.76
IVF-BF16-nl273-np16 (query)                              679.19       299.03       978.21       0.9717             NaN        36.76
IVF-BF16-nl273-np23 (query)                              679.19       417.08     1_096.26       0.9717             NaN        36.76
IVF-BF16-nl273 (self)                                    679.19     4_308.59     4_987.78       0.9674             NaN        36.76
IVF-BF16-nl387-np19 (query)                            1_212.54       272.69     1_485.24       0.9717             NaN        36.81
IVF-BF16-nl387-np27 (query)                            1_212.54       359.73     1_572.27       0.9717             NaN        36.81
IVF-BF16-nl387 (self)                                  1_212.54     3_662.06     4_874.61       0.9674             NaN        36.81
IVF-BF16-nl547-np23 (query)                            2_466.87       260.87     2_727.74       0.9717             NaN        36.89
IVF-BF16-nl547-np27 (query)                            2_466.87       287.43     2_754.30       0.9717             NaN        36.89
IVF-BF16-nl547-np33 (query)                            2_466.87       328.49     2_795.36       0.9717             NaN        36.89
IVF-BF16-nl547 (self)                                  2_466.87     3_323.71     5_790.58       0.9674             NaN        36.89
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
Exhaustive (query)                                         3.13     1_426.27     1_429.40       1.0000          1.0000        18.31
Exhaustive (self)                                          3.13    14_633.98    14_637.11       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.84       710.46       717.30       0.7939             NaN         4.58
Exhaustive-SQ8 (self)                                      6.84     7_337.67     7_344.51       0.7931             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               369.90        49.93       419.83       0.7862             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               369.90        60.48       430.38       0.7871             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               369.90        76.73       446.63       0.7872             NaN         4.61
IVF-SQ8-nl273 (self)                                     369.90       770.55     1_140.46       0.7862             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               715.89        53.07       768.96       0.7965             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               715.89        67.81       783.69       0.7968             NaN         4.63
IVF-SQ8-nl387 (self)                                     715.89       683.93     1_399.82       0.7961             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_384.49        51.26     1_435.75       0.7919             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_384.49        60.29     1_444.78       0.7936             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_384.49        64.91     1_449.40       0.7940             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_384.49       651.73     2_036.22       0.7931             NaN         4.65
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
Exhaustive (query)                                         3.86     1_503.08     1_506.95       1.0000          1.0000        18.88
Exhaustive (self)                                          3.86    15_315.80    15_319.66       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                     7.17       978.56       985.73       0.8273             NaN         5.15
Exhaustive-SQ8 (self)                                      7.17     9_785.78     9_792.95       0.8260             NaN         5.15
IVF-SQ8-nl273-np13 (query)                               367.60        62.46       430.06       0.8256             NaN         5.19
IVF-SQ8-nl273-np16 (query)                               367.60        72.34       439.94       0.8262             NaN         5.19
IVF-SQ8-nl273-np23 (query)                               367.60        99.05       466.64       0.8263             NaN         5.19
IVF-SQ8-nl273 (self)                                     367.60       981.89     1_349.49       0.8252             NaN         5.19
IVF-SQ8-nl387-np19 (query)                               704.85        66.10       770.95       0.8269             NaN         5.20
IVF-SQ8-nl387-np27 (query)                               704.85        88.01       792.86       0.8272             NaN         5.20
IVF-SQ8-nl387 (self)                                     704.85       857.83     1_562.68       0.8264             NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_361.79        61.80     1_423.59       0.8259             NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_361.79        69.55     1_431.34       0.8275             NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_361.79        80.22     1_442.01       0.8278             NaN         5.22
IVF-SQ8-nl547 (self)                                   1_361.79       798.20     2_159.99       0.8267             NaN         5.22
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
Exhaustive (query)                                         3.20     1_467.14     1_470.34       1.0000          1.0000        18.31
Exhaustive (self)                                          3.20    14_741.37    14_744.57       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.69       708.74       715.42       0.7705             NaN         4.58
Exhaustive-SQ8 (self)                                      6.69     7_857.40     7_864.09       0.7670             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               368.96        51.69       420.65       0.7717             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               368.96        61.26       430.22       0.7716             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               368.96        82.57       451.53       0.7716             NaN         4.61
IVF-SQ8-nl273 (self)                                     368.96       833.07     1_202.03       0.7688             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               704.43        53.50       757.93       0.7715             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               704.43        72.29       776.72       0.7715             NaN         4.63
IVF-SQ8-nl387 (self)                                     704.43       706.00     1_410.43       0.7684             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_365.71        49.69     1_415.40       0.7709             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_365.71        54.99     1_420.70       0.7709             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_365.71        65.70     1_431.41       0.7709             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_365.71       638.09     2_003.80       0.7672             NaN         4.65
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
Exhaustive (query)                                         3.02     1_562.15     1_565.17       1.0000          1.0000        18.31
Exhaustive (self)                                          3.02    14_731.73    14_734.75       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.93       732.05       738.98       0.7050             NaN         4.58
Exhaustive-SQ8 (self)                                      6.93     7_324.40     7_331.32       0.7116             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               369.97        46.59       416.57       0.7055             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               369.97        52.05       422.02       0.7056             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               369.97        69.06       439.03       0.7055             NaN         4.61
IVF-SQ8-nl273 (self)                                     369.97       694.40     1_064.38       0.7124             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               709.18        49.40       758.58       0.7057             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               709.18        61.91       771.09       0.7056             NaN         4.63
IVF-SQ8-nl387 (self)                                     709.18       624.63     1_333.81       0.7122             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_367.30        47.26     1_414.57       0.7050             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_367.30        52.44     1_419.75       0.7050             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_367.30        57.87     1_425.17       0.7049             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_367.30       592.27     1_959.57       0.7117             NaN         4.65
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
Exhaustive (query)                                        14.73     5_676.24     5_690.97       1.0000          1.0000        73.24
Exhaustive (self)                                         14.73    57_541.48    57_556.21       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    39.02     1_667.90     1_706.92       0.7859             NaN        18.31
Exhaustive-SQ8 (self)                                     39.02    16_880.06    16_919.08       0.8138             NaN        18.31
IVF-SQ8-nl273-np13 (query)                               637.69        92.86       730.55       0.7872             NaN        18.45
IVF-SQ8-nl273-np16 (query)                               637.69       104.31       742.00       0.7872             NaN        18.45
IVF-SQ8-nl273-np23 (query)                               637.69       141.30       778.99       0.7872             NaN        18.45
IVF-SQ8-nl273 (self)                                     637.69     1_341.59     1_979.28       0.8139             NaN        18.45
IVF-SQ8-nl387-np19 (query)                             1_261.61        99.55     1_361.16       0.7873             NaN        18.50
IVF-SQ8-nl387-np27 (query)                             1_261.61       125.90     1_387.51       0.7873             NaN        18.50
IVF-SQ8-nl387 (self)                                   1_261.61     1_190.85     2_452.46       0.8140             NaN        18.50
IVF-SQ8-nl547-np23 (query)                             2_553.87        99.93     2_653.80       0.7864             NaN        18.58
IVF-SQ8-nl547-np27 (query)                             2_553.87       107.54     2_661.40       0.7864             NaN        18.58
IVF-SQ8-nl547-np33 (query)                             2_553.87       120.32     2_674.18       0.7864             NaN        18.58
IVF-SQ8-nl547 (self)                                   2_553.87     1_149.01     3_702.88       0.8139             NaN        18.58
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

Let's start with correlated data.

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.72     4_055.08     4_064.80       1.0000          1.0000        48.83
Exhaustive (self)                                          9.72    13_774.45    13_784.18       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_634.76       648.85     2_283.61       0.1810             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_634.76     2_151.70     3_786.46       0.1575             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_193.53     1_478.77     2_672.29       0.2138             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_193.53     4_928.44     6_121.97       0.1773             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_921.48     3_889.93     5_811.41       0.2946             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_921.48    13_015.30    14_936.79       0.2455             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_801.18       243.75     3_044.93       0.3002             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_801.18       407.68     3_208.86       0.3003             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_801.18       564.80     3_365.97       0.3003             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_801.18     1_901.15     4_702.32       0.2095             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_387.05       399.95     2_786.99       0.4314             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_387.05       670.53     3_057.57       0.4316             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_387.05       941.99     3_329.03       0.4316             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_387.05     3_177.18     5_564.23       0.3423             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_122.55       720.83     3_843.37       0.6710             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_122.55     1_222.76     4_345.31       0.6714             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_122.55     1_722.72     4_845.27       0.6714             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_122.55     5_793.91     8_916.45       0.6119             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_250.46       377.97     2_628.43       0.3050             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_250.46       471.15     2_721.61       0.3050             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_250.46       701.05     2_951.52       0.3050             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_250.46     2_335.90     4_586.36       0.2137             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_733.33       594.78     2_328.11       0.4357             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_733.33       749.32     2_482.64       0.4357             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_733.33     1_121.31     2_854.64       0.4357             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_733.33     3_773.62     5_506.95       0.3448             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_462.36     1_047.87     3_510.23       0.6756             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_462.36     1_324.46     3_786.82       0.6756             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_462.36     1_982.24     4_444.60       0.6756             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_462.36     6_614.28     9_076.64       0.6158             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_476.76       486.37     2_963.13       0.3102             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_476.76       563.35     3_040.11       0.3102             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_476.76       794.53     3_271.28       0.3102             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_476.76     2_619.85     5_096.60       0.2160             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_012.95       767.64     2_780.59       0.4406             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_012.95       863.85     2_876.81       0.4406             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_012.95     1_271.31     3_284.27       0.4406             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_012.95     4_541.60     6_554.55       0.3485             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_742.40     1_348.36     4_090.76       0.6786             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_742.40     1_521.50     4_263.90       0.6786             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_742.40     2_237.59     4_979.99       0.6786             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_742.40     7_468.25    10_210.66       0.6183             NaN         3.61
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
Exhaustive (query)                                        20.24     9_594.11     9_614.35       1.0000          1.0000        97.66
Exhaustive (self)                                         20.24    32_093.91    32_114.15       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_145.52       657.85     1_803.37       0.1437             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_145.52     2_344.52     3_490.04       0.1309             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_286.43     1_614.61     4_901.04       0.1563             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_286.43     4_990.26     8_276.68       0.1378             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_404.79     3_903.67     6_308.46       0.1894             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_404.79    13_525.55    15_930.34       0.1563             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           4_212.39       363.97     4_576.36       0.2147             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          4_212.39       597.86     4_810.24       0.2147             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          4_212.39       831.88     5_044.27       0.2147             NaN         1.57
IVF-PQ-nl158-m16 (self)                                4_212.39     2_750.28     6_962.67       0.1475             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           6_143.82       497.85     6_641.67       0.2772             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          6_143.82       836.58     6_980.40       0.2772             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          6_143.82     1_173.89     7_317.71       0.2772             NaN         2.34
IVF-PQ-nl158-m32 (self)                                6_143.82     3_939.62    10_083.44       0.1883             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_968.80       830.06     5_798.86       0.4068             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_968.80     1_392.31     6_361.10       0.4069             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_968.80     1_948.27     6_917.07       0.4069             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_968.80     6_489.95    11_458.74       0.3186             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_138.74       520.46     2_659.20       0.2184             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_138.74       658.04     2_796.79       0.2184             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_138.74       971.69     3_110.43       0.2184             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_138.74     3_218.27     5_357.01       0.1501             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_290.90       725.78     5_016.68       0.2817             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_290.90       919.98     5_210.88       0.2817             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_290.90     1_381.85     5_672.75       0.2817             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_290.90     4_532.21     8_823.11       0.1912             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_412.63     1_197.22     4_609.85       0.4136             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_412.63     1_509.39     4_922.02       0.4136             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_412.63     2_248.07     5_660.70       0.4136             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_412.63     7_500.77    10_913.40       0.3225             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_498.80       671.39     3_170.20       0.2211             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_498.80       764.59     3_263.40       0.2211             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_498.80     1_105.82     3_604.63       0.2211             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_498.80     3_638.24     6_137.04       0.1518             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_677.43       952.57     5_630.00       0.2852             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_677.43     1_067.20     5_744.63       0.2852             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_677.43     1_550.48     6_227.91       0.2852             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_677.43     5_561.42    10_238.85       0.1932             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_758.81     1_580.13     5_338.94       0.4162             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_758.81     1_755.04     5_513.85       0.4162             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_758.81     2_555.88     6_314.70       0.4162             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_758.81     8_529.41    12_288.22       0.3254             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.05    15_932.39    15_965.44       1.0000          1.0000       146.48
Exhaustive (self)                                         33.05    52_551.97    52_585.03       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_467.05       671.09     2_138.14       0.1331             NaN         1.51
Exhaustive-PQ-m16 (self)                               1_467.05     2_230.19     3_697.24       0.1233             NaN         1.51
Exhaustive-PQ-m32 (query)                              2_083.17     1_509.17     3_592.34       0.1397             NaN         2.28
Exhaustive-PQ-m32 (self)                               2_083.17     5_024.11     7_107.28       0.1268             NaN         2.28
Exhaustive-PQ-m64 (query)                              3_151.71     3_974.46     7_126.17       0.1594             NaN         3.80
Exhaustive-PQ-m64 (self)                               3_151.71    13_142.46    16_294.17       0.1362             NaN         3.80
Exhaustive-PQ-m128 (query)                             4_967.88     9_145.63    14_113.51       0.2067             NaN         6.86
Exhaustive-PQ-m128 (self)                              4_967.88    30_473.00    35_440.88       0.1674             NaN         6.86
IVF-PQ-nl158-m16-np7 (query)                           5_388.57       450.44     5_839.02       0.1822             NaN         1.98
IVF-PQ-nl158-m16-np12 (query)                          5_388.57       750.19     6_138.76       0.1822             NaN         1.98
IVF-PQ-nl158-m16-np17 (query)                          5_388.57     1_052.42     6_441.00       0.1822             NaN         1.98
IVF-PQ-nl158-m16 (self)                                5_388.57     3_524.55     8_913.13       0.1308             NaN         1.98
IVF-PQ-nl158-m32-np7 (query)                           6_035.49       606.39     6_641.87       0.2278             NaN         2.74
IVF-PQ-nl158-m32-np12 (query)                          6_035.49     1_010.48     7_045.97       0.2278             NaN         2.74
IVF-PQ-nl158-m32-np17 (query)                          6_035.49     1_417.96     7_453.44       0.2278             NaN         2.74
IVF-PQ-nl158-m32 (self)                                6_035.49     4_755.12    10_790.60       0.1523             NaN         2.74
IVF-PQ-nl158-m64-np7 (query)                           7_071.74       925.31     7_997.04       0.3080             NaN         4.27
IVF-PQ-nl158-m64-np12 (query)                          7_071.74     1_547.87     8_619.61       0.3080             NaN         4.27
IVF-PQ-nl158-m64-np17 (query)                          7_071.74     2_152.16     9_223.90       0.3080             NaN         4.27
IVF-PQ-nl158-m64 (self)                                7_071.74     7_142.89    14_214.63       0.2164             NaN         4.27
IVF-PQ-nl158-m128-np7 (query)                          8_866.98     1_893.53    10_760.50       0.4926             NaN         7.32
IVF-PQ-nl158-m128-np12 (query)                         8_866.98     3_212.97    12_079.95       0.4928             NaN         7.32
IVF-PQ-nl158-m128-np17 (query)                         8_866.98     4_550.11    13_417.09       0.4928             NaN         7.32
IVF-PQ-nl158-m128 (self)                               8_866.98    14_958.02    23_825.00       0.4148             NaN         7.32
IVF-PQ-nl223-m16-np11 (query)                          3_011.27       643.93     3_655.20       0.1848             NaN         2.17
IVF-PQ-nl223-m16-np14 (query)                          3_011.27       803.77     3_815.04       0.1848             NaN         2.17
IVF-PQ-nl223-m16-np21 (query)                          3_011.27     1_178.95     4_190.22       0.1848             NaN         2.17
IVF-PQ-nl223-m16 (self)                                3_011.27     3_929.92     6_941.19       0.1332             NaN         2.17
IVF-PQ-nl223-m32-np11 (query)                          3_664.50       873.26     4_537.76       0.2309             NaN         2.93
IVF-PQ-nl223-m32-np14 (query)                          3_664.50     1_082.58     4_747.08       0.2309             NaN         2.93
IVF-PQ-nl223-m32-np21 (query)                          3_664.50     1_696.76     5_361.26       0.2309             NaN         2.93
IVF-PQ-nl223-m32 (self)                                3_664.50     5_283.64     8_948.13       0.1542             NaN         2.93
IVF-PQ-nl223-m64-np11 (query)                          4_731.77     1_318.28     6_050.06       0.3111             NaN         4.46
IVF-PQ-nl223-m64-np14 (query)                          4_731.77     1_654.73     6_386.50       0.3111             NaN         4.46
IVF-PQ-nl223-m64-np21 (query)                          4_731.77     2_438.08     7_169.86       0.3111             NaN         4.46
IVF-PQ-nl223-m64 (self)                                4_731.77     8_125.93    12_857.71       0.2193             NaN         4.46
IVF-PQ-nl223-m128-np11 (query)                         6_598.67     2_818.88     9_417.55       0.4976             NaN         7.51
IVF-PQ-nl223-m128-np14 (query)                         6_598.67     3_509.27    10_107.94       0.4977             NaN         7.51
IVF-PQ-nl223-m128-np21 (query)                         6_598.67     5_201.77    11_800.45       0.4977             NaN         7.51
IVF-PQ-nl223-m128 (self)                               6_598.67    17_516.04    24_114.71       0.4167             NaN         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_517.30       861.57     4_378.87       0.1879             NaN         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_517.30       983.72     4_501.03       0.1879             NaN         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_517.30     1_392.54     4_909.84       0.1879             NaN         2.44
IVF-PQ-nl316-m16 (self)                                3_517.30     4_639.78     8_157.08       0.1343             NaN         2.44
IVF-PQ-nl316-m32-np15 (query)                          4_184.93     1_142.49     5_327.43       0.2341             NaN         3.21
IVF-PQ-nl316-m32-np17 (query)                          4_184.93     1_283.60     5_468.54       0.2341             NaN         3.21
IVF-PQ-nl316-m32-np25 (query)                          4_184.93     1_849.92     6_034.85       0.2341             NaN         3.21
IVF-PQ-nl316-m32 (self)                                4_184.93     6_184.16    10_369.09       0.1555             NaN         3.21
IVF-PQ-nl316-m64-np15 (query)                          5_221.35     1_765.36     6_986.71       0.3174             NaN         4.73
IVF-PQ-nl316-m64-np17 (query)                          5_221.35     1_965.55     7_186.90       0.3174             NaN         4.73
IVF-PQ-nl316-m64-np25 (query)                          5_221.35     2_810.64     8_031.99       0.3174             NaN         4.73
IVF-PQ-nl316-m64 (self)                                5_221.35     9_358.18    14_579.53       0.2219             NaN         4.73
IVF-PQ-nl316-m128-np15 (query)                         7_252.16     3_657.82    10_909.98       0.5018             NaN         7.78
IVF-PQ-nl316-m128-np17 (query)                         7_252.16     4_112.56    11_364.72       0.5018             NaN         7.78
IVF-PQ-nl316-m128-np25 (query)                         7_252.16     5_966.46    13_218.63       0.5018             NaN         7.78
IVF-PQ-nl316-m128 (self)                               7_252.16    19_994.86    27_247.03       0.4200             NaN         7.78
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Lowrank data

Data where the structure resides on a lower-dimensional manifold.

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.85     4_174.20     4_184.05       1.0000          1.0000        48.83
Exhaustive (self)                                          9.85    13_733.57    13_743.42       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_644.46       649.81     2_294.27       0.2962             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_644.46     2_159.56     3_804.02       0.2324             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_197.81     1_483.42     2_681.23       0.4045             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_197.81     4_951.61     6_149.42       0.3204             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_919.10     3_886.81     5_805.91       0.5368             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_919.10    12_964.33    14_883.43       0.4607             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_697.57       241.53     2_939.10       0.5293             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_697.57       389.64     3_087.21       0.5293             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_697.57       541.25     3_238.82       0.5293             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_697.57     1_816.91     4_514.47       0.4281             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_292.53       388.89     2_681.42       0.6699             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_292.53       637.20     2_929.73       0.6699             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_292.53       870.88     3_163.42       0.6699             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_292.53     2_933.41     5_225.95       0.6073             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_012.06       699.47     3_711.52       0.8316             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_012.06     1_133.05     4_145.11       0.8316             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_012.06     1_574.58     4_586.64       0.8316             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_012.06     5_295.85     8_307.91       0.7984             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_319.82       369.08     2_688.91       0.5316             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_319.82       461.03     2_780.85       0.5317             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_319.82       704.23     3_024.05       0.5317             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_319.82     2_266.63     4_586.45       0.4205             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_886.12       580.30     2_466.42       0.6716             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_886.12       737.65     2_623.77       0.6718             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_886.12     1_087.40     2_973.52       0.6718             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_886.12     3_627.91     5_514.03       0.6048             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_602.88     1_024.37     3_627.26       0.8350             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_602.88     1_287.38     3_890.26       0.8354             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_602.88     1_909.14     4_512.02       0.8354             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_602.88     6_343.78     8_946.66       0.8013             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_557.60       486.34     3_043.94       0.5298             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_557.60       549.10     3_106.70       0.5298             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_557.60       783.37     3_340.97       0.5298             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_557.60     2_602.88     5_160.47       0.4120             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_100.13       757.44     2_857.57       0.6736             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_100.13       856.20     2_956.33       0.6737             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_100.13     1_238.08     3_338.20       0.6738             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_100.13     4_167.08     6_267.20       0.6015             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_847.01     1_330.62     4_177.63       0.8371             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_847.01     1_499.82     4_346.83       0.8373             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_847.01     2_168.37     5_015.39       0.8373             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_847.01     7_236.18    10_083.19       0.8028             NaN         3.61
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
Exhaustive (query)                                        20.42     9_629.44     9_649.86       1.0000          1.0000        97.66
Exhaustive (self)                                         20.42    31_990.86    32_011.28       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_146.24       710.86     1_857.10       0.2164             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_146.24     2_210.47     3_356.71       0.1776             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_304.40     1_502.11     4_806.51       0.2862             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_304.40     4_984.61     8_289.01       0.2244             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_398.89     3_904.72     6_303.61       0.3812             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_398.89    13_159.96    15_558.85       0.3011             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_516.94       344.30     3_861.24       0.3740             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_516.94       563.34     4_080.28       0.3740             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_516.94       790.17     4_307.11       0.3740             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_516.94     2_618.22     6_135.15       0.2675             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_516.93       486.77     6_003.70       0.4850             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_516.93       797.51     6_314.44       0.4850             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_516.93     1_102.69     6_619.61       0.4850             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_516.93     3_695.15     9_212.07       0.3901             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_668.16       803.56     5_471.72       0.6268             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_668.16     1_288.16     5_956.32       0.6268             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_668.16     1_775.99     6_444.15       0.6268             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_668.16     5_921.21    10_589.36       0.5760             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_290.54       515.31     2_805.85       0.3699             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_290.54       654.46     2_945.00       0.3699             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_290.54       939.46     3_230.00       0.3699             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_290.54     3_121.00     5_411.54       0.2575             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_459.23       713.56     5_172.79       0.4847             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_459.23       899.68     5_358.91       0.4847             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_459.23     1_326.73     5_785.96       0.4847             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_459.23     4_366.90     8_826.13       0.3781             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_584.22     1_160.84     4_745.06       0.6276             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_584.22     1_442.26     5_026.49       0.6276             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_584.22     2_134.22     5_718.44       0.6276             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_584.22     7_051.78    10_636.00       0.5701             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_708.80       673.11     3_381.91       0.3692             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_708.80       757.18     3_465.98       0.3692             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_708.80     1_153.59     3_862.39       0.3692             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_708.80     3_582.28     6_291.08       0.2524             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_857.34       942.82     5_800.16       0.4838             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_857.34     1_057.40     5_914.73       0.4838             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_857.34     1_514.69     6_372.03       0.4838             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_857.34     5_096.73     9_954.07       0.3703             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_972.08     1_551.34     5_523.42       0.6309             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_972.08     1_725.34     5_697.42       0.6309             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_972.08     2_457.45     6_429.54       0.6309             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_972.08     8_189.85    12_161.93       0.5684             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.44    16_074.34    16_104.78       1.0000          1.0000       146.48
Exhaustive (self)                                         30.44    53_626.32    53_656.77       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_532.45       669.18     2_201.63       0.2117             NaN         1.51
Exhaustive-PQ-m16 (self)                               1_532.45     2_222.85     3_755.30       0.1771             NaN         1.51
Exhaustive-PQ-m32 (query)                              2_197.65     1_512.50     3_710.15       0.2752             NaN         2.28
Exhaustive-PQ-m32 (self)                               2_197.65     5_024.99     7_222.63       0.2200             NaN         2.28
Exhaustive-PQ-m64 (query)                              3_231.77     3_959.55     7_191.32       0.3604             NaN         3.80
Exhaustive-PQ-m64 (self)                               3_231.77    13_141.66    16_373.44       0.2900             NaN         3.80
Exhaustive-PQ-m128 (query)                             4_988.04     9_145.01    14_133.05       0.4602             NaN         6.86
Exhaustive-PQ-m128 (self)                              4_988.04    30_428.69    35_416.73       0.3920             NaN         6.86
IVF-PQ-nl158-m16-np7 (query)                           4_900.11       440.67     5_340.79       0.3598             NaN         1.98
IVF-PQ-nl158-m16-np12 (query)                          4_900.11       727.99     5_628.10       0.3598             NaN         1.98
IVF-PQ-nl158-m16-np17 (query)                          4_900.11     1_021.67     5_921.78       0.3598             NaN         1.98
IVF-PQ-nl158-m16 (self)                                4_900.11     3_379.44     8_279.56       0.2575             NaN         1.98
IVF-PQ-nl158-m32-np7 (query)                           5_659.94       593.93     6_253.86       0.4576             NaN         2.74
IVF-PQ-nl158-m32-np12 (query)                          5_659.94       965.80     6_625.74       0.4576             NaN         2.74
IVF-PQ-nl158-m32-np17 (query)                          5_659.94     1_346.88     7_006.81       0.4576             NaN         2.74
IVF-PQ-nl158-m32 (self)                                5_659.94     4_488.55    10_148.48       0.3634             NaN         2.74
IVF-PQ-nl158-m64-np7 (query)                           6_628.35       894.40     7_522.75       0.5673             NaN         4.27
IVF-PQ-nl158-m64-np12 (query)                          6_628.35     1_434.21     8_062.56       0.5673             NaN         4.27
IVF-PQ-nl158-m64-np17 (query)                          6_628.35     1_977.00     8_605.35       0.5673             NaN         4.27
IVF-PQ-nl158-m64 (self)                                6_628.35     6_542.74    13_171.10       0.5132             NaN         4.27
IVF-PQ-nl158-m128-np7 (query)                          8_418.98     1_866.66    10_285.64       0.7346             NaN         7.32
IVF-PQ-nl158-m128-np12 (query)                         8_418.98     2_937.13    11_356.12       0.7346             NaN         7.32
IVF-PQ-nl158-m128-np17 (query)                         8_418.98     4_065.70    12_484.68       0.7346             NaN         7.32
IVF-PQ-nl158-m128 (self)                               8_418.98    13_511.55    21_930.53       0.7167             NaN         7.32
IVF-PQ-nl223-m16-np11 (query)                          3_141.55       642.56     3_784.11       0.3533             NaN         2.17
IVF-PQ-nl223-m16-np14 (query)                          3_141.55       796.86     3_938.41       0.3533             NaN         2.17
IVF-PQ-nl223-m16-np21 (query)                          3_141.55     1_175.02     4_316.57       0.3533             NaN         2.17
IVF-PQ-nl223-m16 (self)                                3_141.55     3_880.36     7_021.91       0.2434             NaN         2.17
IVF-PQ-nl223-m32-np11 (query)                          3_838.30       849.71     4_688.01       0.4502             NaN         2.93
IVF-PQ-nl223-m32-np14 (query)                          3_838.30     1_056.15     4_894.45       0.4502             NaN         2.93
IVF-PQ-nl223-m32-np21 (query)                          3_838.30     1_564.11     5_402.40       0.4502             NaN         2.93
IVF-PQ-nl223-m32 (self)                                3_838.30     5_173.65     9_011.95       0.3382             NaN         2.93
IVF-PQ-nl223-m64-np11 (query)                          4_886.01     1_292.73     6_178.74       0.5669             NaN         4.46
IVF-PQ-nl223-m64-np14 (query)                          4_886.01     1_603.66     6_489.68       0.5669             NaN         4.46
IVF-PQ-nl223-m64-np21 (query)                          4_886.01     2_343.36     7_229.38       0.5669             NaN         4.46
IVF-PQ-nl223-m64 (self)                                4_886.01     7_813.66    12_699.68       0.4938             NaN         4.46
IVF-PQ-nl223-m128-np11 (query)                         6_782.22     2_751.63     9_533.85       0.7377             NaN         7.51
IVF-PQ-nl223-m128-np14 (query)                         6_782.22     3_393.66    10_175.87       0.7377             NaN         7.51
IVF-PQ-nl223-m128-np21 (query)                         6_782.22     4_990.42    11_772.63       0.7377             NaN         7.51
IVF-PQ-nl223-m128 (self)                               6_782.22    16_595.60    23_377.82       0.7121             NaN         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_664.55       864.01     4_528.56       0.3509             NaN         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_664.55       966.67     4_631.21       0.3509             NaN         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_664.55     1_383.76     5_048.31       0.3509             NaN         2.44
IVF-PQ-nl316-m16 (self)                                3_664.55     4_616.10     8_280.65       0.2365             NaN         2.44
IVF-PQ-nl316-m32-np15 (query)                          4_408.34     1_138.90     5_547.24       0.4442             NaN         3.21
IVF-PQ-nl316-m32-np17 (query)                          4_408.34     1_283.78     5_692.12       0.4442             NaN         3.21
IVF-PQ-nl316-m32-np25 (query)                          4_408.34     1_828.74     6_237.09       0.4442             NaN         3.21
IVF-PQ-nl316-m32 (self)                                4_408.34     6_106.07    10_514.41       0.3192             NaN         3.21
IVF-PQ-nl316-m64-np15 (query)                          5_462.73     1_712.60     7_175.33       0.5655             NaN         4.73
IVF-PQ-nl316-m64-np17 (query)                          5_462.73     1_925.03     7_387.76       0.5655             NaN         4.73
IVF-PQ-nl316-m64-np25 (query)                          5_462.73     2_743.20     8_205.93       0.5655             NaN         4.73
IVF-PQ-nl316-m64 (self)                                5_462.73     9_154.98    14_617.71       0.4802             NaN         4.73
IVF-PQ-nl316-m128-np15 (query)                         7_220.08     3_602.24    10_822.32       0.7393             NaN         7.78
IVF-PQ-nl316-m128-np17 (query)                         7_220.08     4_093.48    11_313.56       0.7393             NaN         7.78
IVF-PQ-nl316-m128-np25 (query)                         7_220.08     5_792.81    13_012.89       0.7393             NaN         7.78
IVF-PQ-nl316-m128 (self)                               7_220.08    20_935.43    28_155.51       0.7111             NaN         7.78
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Cell embeddings

Synthetic data that resembles the embeddings generated by single cell models
such as GeneFormer, scGPT, etc.

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.71     4_115.90     4_125.61       1.0000          1.0000        48.83
Exhaustive (self)                                          9.71    13_755.00    13_764.71       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_638.89       649.73     2_288.62       0.7118             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_638.89     2_157.30     3_796.20       0.6210             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_196.28     1_484.12     2_680.40       0.7717             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_196.28     4_942.86     6_139.14       0.6993             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_921.62     3_888.33     5_809.96       0.8251             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_921.62    13_031.59    14_953.21       0.7675             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_813.39       253.87     3_067.26       0.8272             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_813.39       418.62     3_232.01       0.8277             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_813.39       587.39     3_400.79       0.8277             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_813.39     1_954.02     4_767.41       0.7669             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_384.66       422.58     2_807.24       0.8746             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_384.66       706.46     3_091.12       0.8752             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_384.66       992.42     3_377.08       0.8752             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_384.66     3_315.68     5_700.34       0.8288             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_116.88       784.92     3_901.80       0.9048             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_116.88     1_325.55     4_442.44       0.9056             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_116.88     1_865.31     4_982.19       0.9056             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_116.88     6_302.33     9_419.21       0.8704             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          2_183.07       380.08     2_563.15       0.8424             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          2_183.07       479.69     2_662.77       0.8424             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          2_183.07       709.62     2_892.69       0.8424             NaN         1.23
IVF-PQ-nl223-m16 (self)                                2_183.07     2_400.08     4_583.15       0.7839             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_730.94       601.58     2_332.53       0.8833             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_730.94       767.40     2_498.34       0.8834             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_730.94     1_138.56     2_869.50       0.8834             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_730.94     3_843.44     5_574.38       0.8398             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_518.14     1_077.27     3_595.41       0.9099             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_518.14     1_375.98     3_894.12       0.9101             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_518.14     2_056.24     4_574.38       0.9101             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_518.14     6_826.45     9_344.59       0.8765             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          2_394.06       501.90     2_895.96       0.8500             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          2_394.06       556.77     2_950.83       0.8500             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          2_394.06       813.89     3_207.95       0.8500             NaN         1.32
IVF-PQ-nl316-m16 (self)                                2_394.06     2_687.54     5_081.61       0.7913             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_961.31       775.90     2_737.21       0.8870             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_961.31       878.20     2_839.51       0.8871             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_961.31     1_272.24     3_233.55       0.8871             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_961.31     4_247.19     6_208.50       0.8430             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_673.44     1_374.52     4_047.96       0.9128             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_673.44     1_553.90     4_227.34       0.9128             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_673.44     2_269.41     4_942.85       0.9129             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_673.44     7_596.21    10_269.65       0.8801             NaN         3.61
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
Exhaustive (query)                                        21.05     9_616.13     9_637.18       1.0000          1.0000        97.66
Exhaustive (self)                                         21.05    32_195.78    32_216.83       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_150.11       664.25     1_814.36       0.6791             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_150.11     2_214.70     3_364.81       0.5853             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_294.92     1_499.35     4_794.27       0.7374             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_294.92     4_997.58     8_292.50       0.6552             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_398.79     3_939.11     6_337.89       0.7805             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_398.79    13_076.97    15_475.76       0.7136             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_927.39       361.53     4_288.92       0.8455             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_927.39       601.47     4_528.86       0.8458             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_927.39       843.82     4_771.21       0.8458             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_927.39     2_814.83     6_742.22       0.7844             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_847.24       522.79     6_370.03       0.8726             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_847.24       884.46     6_731.71       0.8731             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_847.24     1_232.49     7_079.73       0.8731             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_847.24     4_157.14    10_004.38       0.8208             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_019.19       884.42     5_903.61       0.8936             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_019.19     1_510.40     6_529.58       0.8941             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_019.19     2_099.10     7_118.29       0.8941             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_019.19     6_926.38    11_945.56       0.8494             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_193.53       518.96     2_712.49       0.8549             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_193.53       658.19     2_851.72       0.8550             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_193.53       970.69     3_164.22       0.8550             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_193.53     3_219.82     5_413.35       0.7968             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          4_197.36       735.53     4_932.89       0.8808             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          4_197.36       928.62     5_125.99       0.8809             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          4_197.36     1_375.86     5_573.22       0.8809             NaN         2.46
IVF-PQ-nl223-m32 (self)                                4_197.36     4_644.66     8_842.02       0.8303             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_270.93     1_227.86     4_498.78       0.8994             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_270.93     1_550.96     4_821.89       0.8994             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_270.93     2_324.90     5_595.82       0.8995             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_270.93     7_651.29    10_922.22       0.8567             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_274.65       688.63     2_963.28       0.8705             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_274.65       764.67     3_039.32       0.8705             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_274.65     1_116.70     3_391.35       0.8705             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_274.65     3_701.05     5_975.69       0.8156             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          4_436.85       960.57     5_397.42       0.8917             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          4_436.85     1_078.34     5_515.19       0.8917             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          4_436.85     1_561.43     5_998.28       0.8917             NaN         2.65
IVF-PQ-nl316-m32 (self)                                4_436.85     5_216.52     9_653.37       0.8455             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_568.40     1_591.54     5_159.94       0.9064             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_568.40     1_790.03     5_358.44       0.9064             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_568.40     2_630.24     6_198.64       0.9064             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_568.40     8_691.54    12_259.95       0.8665             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.16    15_601.88    15_632.05       1.0000          1.0000       146.48
Exhaustive (self)                                         30.16    52_306.41    52_336.57       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_472.72       669.83     2_142.55       0.6502             NaN         1.51
Exhaustive-PQ-m16 (self)                               1_472.72     2_223.49     3_696.20       0.5522             NaN         1.51
Exhaustive-PQ-m32 (query)                              2_126.75     1_524.82     3_651.57       0.7657             NaN         2.28
Exhaustive-PQ-m32 (self)                               2_126.75     5_121.38     7_248.13       0.6925             NaN         2.28
Exhaustive-PQ-m64 (query)                              3_193.66     3_923.20     7_116.86       0.8202             NaN         3.80
Exhaustive-PQ-m64 (self)                               3_193.66    13_139.26    16_332.92       0.7633             NaN         3.80
Exhaustive-PQ-m128 (query)                             4_993.96     9_123.27    14_117.23       0.8668             NaN         6.86
Exhaustive-PQ-m128 (self)                              4_993.96    30_434.24    35_428.20       0.8261             NaN         6.86
IVF-PQ-nl158-m16-np7 (query)                           5_418.02       458.50     5_876.52       0.8522             NaN         1.98
IVF-PQ-nl158-m16-np12 (query)                          5_418.02       757.95     6_175.97       0.8524             NaN         1.98
IVF-PQ-nl158-m16-np17 (query)                          5_418.02     1_124.68     6_542.70       0.8524             NaN         1.98
IVF-PQ-nl158-m16 (self)                                5_418.02     3_546.83     8_964.85       0.7910             NaN         1.98
IVF-PQ-nl158-m32-np7 (query)                           6_028.73       631.40     6_660.13       0.9000             NaN         2.74
IVF-PQ-nl158-m32-np12 (query)                          6_028.73     1_050.49     7_079.22       0.9001             NaN         2.74
IVF-PQ-nl158-m32-np17 (query)                          6_028.73     1_460.29     7_489.02       0.9001             NaN         2.74
IVF-PQ-nl158-m32 (self)                                6_028.73     4_885.40    10_914.13       0.8546             NaN         2.74
IVF-PQ-nl158-m64-np7 (query)                           6_987.40       975.78     7_963.17       0.9202             NaN         4.27
IVF-PQ-nl158-m64-np12 (query)                          6_987.40     1_649.48     8_636.87       0.9204             NaN         4.27
IVF-PQ-nl158-m64-np17 (query)                          6_987.40     2_279.59     9_266.99       0.9204             NaN         4.27
IVF-PQ-nl158-m64 (self)                                6_987.40     8_165.83    15_153.22       0.8831             NaN         4.27
IVF-PQ-nl158-m128-np7 (query)                         10_045.41     2_087.52    12_132.92       0.9393             NaN         7.32
IVF-PQ-nl158-m128-np12 (query)                        10_045.41     3_509.57    13_554.98       0.9395             NaN         7.32
IVF-PQ-nl158-m128-np17 (query)                        10_045.41     4_836.85    14_882.26       0.9395             NaN         7.32
IVF-PQ-nl158-m128 (self)                              10_045.41    16_137.03    26_182.43       0.9071             NaN         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_657.07       652.75     3_309.82       0.8626             NaN         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_657.07       818.14     3_475.21       0.8626             NaN         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_657.07     1_200.40     3_857.47       0.8626             NaN         2.17
IVF-PQ-nl223-m16 (self)                                2_657.07     4_291.67     6_948.74       0.8060             NaN         2.17
IVF-PQ-nl223-m32-np11 (query)                          3_337.21       874.80     4_212.01       0.9086             NaN         2.93
IVF-PQ-nl223-m32-np14 (query)                          3_337.21     1_100.80     4_438.01       0.9087             NaN         2.93
IVF-PQ-nl223-m32-np21 (query)                          3_337.21     1_635.98     4_973.19       0.9087             NaN         2.93
IVF-PQ-nl223-m32 (self)                                3_337.21     5_432.40     8_769.61       0.8680             NaN         2.93
IVF-PQ-nl223-m64-np11 (query)                          4_388.78     1_360.83     5_749.61       0.9273             NaN         4.46
IVF-PQ-nl223-m64-np14 (query)                          4_388.78     1_718.39     6_107.17       0.9274             NaN         4.46
IVF-PQ-nl223-m64-np21 (query)                          4_388.78     2_532.70     6_921.48       0.9274             NaN         4.46
IVF-PQ-nl223-m64 (self)                                4_388.78     8_441.62    12_830.40       0.8926             NaN         4.46
IVF-PQ-nl223-m128-np11 (query)                         6_304.11     2_883.20     9_187.31       0.9439             NaN         7.51
IVF-PQ-nl223-m128-np14 (query)                         6_304.11     3_635.68     9_939.79       0.9439             NaN         7.51
IVF-PQ-nl223-m128-np21 (query)                         6_304.11     5_410.41    11_714.52       0.9439             NaN         7.51
IVF-PQ-nl223-m128 (self)                               6_304.11    18_069.72    24_373.83       0.9132             NaN         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_046.21       873.22     3_919.43       0.8688             NaN         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_046.21       983.95     4_030.15       0.8688             NaN         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_046.21     1_410.45     4_456.66       0.8688             NaN         2.44
IVF-PQ-nl316-m16 (self)                                3_046.21     4_707.09     7_753.30       0.8141             NaN         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_713.61     1_162.25     4_875.85       0.9129             NaN         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_713.61     1_309.01     5_022.61       0.9129             NaN         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_713.61     1_901.50     5_615.10       0.9129             NaN         3.21
IVF-PQ-nl316-m32 (self)                                3_713.61     6_300.13    10_013.74       0.8726             NaN         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_755.70     1_794.75     6_550.45       0.9302             NaN         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_755.70     2_008.75     6_764.46       0.9302             NaN         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_755.70     2_886.13     7_641.83       0.9302             NaN         4.73
IVF-PQ-nl316-m64 (self)                                4_755.70     9_612.80    14_368.50       0.8965             NaN         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_578.88     3_732.32    10_311.21       0.9458             NaN         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_578.88     4_214.48    10_793.36       0.9458             NaN         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_578.88     6_138.77    12_717.65       0.9458             NaN         7.78
IVF-PQ-nl316-m128 (self)                               6_578.88    20_476.03    27_054.91       0.9164             NaN         7.78
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

As for PQ, let's start with correlated data.

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.82     4_096.96     4_106.78       1.0000          1.0000        48.83
Exhaustive (self)                                          9.82    13_909.66    13_919.48       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             8_497.78       654.08     9_151.86       0.1934             NaN         1.26
Exhaustive-OPQ-m16 (self)                              8_497.78     2_503.21    11_000.99       0.1801             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_245.40     1_496.91     7_742.31       0.2388             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_245.40     5_278.69    11_524.09       0.1982             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_691.98     3_906.40    13_598.38       0.3150             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_691.98    13_389.59    23_081.57       0.2571             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          9_698.12       249.46     9_947.58       0.3023             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         9_698.12       418.09    10_116.21       0.3023             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         9_698.12       583.29    10_281.41       0.3023             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               9_698.12     2_434.92    12_133.04       0.2155             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_844.17       413.40     8_257.57       0.4320             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_844.17       700.97     8_545.15       0.4321             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_844.17       995.71     8_839.88       0.4321             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_844.17     3_639.28    11_483.45       0.3438             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_424.22       756.92    12_181.14       0.6704             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_424.22     1_318.86    12_743.08       0.6709             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_424.22     2_200.48    13_624.70       0.6709             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_424.22     6_845.00    18_269.22       0.6112             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                        10_855.32       395.08    11_250.40       0.3078             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                        10_855.32       497.59    11_352.91       0.3078             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                        10_855.32       741.88    11_597.20       0.3078             NaN         1.48
IVF-OPQ-nl223-m16 (self)                              10_855.32     2_785.83    13_641.15       0.2192             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_429.14       656.53     8_085.67       0.4361             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_429.14       798.80     8_227.94       0.4361             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_429.14     1_180.18     8_609.32       0.4361             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_429.14     4_315.35    11_744.50       0.3461             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        11_236.31     1_105.41    12_341.71       0.6765             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        11_236.31     1_410.28    12_646.58       0.6765             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        11_236.31     2_141.28    13_377.59       0.6765             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              11_236.31     7_420.44    18_656.75       0.6165             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                        10_105.05       517.64    10_622.70       0.3115             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                        10_105.05       580.21    10_685.26       0.3115             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                        10_105.05       848.27    10_953.32       0.3115             NaN         1.57
IVF-OPQ-nl316-m16 (self)                              10_105.05     3_184.12    13_289.18       0.2213             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_690.85       814.77     8_505.63       0.4405             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_690.85       919.43     8_610.28       0.4406             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_690.85     1_381.60     9_072.45       0.4406             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_690.85     4_888.56    12_579.42       0.3493             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_799.20     1_438.82    13_238.01       0.6803             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_799.20     1_639.02    13_438.22       0.6803             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_799.20     2_422.98    14_222.17       0.6803             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_799.20     8_487.44    20_286.63       0.6197             NaN         3.86
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
Exhaustive (query)                                        20.68    10_000.23    10_020.91       1.0000          1.0000        97.66
Exhaustive (self)                                         20.68    33_729.87    33_750.55       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_518.67       728.92     8_247.60       0.1536             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_518.67     3_768.25    11_286.92       0.1519             NaN         2.26
Exhaustive-OPQ-m32 (query)                            18_805.86     1_565.20    20_371.06       0.1737             NaN         3.03
Exhaustive-OPQ-m32 (self)                             18_805.86     6_598.48    25_404.34       0.1581             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_643.20     4_096.86    17_740.06       0.2173             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_643.20    14_944.83    28_588.03       0.1750             NaN         4.55
Exhaustive-OPQ-m128 (query)                           20_427.13     9_438.06    29_865.19       0.2888             NaN         7.61
Exhaustive-OPQ-m128 (self)                            20_427.13    32_893.01    53_320.14       0.2357             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_497.05       355.23    10_852.28       0.2139             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_497.05       609.56    11_106.61       0.2139             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_497.05       853.46    11_350.51       0.2139             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_497.05     4_290.66    14_787.71       0.1540             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         21_716.02       515.26    22_231.28       0.2762             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        21_716.02       895.67    22_611.68       0.2762             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        21_716.02     1_216.89    22_932.90       0.2762             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              21_716.02     5_548.29    27_264.31       0.1915             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_741.62       866.50    17_608.12       0.4052             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_741.62     1_467.69    18_209.31       0.4053             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_741.62     2_071.89    18_813.50       0.4053             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_741.62     8_323.86    25_065.48       0.3197             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        23_207.56     1_551.43    24_758.99       0.6564             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       23_207.56     2_622.35    25_829.92       0.6567             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       23_207.56     3_695.73    26_903.29       0.6567             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             23_207.56    13_856.80    37_064.36       0.5968             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         9_368.24       614.34     9_982.58       0.2171             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_368.24       785.04    10_153.28       0.2171             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_368.24     1_082.45    10_450.69       0.2171             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_368.24     4_863.69    14_231.93       0.1564             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        21_189.04       744.37    21_933.41       0.2808             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        21_189.04       943.63    22_132.67       0.2808             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        21_189.04     1_402.57    22_591.61       0.2808             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              21_189.04     6_156.62    27_345.66       0.1940             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_762.58     1_258.50    16_021.09       0.4113             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_762.58     1_611.63    16_374.21       0.4113             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_762.58     2_463.64    17_226.23       0.4113             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_762.58     9_488.80    24_251.38       0.3234             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       21_619.38     2_246.11    23_865.48       0.6608             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       21_619.38     2_880.07    24_499.45       0.6609             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       21_619.38     4_330.16    25_949.53       0.6609             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             21_619.38    15_922.10    37_541.48       0.6004             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_998.22       711.16     9_709.38       0.2190             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_998.22       822.09     9_820.31       0.2190             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_998.22     1_179.82    10_178.04       0.2190             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_998.22     5_341.28    14_339.50       0.1576             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        20_531.00       989.55    21_520.55       0.2838             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        20_531.00     1_125.89    21_656.89       0.2838             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        20_531.00     1_621.39    22_152.39       0.2838             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              20_531.00     6_923.63    27_454.62       0.1957             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        15_247.55     1_628.39    16_875.94       0.4141             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        15_247.55     1_842.53    17_090.08       0.4141             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        15_247.55     2_702.62    17_950.17       0.4141             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              15_247.55    10_540.90    25_788.45       0.3252             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       22_219.32     2_873.91    25_093.23       0.6599             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       22_219.32     3_288.96    25_508.28       0.6600             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       22_219.32     4_863.61    27_082.93       0.6600             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             22_219.32    17_706.96    39_926.27       0.5998             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.86    16_424.69    16_457.54       1.0000          1.0000       146.48
Exhaustive (self)                                         32.86    55_283.86    55_316.71       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                            11_841.46       763.63    12_605.09       0.1406             NaN         3.76
Exhaustive-OPQ-m16 (self)                             11_841.46     5_706.13    17_547.58       0.1427             NaN         3.76
Exhaustive-OPQ-m32 (query)                            16_250.46     1_586.60    17_837.06       0.1559             NaN         4.53
Exhaustive-OPQ-m32 (self)                             16_250.46     8_532.53    24_782.99       0.1483             NaN         4.53
Exhaustive-OPQ-m64 (query)                            21_804.34     4_120.47    25_924.81       0.1830             NaN         6.05
Exhaustive-OPQ-m64 (self)                             21_804.34    16_890.10    38_694.44       0.1558             NaN         6.05
Exhaustive-OPQ-m128 (query)                           27_615.73     9_494.59    37_110.32       0.2282             NaN         9.11
Exhaustive-OPQ-m128 (self)                            27_615.73    35_477.69    63_093.42       0.1810             NaN         9.11
IVF-OPQ-nl158-m16-np7 (query)                         16_022.35       439.95    16_462.29       0.1816             NaN         4.23
IVF-OPQ-nl158-m16-np12 (query)                        16_022.35       750.42    16_772.76       0.1816             NaN         4.23
IVF-OPQ-nl158-m16-np17 (query)                        16_022.35     1_064.23    17_086.58       0.1816             NaN         4.23
IVF-OPQ-nl158-m16 (self)                              16_022.35     6_735.91    22_758.26       0.1375             NaN         4.23
IVF-OPQ-nl158-m32-np7 (query)                         19_180.64       611.14    19_791.79       0.2250             NaN         4.99
IVF-OPQ-nl158-m32-np12 (query)                        19_180.64     1_045.64    20_226.29       0.2250             NaN         4.99
IVF-OPQ-nl158-m32-np17 (query)                        19_180.64     1_452.70    20_633.34       0.2250             NaN         4.99
IVF-OPQ-nl158-m32 (self)                              19_180.64     8_182.91    27_363.55       0.1561             NaN         4.99
IVF-OPQ-nl158-m64-np7 (query)                         24_624.65       979.19    25_603.84       0.3047             NaN         6.52
IVF-OPQ-nl158-m64-np12 (query)                        24_624.65     1_678.76    26_303.41       0.3047             NaN         6.52
IVF-OPQ-nl158-m64-np17 (query)                        24_624.65     2_263.82    26_888.48       0.3047             NaN         6.52
IVF-OPQ-nl158-m64 (self)                              24_624.65    10_889.24    35_513.89       0.2169             NaN         6.52
IVF-OPQ-nl158-m128-np7 (query)                        32_784.10     1_988.38    34_772.48       0.4906             NaN         9.57
IVF-OPQ-nl158-m128-np12 (query)                       32_784.10     3_364.06    36_148.16       0.4907             NaN         9.57
IVF-OPQ-nl158-m128-np17 (query)                       32_784.10     4_784.55    37_568.65       0.4907             NaN         9.57
IVF-OPQ-nl158-m128 (self)                             32_784.10    19_389.98    52_174.08       0.4145             NaN         9.57
IVF-OPQ-nl223-m16-np11 (query)                        13_150.98       702.04    13_853.01       0.1835             NaN         4.42
IVF-OPQ-nl223-m16-np14 (query)                        13_150.98       849.25    14_000.23       0.1835             NaN         4.42
IVF-OPQ-nl223-m16-np21 (query)                        13_150.98     1_261.31    14_412.28       0.1835             NaN         4.42
IVF-OPQ-nl223-m16 (self)                              13_150.98     7_388.87    20_539.85       0.1401             NaN         4.42
IVF-OPQ-nl223-m32-np11 (query)                        16_467.07       913.97    17_381.04       0.2288             NaN         5.18
IVF-OPQ-nl223-m32-np14 (query)                        16_467.07     1_141.38    17_608.45       0.2288             NaN         5.18
IVF-OPQ-nl223-m32-np21 (query)                        16_467.07     1_675.03    18_142.10       0.2288             NaN         5.18
IVF-OPQ-nl223-m32 (self)                              16_467.07     8_918.54    25_385.62       0.1579             NaN         5.18
IVF-OPQ-nl223-m64-np11 (query)                        23_226.01     1_451.75    24_677.76       0.3091             NaN         6.71
IVF-OPQ-nl223-m64-np14 (query)                        23_226.01     1_796.65    25_022.66       0.3091             NaN         6.71
IVF-OPQ-nl223-m64-np21 (query)                        23_226.01     2_665.86    25_891.87       0.3091             NaN         6.71
IVF-OPQ-nl223-m64 (self)                              23_226.01    12_268.24    35_494.25       0.2202             NaN         6.71
IVF-OPQ-nl223-m128-np11 (query)                       30_652.18     3_127.05    33_779.23       0.4962             NaN         9.76
IVF-OPQ-nl223-m128-np14 (query)                       30_652.18     4_005.64    34_657.82       0.4962             NaN         9.76
IVF-OPQ-nl223-m128-np21 (query)                       30_652.18     5_991.01    36_643.19       0.4962             NaN         9.76
IVF-OPQ-nl223-m128 (self)                             30_652.18    22_468.88    53_121.06       0.4173             NaN         9.76
IVF-OPQ-nl316-m16-np15 (query)                        13_847.49       966.52    14_814.01       0.1864             NaN         4.69
IVF-OPQ-nl316-m16-np17 (query)                        13_847.49     1_076.43    14_923.92       0.1864             NaN         4.69
IVF-OPQ-nl316-m16-np25 (query)                        13_847.49     1_512.06    15_359.55       0.1864             NaN         4.69
IVF-OPQ-nl316-m16 (self)                              13_847.49     8_286.66    22_134.15       0.1409             NaN         4.69
IVF-OPQ-nl316-m32-np15 (query)                        17_276.07     1_182.18    18_458.25       0.2315             NaN         5.46
IVF-OPQ-nl316-m32-np17 (query)                        17_276.07     1_337.08    18_613.15       0.2315             NaN         5.46
IVF-OPQ-nl316-m32-np25 (query)                        17_276.07     1_994.56    19_270.63       0.2315             NaN         5.46
IVF-OPQ-nl316-m32 (self)                              17_276.07    10_138.16    27_414.23       0.1581             NaN         5.46
IVF-OPQ-nl316-m64-np15 (query)                        22_165.33     1_785.81    23_951.14       0.3129             NaN         6.98
IVF-OPQ-nl316-m64-np17 (query)                        22_165.33     2_041.64    24_206.97       0.3129             NaN         6.98
IVF-OPQ-nl316-m64-np25 (query)                        22_165.33     2_985.95    25_151.28       0.3129             NaN         6.98
IVF-OPQ-nl316-m64 (self)                              22_165.33    13_478.53    35_643.86       0.2220             NaN         6.98
IVF-OPQ-nl316-m128-np15 (query)                       28_246.18     3_714.41    31_960.59       0.4985             NaN        10.04
IVF-OPQ-nl316-m128-np17 (query)                       28_246.18     4_254.72    32_500.90       0.4985             NaN        10.04
IVF-OPQ-nl316-m128-np25 (query)                       28_246.18     6_240.15    34_486.33       0.4985             NaN        10.04
IVF-OPQ-nl316-m128 (self)                             28_246.18    23_781.84    52_028.01       0.4190             NaN        10.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.28     4_391.74     4_402.02       1.0000          1.0000        48.83
Exhaustive (self)                                         10.28    14_787.60    14_797.88       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                            11_458.19       676.10    12_134.28       0.2878             NaN         1.26
Exhaustive-OPQ-m16 (self)                             11_458.19     2_574.94    14_033.13       0.2016             NaN         1.26
Exhaustive-OPQ-m32 (query)                             7_772.34     1_556.59     9_328.92       0.4077             NaN         2.03
Exhaustive-OPQ-m32 (self)                              7_772.34     5_496.31    13_268.65       0.2913             NaN         2.03
Exhaustive-OPQ-m64 (query)                            11_957.23     4_094.97    16_052.21       0.5546             NaN         3.55
Exhaustive-OPQ-m64 (self)                             11_957.23    14_131.76    26_089.00       0.4616             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                         10_355.35       262.04    10_617.39       0.6092             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                        10_355.35       420.04    10_775.39       0.6092             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                        10_355.35       585.88    10_941.24       0.6092             NaN         1.42
IVF-OPQ-nl158-m16 (self)                              10_355.35     2_322.46    12_677.81       0.5571             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_988.76       412.50     8_401.26       0.7378             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_988.76       683.05     8_671.81       0.7379             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_988.76       960.38     8_949.14       0.7379             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_988.76     3_526.31    11_515.07       0.7040             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_835.63       743.13    12_578.76       0.8487             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_835.63     1_280.90    13_116.53       0.8488             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_835.63     1_678.67    13_514.30       0.8488             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_835.63     6_019.39    17_855.02       0.8226             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                        10_132.84       390.47    10_523.31       0.6133             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                        10_132.84       486.10    10_618.94       0.6134             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                        10_132.84       723.82    10_856.66       0.6134             NaN         1.48
IVF-OPQ-nl223-m16 (self)                              10_132.84     2_788.47    12_921.31       0.5650             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_605.11       616.60     8_221.71       0.7368             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_605.11       800.86     8_405.96       0.7370             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_605.11     1_151.70     8_756.81       0.7370             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_605.11     4_193.51    11_798.61       0.7065             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        11_522.70     1_084.98    12_607.68       0.8500             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        11_522.70     1_374.07    12_896.78       0.8503             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        11_522.70     2_046.61    13_569.31       0.8503             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              11_522.70     7_101.95    18_624.65       0.8258             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                        10_125.83       507.61    10_633.44       0.6163             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                        10_125.83       589.99    10_715.82       0.6163             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                        10_125.83       830.20    10_956.03       0.6163             NaN         1.57
IVF-OPQ-nl316-m16 (self)                              10_125.83     3_134.80    13_260.63       0.5677             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_804.64       808.56     8_613.20       0.7389             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_804.64       918.70     8_723.34       0.7390             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_804.64     1_344.25     9_148.89       0.7390             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_804.64     4_813.53    12_618.17       0.7090             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_529.15     1_470.16    12_999.30       0.8511             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_529.15     1_591.02    13_120.17       0.8513             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_529.15     2_326.34    13_855.49       0.8513             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_529.15     8_135.27    19_664.41       0.8270             NaN         3.86
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
Exhaustive (query)                                        19.93    10_044.58    10_064.51       1.0000          1.0000        97.66
Exhaustive (self)                                         19.93    33_935.83    33_955.76       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_958.81       690.04     8_648.85       0.1854             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_958.81     3_776.08    11_734.89       0.1630             NaN         2.26
Exhaustive-OPQ-m32 (query)                            19_233.14     1_577.79    20_810.93       0.2715             NaN         3.03
Exhaustive-OPQ-m32 (self)                             19_233.14     6_648.81    25_881.96       0.2139             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_625.22     4_095.03    17_720.26       0.3928             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_625.22    15_022.09    28_647.31       0.3001             NaN         4.55
Exhaustive-OPQ-m128 (query)                           20_419.80     9_415.08    29_834.88       0.5561             NaN         7.61
Exhaustive-OPQ-m128 (self)                            20_419.80    32_899.85    53_319.64       0.4604             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_279.86       367.35    10_647.21       0.4397             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_279.86       587.78    10_867.64       0.4397             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_279.86       800.25    11_080.11       0.4397             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_279.86     4_156.08    14_435.94       0.3671             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         21_403.19       502.62    21_905.81       0.5879             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        21_403.19       825.14    22_228.33       0.5879             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        21_403.19     1_157.43    22_560.62       0.5879             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              21_403.19     5_299.50    26_702.69       0.5307             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_081.31       839.41    16_920.72       0.7237             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_081.31     1_356.43    17_437.74       0.7237             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_081.31     1_887.66    17_968.96       0.7237             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_081.31     7_833.21    23_914.51       0.6826             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_972.43     1_486.07    24_458.50       0.8341             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_972.43     2_370.36    25_342.79       0.8341             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_972.43     3_272.81    26_245.24       0.8341             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_972.43    12_485.19    35_457.62       0.8069             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_709.99       532.03     9_242.02       0.4461             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_709.99       693.80     9_403.79       0.4461             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_709.99     1_024.33     9_734.32       0.4461             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_709.99     4_756.61    13_466.60       0.3745             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        20_291.14       740.53    21_031.67       0.5906             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        20_291.14       921.72    21_212.85       0.5906             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        20_291.14     1_452.76    21_743.90       0.5906             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              20_291.14     6_028.36    26_319.49       0.5338             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_957.53     1_224.20    16_181.73       0.7242             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_957.53     1_539.58    16_497.11       0.7242             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_957.53     2_249.24    17_206.77       0.7242             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_957.53     9_064.84    24_022.37       0.6856             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       21_978.68     2_178.90    24_157.58       0.8358             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       21_978.68     2_712.50    24_691.18       0.8358             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       21_978.68     3_977.73    25_956.41       0.8358             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             21_978.68    14_756.24    36_734.91       0.8104             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_142.98       708.44     9_851.43       0.4478             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_142.98       826.33     9_969.31       0.4478             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_142.98     1_172.87    10_315.85       0.4478             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_142.98     5_293.11    14_436.09       0.3767             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        20_542.22       985.14    21_527.35       0.5910             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        20_542.22     1_099.11    21_641.33       0.5910             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        20_542.22     1_601.50    22_143.72       0.5910             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              20_542.22     7_058.41    27_600.63       0.5372             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        17_405.90     1_891.57    19_297.47       0.7247             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        17_405.90     2_160.57    19_566.47       0.7247             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        17_405.90     2_850.99    20_256.88       0.7247             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              17_405.90    11_122.45    28_528.35       0.6865             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       27_202.03     2_926.05    30_128.07       0.8377             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       27_202.03     3_308.06    30_510.08       0.8377             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       27_202.03     4_838.10    32_040.13       0.8377             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             27_202.03    16_716.82    43_918.85       0.8114             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.28    15_854.48    15_886.76       1.0000          1.0000       146.48
Exhaustive (self)                                         32.28    56_046.30    56_078.57       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                            12_015.79       713.74    12_729.53       0.1745             NaN         3.76
Exhaustive-OPQ-m16 (self)                             12_015.79     5_777.23    17_793.02       0.1562             NaN         3.76
Exhaustive-OPQ-m32 (query)                            15_183.47     1_783.24    16_966.71       0.2416             NaN         4.53
Exhaustive-OPQ-m32 (self)                             15_183.47     9_068.88    24_252.35       0.1951             NaN         4.53
Exhaustive-OPQ-m64 (query)                            19_981.33     4_010.84    23_992.17       0.3557             NaN         6.05
Exhaustive-OPQ-m64 (self)                             19_981.33    16_387.19    36_368.52       0.2747             NaN         6.05
Exhaustive-OPQ-m128 (query)                           25_518.63     9_189.30    34_707.92       0.4983             NaN         9.11
Exhaustive-OPQ-m128 (self)                            25_518.63    33_924.58    59_443.21       0.4004             NaN         9.11
IVF-OPQ-nl158-m16-np7 (query)                         14_771.61       428.14    15_199.75       0.4184             NaN         4.23
IVF-OPQ-nl158-m16-np12 (query)                        14_771.61       708.84    15_480.44       0.4184             NaN         4.23
IVF-OPQ-nl158-m16-np17 (query)                        14_771.61       995.32    15_766.93       0.4184             NaN         4.23
IVF-OPQ-nl158-m16 (self)                              14_771.61     6_444.52    21_216.13       0.3538             NaN         4.23
IVF-OPQ-nl158-m32-np7 (query)                         17_583.22       573.15    18_156.37       0.5520             NaN         4.99
IVF-OPQ-nl158-m32-np12 (query)                        17_583.22       939.42    18_522.65       0.5520             NaN         4.99
IVF-OPQ-nl158-m32-np17 (query)                        17_583.22     1_308.50    18_891.72       0.5520             NaN         4.99
IVF-OPQ-nl158-m32 (self)                              17_583.22     7_514.99    25_098.21       0.4969             NaN         4.99
IVF-OPQ-nl158-m64-np7 (query)                         25_595.84       901.33    26_497.17       0.6789             NaN         6.52
IVF-OPQ-nl158-m64-np12 (query)                        25_595.84     1_506.32    27_102.16       0.6789             NaN         6.52
IVF-OPQ-nl158-m64-np17 (query)                        25_595.84     2_160.42    27_756.26       0.6789             NaN         6.52
IVF-OPQ-nl158-m64 (self)                              25_595.84    10_244.55    35_840.39       0.6401             NaN         6.52
IVF-OPQ-nl158-m128-np7 (query)                        35_396.42     1_992.32    37_388.74       0.7977             NaN         9.57
IVF-OPQ-nl158-m128-np12 (query)                       35_396.42     3_208.20    38_604.62       0.7977             NaN         9.57
IVF-OPQ-nl158-m128-np17 (query)                       35_396.42     4_509.43    39_905.85       0.7977             NaN         9.57
IVF-OPQ-nl158-m128 (self)                             35_396.42    18_479.92    53_876.34       0.7712             NaN         9.57
IVF-OPQ-nl223-m16-np11 (query)                        13_862.25       648.84    14_511.09       0.4240             NaN         4.42
IVF-OPQ-nl223-m16-np14 (query)                        13_862.25       823.23    14_685.48       0.4240             NaN         4.42
IVF-OPQ-nl223-m16-np21 (query)                        13_862.25     1_220.68    15_082.93       0.4240             NaN         4.42
IVF-OPQ-nl223-m16 (self)                              13_862.25     7_124.08    20_986.33       0.3616             NaN         4.42
IVF-OPQ-nl223-m32-np11 (query)                        15_514.24       852.32    16_366.56       0.5547             NaN         5.18
IVF-OPQ-nl223-m32-np14 (query)                        15_514.24     1_071.79    16_586.03       0.5547             NaN         5.18
IVF-OPQ-nl223-m32-np21 (query)                        15_514.24     1_581.27    17_095.51       0.5547             NaN         5.18
IVF-OPQ-nl223-m32 (self)                              15_514.24     8_375.49    23_889.73       0.5047             NaN         5.18
IVF-OPQ-nl223-m64-np11 (query)                        20_406.62     1_318.99    21_725.61       0.6781             NaN         6.71
IVF-OPQ-nl223-m64-np14 (query)                        20_406.62     1_654.40    22_061.02       0.6781             NaN         6.71
IVF-OPQ-nl223-m64-np21 (query)                        20_406.62     2_424.72    22_831.34       0.6781             NaN         6.71
IVF-OPQ-nl223-m64 (self)                              20_406.62    11_302.62    31_709.23       0.6425             NaN         6.71
IVF-OPQ-nl223-m128-np11 (query)                       27_272.36     2_755.60    30_027.97       0.7971             NaN         9.76
IVF-OPQ-nl223-m128-np14 (query)                       27_272.36     3_457.89    30_730.26       0.7971             NaN         9.76
IVF-OPQ-nl223-m128-np21 (query)                       27_272.36     5_130.23    32_402.60       0.7971             NaN         9.76
IVF-OPQ-nl223-m128 (self)                             27_272.36    20_238.28    47_510.64       0.7736             NaN         9.76
IVF-OPQ-nl316-m16-np15 (query)                        13_161.83       881.21    14_043.04       0.4281             NaN         4.69
IVF-OPQ-nl316-m16-np17 (query)                        13_161.83       995.40    14_157.23       0.4281             NaN         4.69
IVF-OPQ-nl316-m16-np25 (query)                        13_161.83     1_422.89    14_584.73       0.4281             NaN         4.69
IVF-OPQ-nl316-m16 (self)                              13_161.83     7_860.24    21_022.08       0.3647             NaN         4.69
IVF-OPQ-nl316-m32-np15 (query)                        16_003.48     1_151.28    17_154.76       0.5554             NaN         5.46
IVF-OPQ-nl316-m32-np17 (query)                        16_003.48     1_296.68    17_300.16       0.5554             NaN         5.46
IVF-OPQ-nl316-m32-np25 (query)                        16_003.48     1_856.91    17_860.39       0.5554             NaN         5.46
IVF-OPQ-nl316-m32 (self)                              16_003.48     9_451.48    25_454.97       0.5034             NaN         5.46
IVF-OPQ-nl316-m64-np15 (query)                        21_045.76     1_770.16    22_815.92       0.6787             NaN         6.98
IVF-OPQ-nl316-m64-np17 (query)                        21_045.76     2_004.66    23_050.42       0.6787             NaN         6.98
IVF-OPQ-nl316-m64-np25 (query)                        21_045.76     2_913.69    23_959.45       0.6787             NaN         6.98
IVF-OPQ-nl316-m64 (self)                              21_045.76    13_193.04    34_238.80       0.6437             NaN         6.98
IVF-OPQ-nl316-m128-np15 (query)                       28_464.12     3_742.81    32_206.93       0.7987             NaN        10.04
IVF-OPQ-nl316-m128-np17 (query)                       28_464.12     4_272.98    32_737.10       0.7987             NaN        10.04
IVF-OPQ-nl316-m128-np25 (query)                       28_464.12     6_181.27    34_645.40       0.7987             NaN        10.04
IVF-OPQ-nl316-m128 (self)                             28_464.12    23_658.68    52_122.81       0.7741             NaN        10.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Cell embeddings

Lastly, also here the synthetic data that resembles the embeddings generated by
single cell models.

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.42     4_353.82     4_364.24       1.0000          1.0000        48.83
Exhaustive (self)                                         10.42    14_937.59    14_948.01       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             9_298.94       674.25     9_973.19       0.7808             NaN         1.26
Exhaustive-OPQ-m16 (self)                              9_298.94     2_578.63    11_877.57       0.7140             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_831.09     1_551.41     8_382.50       0.8259             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_831.09     5_477.11    12_308.20       0.7725             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_880.22     4_097.28    14_977.50       0.8530             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_880.22    14_037.46    24_917.68       0.8066             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                         10_394.96       264.29    10_659.25       0.8764             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                        10_394.96       445.55    10_840.51       0.8770             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                        10_394.96       632.71    11_027.66       0.8770             NaN         1.42
IVF-OPQ-nl158-m16 (self)                              10_394.96     2_460.93    12_855.89       0.8370             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          8_082.07       451.71     8_533.78       0.9041             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         8_082.07       762.01     8_844.09       0.9049             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         8_082.07     1_071.67     9_153.74       0.9049             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               8_082.07     3_924.14    12_006.21       0.8721             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_797.08       832.19    12_629.27       0.9212             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_797.08     1_429.99    13_227.07       0.9220             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_797.08     2_024.41    13_821.49       0.9220             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_797.08     7_079.33    18_876.41       0.8939             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         9_831.87       387.21    10_219.08       0.8776             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         9_831.87       502.95    10_334.82       0.8777             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         9_831.87       735.90    10_567.77       0.8777             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               9_831.87     2_827.42    12_659.28       0.8421             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_472.79       644.57     8_117.35       0.9034             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_472.79       820.21     8_293.00       0.9035             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_472.79     1_221.17     8_693.96       0.9036             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_472.79     4_381.28    11_854.07       0.8741             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        11_399.16     1_144.73    12_543.89       0.9225             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        11_399.16     1_487.66    12_886.82       0.9227             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        11_399.16     2_205.15    13_604.31       0.9227             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              11_399.16     7_710.45    19_109.62       0.8958             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                        10_210.55       509.85    10_720.40       0.8759             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                        10_210.55       591.82    10_802.37       0.8759             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                        10_210.55       854.63    11_065.18       0.8760             NaN         1.57
IVF-OPQ-nl316-m16 (self)                              10_210.55     3_223.29    13_433.84       0.8435             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_721.25       840.72     8_561.96       0.9014             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_721.25       958.44     8_679.69       0.9015             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_721.25     1_381.24     9_102.49       0.9015             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_721.25     4_948.64    12_669.89       0.8725             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_511.65     1_468.16    12_979.81       0.9220             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_511.65     1_667.21    13_178.86       0.9220             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_511.65     2_443.39    13_955.04       0.9221             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_511.65     8_565.65    20_077.30       0.8966             NaN         3.86
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
Exhaustive (query)                                        20.74     9_794.08     9_814.82       1.0000          1.0000        97.66
Exhaustive (self)                                         20.74    32_719.86    32_740.60       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_277.12       663.96     7_941.08       0.7453             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_277.12     3_686.67    10_963.79       0.6701             NaN         2.26
Exhaustive-OPQ-m32 (query)                            17_852.17     1_505.91    19_358.08       0.7984             NaN         3.03
Exhaustive-OPQ-m32 (self)                             17_852.17     6_419.98    24_272.15       0.7383             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_644.75     4_222.71    16_867.46       0.8363             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_644.75    15_482.11    28_126.86       0.7874             NaN         4.55
Exhaustive-OPQ-m128 (query)                           18_741.48     9_170.43    27_911.91       0.9166             NaN         7.61
Exhaustive-OPQ-m128 (self)                            18_741.48    31_893.64    50_635.11       0.8913             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_422.53       359.63    10_782.16       0.8653             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_422.53       593.37    11_015.90       0.8656             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_422.53       837.93    11_260.45       0.8656             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_422.53     4_230.26    14_652.79       0.8283             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         20_666.58       516.82    21_183.40       0.8826             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        20_666.58       883.54    21_550.11       0.8831             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        20_666.58     1_233.31    21_899.88       0.8831             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              20_666.58     5_541.37    26_207.94       0.8488             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_322.31       890.14    16_212.45       0.8944             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_322.31     1_523.56    16_845.87       0.8948             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_322.31     2_145.10    17_467.41       0.8949             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_322.31     8_576.46    23_898.77       0.8640             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_557.09     1_700.69    24_257.78       0.9512             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_557.09     2_854.32    25_411.41       0.9518             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_557.09     3_986.76    26_543.85       0.9518             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_557.09    14_777.39    37_334.48       0.9376             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         9_249.96       539.03     9_789.00       0.8746             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_249.96       692.65     9_942.62       0.8746             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_249.96     1_009.07    10_259.03       0.8746             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_249.96     4_824.69    14_074.65       0.8387             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        20_286.46       753.26    21_039.72       0.8899             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        20_286.46       974.18    21_260.64       0.8899             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        20_286.46     1_425.74    21_712.20       0.8900             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              20_286.46     6_267.05    26_553.51       0.8570             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_645.97     1_310.38    15_956.35       0.9024             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_645.97     1_647.17    16_293.14       0.9024             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_645.97     2_444.80    17_090.77       0.9025             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_645.97     9_620.90    24_266.87       0.8725             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       21_480.15     2_295.50    23_775.65       0.9545             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       21_480.15     2_951.99    24_432.13       0.9546             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       21_480.15     4_389.60    25_869.74       0.9546             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             21_480.15    16_136.45    37_616.59       0.9406             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_859.04       735.02     9_594.06       0.8873             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_859.04       813.55     9_672.59       0.8874             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_859.04     1_202.65    10_061.69       0.8874             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_859.04     5_349.61    14_208.65       0.8516             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        20_200.29       999.74    21_200.03       0.9005             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        20_200.29     1_133.04    21_333.33       0.9005             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        20_200.29     1_640.75    21_841.04       0.9005             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              20_200.29     6_929.61    27_129.90       0.8675             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_892.79     1_651.05    16_543.84       0.9098             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_892.79     1_876.86    16_769.66       0.9098             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_892.79     2_740.12    17_632.92       0.9098             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_892.79    10_695.94    25_588.73       0.8800             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       21_977.25     2_948.30    24_925.55       0.9594             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       21_977.25     3_355.48    25_332.73       0.9594             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       21_977.25     4_953.65    26_930.89       0.9594             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             21_977.25    17_966.12    39_943.37       0.9451             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.30    16_317.90    16_348.20       1.0000          1.0000       146.48
Exhaustive (self)                                         30.30    54_306.77    54_337.07       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                            16_261.90       697.00    16_958.90       0.7280             NaN         3.76
Exhaustive-OPQ-m16 (self)                             16_261.90     5_593.89    21_855.79       0.6498             NaN         3.76
Exhaustive-OPQ-m32 (query)                            16_131.91     1_566.79    17_698.70       0.8434             NaN         4.53
Exhaustive-OPQ-m32 (self)                             16_131.91     8_472.82    24_604.72       0.7955             NaN         4.53
Exhaustive-OPQ-m64 (query)                            20_577.75     4_075.92    24_653.67       0.8759             NaN         6.05
Exhaustive-OPQ-m64 (self)                             20_577.75    16_703.32    37_281.07       0.8379             NaN         6.05
Exhaustive-OPQ-m128 (query)                           27_165.84     9_405.35    36_571.19       0.9009             NaN         9.11
Exhaustive-OPQ-m128 (self)                            27_165.84    34_998.15    62_163.99       0.8707             NaN         9.11
IVF-OPQ-nl158-m16-np7 (query)                         15_795.90       457.87    16_253.77       0.8729             NaN         4.23
IVF-OPQ-nl158-m16-np12 (query)                        15_795.90       760.12    16_556.02       0.8731             NaN         4.23
IVF-OPQ-nl158-m16-np17 (query)                        15_795.90     1_079.75    16_875.65       0.8731             NaN         4.23
IVF-OPQ-nl158-m16 (self)                              15_795.90     6_778.77    22_574.67       0.8326             NaN         4.23
IVF-OPQ-nl158-m32-np7 (query)                         18_850.70       640.77    19_491.46       0.9158             NaN         4.99
IVF-OPQ-nl158-m32-np12 (query)                        18_850.70     1_063.54    19_914.24       0.9160             NaN         4.99
IVF-OPQ-nl158-m32-np17 (query)                        18_850.70     1_500.40    20_351.10       0.9160             NaN         4.99
IVF-OPQ-nl158-m32 (self)                              18_850.70     8_237.71    27_088.41       0.8906             NaN         4.99
IVF-OPQ-nl158-m64-np7 (query)                         23_715.20     1_022.60    24_737.80       0.9315             NaN         6.52
IVF-OPQ-nl158-m64-np12 (query)                        23_715.20     1_694.39    25_409.59       0.9317             NaN         6.52
IVF-OPQ-nl158-m64-np17 (query)                        23_715.20     2_368.72    26_083.92       0.9317             NaN         6.52
IVF-OPQ-nl158-m64 (self)                              23_715.20    11_129.92    34_845.12       0.9110             NaN         6.52
IVF-OPQ-nl158-m128-np7 (query)                        31_123.10     2_086.78    33_209.88       0.9464             NaN         9.57
IVF-OPQ-nl158-m128-np12 (query)                       31_123.10     3_527.09    34_650.18       0.9466             NaN         9.57
IVF-OPQ-nl158-m128-np17 (query)                       31_123.10     4_943.17    36_066.27       0.9466             NaN         9.57
IVF-OPQ-nl158-m128 (self)                             31_123.10    19_670.53    50_793.63       0.9286             NaN         9.57
IVF-OPQ-nl223-m16-np11 (query)                        12_590.33       664.76    13_255.08       0.8835             NaN         4.42
IVF-OPQ-nl223-m16-np14 (query)                        12_590.33       843.80    13_434.13       0.8835             NaN         4.42
IVF-OPQ-nl223-m16-np21 (query)                        12_590.33     1_240.24    13_830.56       0.8835             NaN         4.42
IVF-OPQ-nl223-m16 (self)                              12_590.33     7_290.69    19_881.02       0.8458             NaN         4.42
IVF-OPQ-nl223-m32-np11 (query)                        15_682.87       903.80    16_586.67       0.9241             NaN         5.18
IVF-OPQ-nl223-m32-np14 (query)                        15_682.87     1_153.27    16_836.13       0.9242             NaN         5.18
IVF-OPQ-nl223-m32-np21 (query)                        15_682.87     1_695.99    17_378.86       0.9242             NaN         5.18
IVF-OPQ-nl223-m32 (self)                              15_682.87     8_806.59    24_489.46       0.9003             NaN         5.18
IVF-OPQ-nl223-m64-np11 (query)                        20_806.24     1_411.33    22_217.57       0.9385             NaN         6.71
IVF-OPQ-nl223-m64-np14 (query)                        20_806.24     1_811.72    22_617.96       0.9385             NaN         6.71
IVF-OPQ-nl223-m64-np21 (query)                        20_806.24     2_672.27    23_478.51       0.9385             NaN         6.71
IVF-OPQ-nl223-m64 (self)                              20_806.24    12_149.53    32_955.77       0.9190             NaN         6.71
IVF-OPQ-nl223-m128-np11 (query)                       28_419.38     2_952.40    31_371.79       0.9511             NaN         9.76
IVF-OPQ-nl223-m128-np14 (query)                       28_419.38     3_755.70    32_175.08       0.9511             NaN         9.76
IVF-OPQ-nl223-m128-np21 (query)                       28_419.38     5_641.32    34_060.71       0.9511             NaN         9.76
IVF-OPQ-nl223-m128 (self)                             28_419.38    21_924.20    50_343.59       0.9330             NaN         9.76
IVF-OPQ-nl316-m16-np15 (query)                        12_926.34       905.72    13_832.07       0.8893             NaN         4.69
IVF-OPQ-nl316-m16-np17 (query)                        12_926.34     1_018.81    13_945.15       0.8893             NaN         4.69
IVF-OPQ-nl316-m16-np25 (query)                        12_926.34     1_477.60    14_403.94       0.8893             NaN         4.69
IVF-OPQ-nl316-m16 (self)                              12_926.34     8_047.26    20_973.60       0.8519             NaN         4.69
IVF-OPQ-nl316-m32-np15 (query)                        15_970.13     1_205.63    17_175.77       0.9272             NaN         5.46
IVF-OPQ-nl316-m32-np17 (query)                        15_970.13     1_350.62    17_320.76       0.9272             NaN         5.46
IVF-OPQ-nl316-m32-np25 (query)                        15_970.13     1_960.80    17_930.93       0.9272             NaN         5.46
IVF-OPQ-nl316-m32 (self)                              15_970.13     9_728.98    25_699.12       0.9047             NaN         5.46
IVF-OPQ-nl316-m64-np15 (query)                        21_674.60     1_830.12    23_504.72       0.9408             NaN         6.98
IVF-OPQ-nl316-m64-np17 (query)                        21_674.60     2_089.80    23_764.41       0.9408             NaN         6.98
IVF-OPQ-nl316-m64-np25 (query)                        21_674.60     3_046.99    24_721.59       0.9408             NaN         6.98
IVF-OPQ-nl316-m64 (self)                              21_674.60    13_401.77    35_076.37       0.9215             NaN         6.98
IVF-OPQ-nl316-m128-np15 (query)                       27_114.37     3_794.98    30_909.34       0.9530             NaN        10.04
IVF-OPQ-nl316-m128-np17 (query)                       27_114.37     4_328.05    31_442.41       0.9531             NaN        10.04
IVF-OPQ-nl316-m128-np25 (query)                       27_114.37     6_363.10    33_477.47       0.9531             NaN        10.04
IVF-OPQ-nl316-m128 (self)                             27_114.37    24_182.23    51_296.60       0.9357             NaN        10.04
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
