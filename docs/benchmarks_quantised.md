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
Exhaustive (query)                                         3.07     1_608.56     1_611.62       1.0000          1.0000        18.31
Exhaustive (self)                                          3.07    17_337.14    17_340.21       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.49     1_331.09     1_336.58       0.9867          1.0000         9.16
Exhaustive-BF16 (self)                                     5.49    17_817.51    17_823.00       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              446.05        89.66       535.71       0.9758          1.0010         9.19
IVF-BF16-nl273-np16 (query)                              446.05       110.36       556.41       0.9845          1.0002         9.19
IVF-BF16-nl273-np23 (query)                              446.05       155.82       601.87       0.9867          1.0000         9.19
IVF-BF16-nl273 (self)                                    446.05     1_572.24     2_018.29       0.9830          1.0001         9.19
IVF-BF16-nl387-np19 (query)                              818.25        93.78       912.03       0.9802          1.0006         9.21
IVF-BF16-nl387-np27 (query)                              818.25       127.28       945.53       0.9865          1.0000         9.21
IVF-BF16-nl387 (self)                                    818.25     1_297.74     2_115.99       0.9828          1.0001         9.21
IVF-BF16-nl547-np23 (query)                            1_580.44        87.40     1_667.85       0.9773          1.0008         9.23
IVF-BF16-nl547-np27 (query)                            1_580.44        95.87     1_676.31       0.9842          1.0002         9.23
IVF-BF16-nl547-np33 (query)                            1_580.44       114.81     1_695.25       0.9866          1.0000         9.23
IVF-BF16-nl547 (self)                                  1_580.44     1_157.99     2_738.44       0.9828          1.0001         9.23
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
Exhaustive (query)                                         4.04     1_645.61     1_649.65       1.0000          1.0000        18.88
Exhaustive (self)                                          4.04    17_363.19    17_367.23       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                    6.16     1_350.43     1_356.59       0.9240          0.9976         9.44
Exhaustive-BF16 (self)                                     6.16    16_995.38    17_001.54       1.0000          1.0000         9.44
IVF-BF16-nl273-np13 (query)                              507.00        90.38       597.38       0.9163          0.9986         9.48
IVF-BF16-nl273-np16 (query)                              507.00       112.17       619.17       0.9222          0.9978         9.48
IVF-BF16-nl273-np23 (query)                              507.00       153.57       660.58       0.9240          0.9976         9.48
IVF-BF16-nl273 (self)                                    507.00     1_587.53     2_094.54       0.9229          0.9974         9.48
IVF-BF16-nl387-np19 (query)                              800.10        95.94       896.04       0.9200          0.9982         9.49
IVF-BF16-nl387-np27 (query)                              800.10       129.23       929.33       0.9239          0.9976         9.49
IVF-BF16-nl387 (self)                                    800.10     1_330.80     2_130.90       0.9228          0.9974         9.49
IVF-BF16-nl547-np23 (query)                            1_621.94        86.00     1_707.95       0.9183          0.9984         9.51
IVF-BF16-nl547-np27 (query)                            1_621.94       101.79     1_723.73       0.9224          0.9978         9.51
IVF-BF16-nl547-np33 (query)                            1_621.94       115.95     1_737.89       0.9239          0.9976         9.51
IVF-BF16-nl547 (self)                                  1_621.94     1_177.30     2_799.24       0.9228          0.9974         9.51
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
Exhaustive (query)                                         3.23     1_618.73     1_621.96       1.0000          1.0000        18.31
Exhaustive (self)                                          3.23    16_266.99    16_270.22       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.90     1_258.99     1_263.89       0.9649          1.0011         9.16
Exhaustive-BF16 (self)                                     4.90    16_568.46    16_573.36       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              407.98        90.58       498.56       0.9649          1.0011         9.19
IVF-BF16-nl273-np16 (query)                              407.98       106.99       514.97       0.9649          1.0011         9.19
IVF-BF16-nl273-np23 (query)                              407.98       150.55       558.53       0.9649          1.0011         9.19
IVF-BF16-nl273 (self)                                    407.98     1_454.34     1_862.32       0.9561          1.0024         9.19
IVF-BF16-nl387-np19 (query)                              818.30        99.11       917.41       0.9649          1.0011         9.21
IVF-BF16-nl387-np27 (query)                              818.30       134.49       952.79       0.9649          1.0011         9.21
IVF-BF16-nl387 (self)                                    818.30     1_288.96     2_107.26       0.9561          1.0024         9.21
IVF-BF16-nl547-np23 (query)                            1_525.14        87.89     1_613.03       0.9649          1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_525.14        95.87     1_621.00       0.9649          1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_525.14       132.49     1_657.63       0.9649          1.0011         9.23
IVF-BF16-nl547 (self)                                  1_525.14     1_145.63     2_670.77       0.9561          1.0024         9.23
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
Exhaustive (query)                                         3.26     1_594.30     1_597.57       1.0000          1.0000        18.31
Exhaustive (self)                                          3.26    16_240.03    16_243.29       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.65     1_275.49     1_280.14       0.9348          1.0021         9.16
Exhaustive-BF16 (self)                                     4.65    16_487.81    16_492.46       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              404.72        88.19       492.91       0.9348          1.0021         9.19
IVF-BF16-nl273-np16 (query)                              404.72       108.76       513.48       0.9348          1.0021         9.19
IVF-BF16-nl273-np23 (query)                              404.72       147.97       552.69       0.9348          1.0021         9.19
IVF-BF16-nl273 (self)                                    404.72     1_499.35     1_904.07       0.9174          1.0042         9.19
IVF-BF16-nl387-np19 (query)                              768.88        91.44       860.32       0.9348          1.0021         9.21
IVF-BF16-nl387-np27 (query)                              768.88       122.74       891.62       0.9348          1.0021         9.21
IVF-BF16-nl387 (self)                                    768.88     1_242.56     2_011.45       0.9174          1.0042         9.21
IVF-BF16-nl547-np23 (query)                            1_499.29        84.23     1_583.51       0.9348          1.0021         9.23
IVF-BF16-nl547-np27 (query)                            1_499.29        98.95     1_598.24       0.9348          1.0021         9.23
IVF-BF16-nl547-np33 (query)                            1_499.29       112.27     1_611.56       0.9348          1.0021         9.23
IVF-BF16-nl547 (self)                                  1_499.29     1_138.05     2_637.33       0.9174          1.0042         9.23
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
Exhaustive (query)                                        14.33     6_196.62     6_210.95       1.0000          1.0000        73.24
Exhaustive (self)                                         14.33    63_181.76    63_196.09       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   23.36     5_233.10     5_256.47       0.9714          1.0025        36.62
Exhaustive-BF16 (self)                                    23.36    63_506.16    63_529.52       1.0000          1.0000        36.62
IVF-BF16-nl273-np13 (query)                              444.94       300.43       745.37       0.9712          1.0025        36.76
IVF-BF16-nl273-np16 (query)                              444.94       365.15       810.09       0.9713          1.0025        36.76
IVF-BF16-nl273-np23 (query)                              444.94       526.06       970.99       0.9714          1.0025        36.76
IVF-BF16-nl273 (self)                                    444.94     5_391.27     5_836.20       0.9637          1.0047        36.76
IVF-BF16-nl387-np19 (query)                              810.68       307.03     1_117.71       0.9713          1.0025        36.81
IVF-BF16-nl387-np27 (query)                              810.68       421.45     1_232.13       0.9714          1.0025        36.81
IVF-BF16-nl387 (self)                                    810.68     4_334.20     5_144.87       0.9637          1.0047        36.81
IVF-BF16-nl547-np23 (query)                            1_628.91       306.90     1_935.81       0.9713          1.0025        36.89
IVF-BF16-nl547-np27 (query)                            1_628.91       328.91     1_957.83       0.9714          1.0025        36.89
IVF-BF16-nl547-np33 (query)                            1_628.91       394.66     2_023.58       0.9714          1.0025        36.89
IVF-BF16-nl547 (self)                                  1_628.91     3_918.09     5_547.00       0.9637          1.0047        36.89
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
Exhaustive (query)                                         3.12     1_508.31     1_511.43       1.0000          1.0000        18.31
Exhaustive (self)                                          3.12    16_577.26    16_580.38       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     7.12       736.18       743.30       0.8011             NaN         4.58
Exhaustive-SQ8 (self)                                      7.12     7_907.68     7_914.80       0.8007             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               418.39        57.92       476.31       0.7779             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               418.39        64.17       482.56       0.7813             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               418.39        93.03       511.42       0.7822             NaN         4.61
IVF-SQ8-nl273 (self)                                     418.39       838.41     1_256.80       0.7819             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               772.72        53.20       825.92       0.7853             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               772.72        70.44       843.16       0.7878             NaN         4.63
IVF-SQ8-nl387 (self)                                     772.72       688.10     1_460.82       0.7872             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_513.27        50.54     1_563.81       0.7972             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_513.27        57.69     1_570.96       0.8002             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_513.27        66.43     1_579.70       0.8012             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_513.27       635.48     2_148.75       0.8007             NaN         4.65
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
Exhaustive (query)                                         4.11     1_571.84     1_575.94       1.0000          1.0000        18.88
Exhaustive (self)                                          4.11    16_196.21    16_200.32       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                     7.27       719.49       726.76       0.8501             NaN         5.15
Exhaustive-SQ8 (self)                                      7.27     7_584.58     7_591.85       0.8497             NaN         5.15
IVF-SQ8-nl273-np13 (query)                               439.72        50.12       489.84       0.8423             NaN         5.19
IVF-SQ8-nl273-np16 (query)                               439.72        58.39       498.11       0.8463             NaN         5.19
IVF-SQ8-nl273-np23 (query)                               439.72        78.28       517.99       0.8473             NaN         5.19
IVF-SQ8-nl273 (self)                                     439.72       803.34     1_243.06       0.8467             NaN         5.19
IVF-SQ8-nl387-np19 (query)                               731.31        52.59       783.90       0.8420             NaN         5.20
IVF-SQ8-nl387-np27 (query)                               731.31        67.83       799.14       0.8449             NaN         5.20
IVF-SQ8-nl387 (self)                                     731.31       683.16     1_414.47       0.8446             NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_408.31        49.54     1_457.85       0.8437             NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_408.31        53.72     1_462.03       0.8467             NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_408.31        65.38     1_473.68       0.8477             NaN         5.22
IVF-SQ8-nl547 (self)                                   1_408.31       621.60     2_029.90       0.8473             NaN         5.22
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
Exhaustive (query)                                         3.19     1_542.25     1_545.45       1.0000          1.0000        18.31
Exhaustive (self)                                          3.19    15_554.27    15_557.46       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     7.45       735.23       742.68       0.6828             NaN         4.58
Exhaustive-SQ8 (self)                                      7.45     7_825.13     7_832.59       0.6835             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               407.29        52.15       459.44       0.6821             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               407.29        63.65       470.94       0.6821             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               407.29        82.47       489.76       0.6820             NaN         4.61
IVF-SQ8-nl273 (self)                                     407.29       807.35     1_214.64       0.6832             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               761.63        54.34       815.96       0.6842             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               761.63        75.64       837.26       0.6842             NaN         4.63
IVF-SQ8-nl387 (self)                                     761.63       700.87     1_462.50       0.6851             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_461.13        52.49     1_513.63       0.6826             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_461.13        57.81     1_518.94       0.6826             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_461.13        65.32     1_526.45       0.6826             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_461.13       652.45     2_113.58       0.6832             NaN         4.65
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
Exhaustive (query)                                         3.53     1_521.67     1_525.20       1.0000          1.0000        18.31
Exhaustive (self)                                          3.53    15_557.00    15_560.53       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.65       735.84       742.49       0.4800             NaN         4.58
Exhaustive-SQ8 (self)                                      6.65     7_806.10     7_812.75       0.4862             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               398.11        50.63       448.73       0.4788             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               398.11        63.23       461.34       0.4787             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               398.11        87.90       486.01       0.4786             NaN         4.61
IVF-SQ8-nl273 (self)                                     398.11       828.41     1_226.52       0.4863             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               763.22        53.50       816.72       0.4790             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               763.22        70.36       833.58       0.4790             NaN         4.63
IVF-SQ8-nl387 (self)                                     763.22       698.47     1_461.69       0.4861             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_472.77        49.23     1_522.00       0.4800             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_472.77        54.73     1_527.50       0.4799             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_472.77        64.96     1_537.73       0.4799             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_472.77       647.80     2_120.57       0.4865             NaN         4.65
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
Exhaustive (query)                                        14.42     6_190.29     6_204.72       1.0000          1.0000        73.24
Exhaustive (self)                                         14.42    63_068.70    63_083.13       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    42.32     1_685.22     1_727.54       0.8081             NaN        18.31
Exhaustive-SQ8 (self)                                     42.32    17_461.18    17_503.51       0.8095             NaN        18.31
IVF-SQ8-nl273-np13 (query)                               458.87       109.43       568.30       0.8062             NaN        18.45
IVF-SQ8-nl273-np16 (query)                               458.87       124.00       582.87       0.8062             NaN        18.45
IVF-SQ8-nl273-np23 (query)                               458.87       170.01       628.88       0.8062             NaN        18.45
IVF-SQ8-nl273 (self)                                     458.87     1_637.19     2_096.06       0.8082             NaN        18.45
IVF-SQ8-nl387-np19 (query)                               796.41       112.53       908.94       0.8062             NaN        18.50
IVF-SQ8-nl387-np27 (query)                               796.41       149.34       945.75       0.8062             NaN        18.50
IVF-SQ8-nl387 (self)                                     796.41     1_372.35     2_168.76       0.8086             NaN        18.50
IVF-SQ8-nl547-np23 (query)                             1_603.91       105.40     1_709.31       0.8078             NaN        18.58
IVF-SQ8-nl547-np27 (query)                             1_603.91       118.49     1_722.40       0.8078             NaN        18.58
IVF-SQ8-nl547-np33 (query)                             1_603.91       139.84     1_743.75       0.8078             NaN        18.58
IVF-SQ8-nl547 (self)                                   1_603.91     1_297.65     2_901.56       0.8097             NaN        18.58
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
Exhaustive (query)                                         4.73     1_786.86     1_791.59       1.0000          1.0000        24.41
Exhaustive (self)                                          4.73     5_954.51     5_959.24       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                696.76       688.48     1_385.24       0.4099             NaN         0.89
Exhaustive-PQ-m16 (self)                                 696.76     2_241.08     2_937.85       0.3229             NaN         0.89
Exhaustive-PQ-m32 (query)                              1_090.56     1_531.11     2_621.67       0.5325             NaN         1.65
Exhaustive-PQ-m32 (self)                               1_090.56     5_100.77     6_191.33       0.4486             NaN         1.65
Exhaustive-PQ-m64 (query)                              2_075.58     4_117.56     6_193.14       0.6898             NaN         3.18
Exhaustive-PQ-m64 (self)                               2_075.58    13_637.91    15_713.50       0.6265             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_330.14       247.46     1_577.60       0.5976             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_330.14       395.62     1_725.76       0.6008             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_330.14       546.68     1_876.81       0.6013             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_330.14     1_828.19     3_158.33       0.5141             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_662.84       460.37     2_123.22       0.7401             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_662.84       735.51     2_398.35       0.7456             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_662.84       996.36     2_659.20       0.7463             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_662.84     3_328.04     4_990.88       0.6861             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_716.27       987.47     3_703.73       0.8664             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_716.27     1_577.94     4_294.21       0.8745             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_716.27     2_180.70     4_896.97       0.8754             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_716.27     7_161.53     9_877.79       0.8464             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            973.34       389.45     1_362.79       0.6040             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            973.34       465.22     1_438.55       0.6047             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            973.34       690.24     1_663.58       0.6049             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  973.34     2_279.83     3_253.17       0.5193             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_378.31       600.67     1_978.97       0.7472             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_378.31       774.43     2_152.73       0.7487             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_378.31     1_152.67     2_530.98       0.7491             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_378.31     3_780.04     5_158.35       0.6891             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_466.04     1_289.96     3_756.00       0.8733             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_466.04     1_641.81     4_107.85       0.8757             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_466.04     2_467.23     4_933.27       0.8761             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_466.04     8_144.08    10_610.12       0.8474             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_113.93       467.48     1_581.41       0.6065             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_113.93       537.19     1_651.11       0.6070             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_113.93       739.33     1_853.26       0.6072             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_113.93     2_556.14     3_670.07       0.5206             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_517.37       777.31     2_294.67       0.7489             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_517.37       912.26     2_429.63       0.7498             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_517.37     1_304.77     2_822.14       0.7504             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_517.37     4_294.53     5_811.90       0.6916             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_536.03     1_673.95     4_209.98       0.8743             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_536.03     1_887.49     4_423.52       0.8757             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_536.03     2_761.28     5_297.31       0.8767             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_536.03     9_226.69    11_762.72       0.8492             NaN         3.34
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
Exhaustive (query)                                        10.20     4_147.39     4_157.58       1.0000          1.0000        48.83
Exhaustive (self)                                         10.20    13_986.44    13_996.64       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_117.36       669.58     1_786.94       0.2875             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_117.36     2_349.79     3_467.15       0.2108             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_240.68     1_480.01     2_720.69       0.4059             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_240.68     4_885.78     6_126.46       0.3192             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_019.00     3_892.13     5_911.13       0.5236             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_019.00    13_008.81    15_027.81       0.4429             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_375.81       293.27     2_669.09       0.4315             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_375.81       489.49     2_865.30       0.4362             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_375.81       673.73     3_049.55       0.4371             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_375.81     2_202.28     4_578.09       0.3418             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_487.47       448.02     2_935.49       0.5867             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_487.47       758.23     3_245.70       0.5953             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_487.47     1_045.34     3_532.81       0.5973             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_487.47     3_377.33     5_864.79       0.5151             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_261.31       794.49     4_055.79       0.7200             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_261.31     1_311.32     4_572.63       0.7333             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_261.31     1_807.96     5_069.27       0.7362             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_261.31     5_989.99     9_251.29       0.6796             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_610.76       413.49     2_024.25       0.4376             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_610.76       522.22     2_132.98       0.4385             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_610.76       784.13     2_394.89       0.4390             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_610.76     2_539.18     4_149.94       0.3448             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_757.06       627.83     2_384.89       0.5980             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_757.06       792.64     2_549.70       0.6000             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_757.06     1_171.50     2_928.56       0.6009             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_757.06     3_904.60     5_661.66       0.5176             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_499.69     1_101.66     3_601.34       0.7338             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_499.69     1_390.76     3_890.45       0.7369             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_499.69     2_064.58     4_564.27       0.7384             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_499.69     6_859.24     9_358.92       0.6823             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_848.26       544.76     2_393.02       0.4419             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_848.26       624.50     2_472.76       0.4425             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_848.26       892.80     2_741.06       0.4430             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_848.26     2_870.33     4_718.59       0.3475             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_960.76       836.50     2_797.26       0.5995             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_960.76       950.46     2_911.23       0.6008             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_960.76     1_360.48     3_321.24       0.6020             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_960.76     4_525.54     6_486.30       0.5196             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_696.40     1_419.94     4_116.34       0.7357             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_696.40     1_596.79     4_293.20       0.7377             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_696.40     2_317.30     5_013.70       0.7396             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_696.40     7_701.17    10_397.57       0.6836             NaN         3.61
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
Exhaustive (query)                                        20.78     9_640.60     9_661.38       1.0000          1.0000        97.66
Exhaustive (self)                                         20.78    32_219.05    32_239.83       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_200.04       671.26     1_871.31       0.2182             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_200.04     2_234.83     3_434.87       0.1623             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_251.46     1_490.30     3_741.76       0.2969             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_251.46     4_980.76     7_232.21       0.2221             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_535.39     3_903.62     6_439.01       0.4160             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_535.39    13_026.03    15_561.42       0.3341             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_974.41       394.91     4_369.32       0.3110             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_974.41       643.02     4_617.43       0.3141             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_974.41       889.61     4_864.02       0.3147             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_974.41     2_927.37     6_901.79       0.2306             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_979.81       565.47     5_545.28       0.4430             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_979.81       925.43     5_905.24       0.4491             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_979.81     1_298.17     6_277.98       0.4506             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_979.81     4_292.63     9_272.44       0.3611             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_225.17       907.14     6_132.31       0.5978             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_225.17     1_477.39     6_702.55       0.6085             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_225.17     2_048.99     7_274.16       0.6114             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_225.17     6_774.04    11_999.20       0.5369             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_151.56       556.33     2_707.90       0.3161             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_151.56       706.54     2_858.11       0.3165             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_151.56     1_026.35     3_177.92       0.3165             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_151.56     3_378.34     5_529.90       0.2310             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_374.78       809.09     4_183.87       0.4524             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_374.78     1_008.83     4_383.61       0.4534             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_374.78     1_478.56     4_853.34       0.4536             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_374.78     4_956.49     8_331.26       0.3641             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_495.34     1_336.73     4_832.07       0.6095             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_495.34     1_564.70     5_060.04       0.6119             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_495.34     2_310.18     5_805.52       0.6123             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_495.34     7_710.64    11_205.98       0.5375             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_614.69       744.95     3_359.64       0.3181             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_614.69       837.93     3_452.63       0.3184             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_614.69     1_200.80     3_815.50       0.3187             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_614.69     3_925.42     6_540.11       0.2325             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_736.96     1_038.27     4_775.23       0.4545             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_736.96     1_173.21     4_910.16       0.4553             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_736.96     1_699.37     5_436.33       0.4557             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_736.96     5_511.13     9_248.08       0.3661             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_909.33     1_606.41     5_515.74       0.6122             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_909.33     1_820.73     5_730.06       0.6139             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_909.33     2_636.83     6_546.16       0.6150             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_909.33     8_787.18    12_696.51       0.5393             NaN         4.17
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
Exhaustive (query)                                         4.66     1_768.72     1_773.38       1.0000          1.0000        24.41
Exhaustive (self)                                          4.66     6_023.63     6_028.29       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                654.79       668.15     1_322.94       0.4611             NaN         0.89
Exhaustive-PQ-m16 (self)                                 654.79     2_276.90     2_931.69       0.3589             NaN         0.89
Exhaustive-PQ-m32 (query)                              1_057.12     1_509.50     2_566.61       0.6024             NaN         1.65
Exhaustive-PQ-m32 (self)                               1_057.12     5_034.72     6_091.84       0.5108             NaN         1.65
Exhaustive-PQ-m64 (query)                              2_066.62     4_081.24     6_147.86       0.7713             NaN         3.18
Exhaustive-PQ-m64 (self)                               2_066.62    13_624.55    15_691.16       0.7164             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_296.47       237.59     1_534.06       0.7301             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_296.47       398.26     1_694.73       0.7305             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_296.47       553.03     1_849.50       0.7305             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_296.47     1_845.52     3_141.99       0.6446             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_691.50       416.34     2_107.84       0.8602             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_691.50       698.11     2_389.62       0.8609             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_691.50       978.10     2_669.61       0.8609             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_691.50     3_303.28     4_994.78       0.8170             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_703.20       873.37     3_576.57       0.9526             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_703.20     1_484.72     4_187.92       0.9539             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_703.20     2_077.79     4_780.99       0.9539             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_703.20     6_787.49     9_490.69       0.9398             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            955.35       346.61     1_301.96       0.7326             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            955.35       451.73     1_407.08       0.7329             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            955.35       645.72     1_601.07       0.7329             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  955.35     2_205.72     3_161.07       0.6467             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_301.01       640.99     1_942.00       0.8623             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_301.01       751.92     2_052.92       0.8628             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_301.01     1_099.61     2_400.62       0.8629             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_301.01     3_561.21     4_862.21       0.8195             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_312.84     1_233.95     3_546.79       0.9538             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_312.84     1_563.45     3_876.28       0.9546             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_312.84     2_337.64     4_650.48       0.9547             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_312.84     7_707.14    10_019.98       0.9408             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_037.86       452.36     1_490.22       0.7357             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_037.86       510.07     1_547.93       0.7358             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_037.86       742.30     1_780.16       0.7359             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_037.86     2_470.10     3_507.96       0.6465             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_394.05       740.75     2_134.80       0.8647             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_394.05       888.53     2_282.57       0.8649             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_394.05     1_252.60     2_646.65       0.8650             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_394.05     4_076.73     5_470.78       0.8212             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_398.74     1_617.27     4_016.01       0.9557             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_398.74     1_818.64     4_217.38       0.9561             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_398.74     2_656.39     5_055.13       0.9563             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_398.74     8_819.11    11_217.85       0.9417             NaN         3.34
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
Exhaustive (query)                                         9.88     4_185.52     4_195.40       1.0000          1.0000        48.83
Exhaustive (self)                                          9.88    13_970.31    13_980.20       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_143.63       662.44     1_806.06       0.3249             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_143.63     2_203.40     3_347.02       0.2378             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_243.29     1_487.33     2_730.62       0.4336             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_243.29     4_870.33     6_113.62       0.3345             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_997.84     3_915.84     5_913.69       0.5527             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_997.84    13_051.18    15_049.02       0.4781             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_359.36       274.42     2_633.78       0.5294             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_359.36       486.90     2_846.26       0.5303             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_359.36       637.30     2_996.66       0.5303             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_359.36     2_122.60     4_481.96       0.4148             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_483.90       423.61     2_907.51       0.6733             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_483.90       715.75     3_199.65       0.6755             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_483.90     1_012.36     3_496.27       0.6755             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_483.90     3_319.29     5_803.19       0.6057             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_242.18       750.62     3_992.80       0.8414             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_242.18     1_280.13     4_522.31       0.8457             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_242.18     1_817.86     5_060.04       0.8457             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_242.18     6_030.95     9_273.13       0.8109             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_669.20       423.81     2_093.01       0.5299             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_669.20       538.86     2_208.05       0.5300             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_669.20       794.88     2_464.08       0.5300             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_669.20     2_526.84     4_196.04       0.4077             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_777.53       625.11     2_402.64       0.6775             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_777.53       789.24     2_566.77       0.6776             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_777.53     1_173.48     2_951.01       0.6776             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_777.53     3_856.05     5_633.59       0.6039             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_546.25     1_084.11     3_630.36       0.8475             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_546.25     1_372.61     3_918.86       0.8480             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_546.25     2_041.83     4_588.09       0.8480             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_546.25     6_780.20     9_326.45       0.8127             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_883.70       530.60     2_414.30       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_883.70       594.98     2_478.68       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_883.70       863.43     2_747.13       0.5307             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_883.70     2_860.93     4_744.63       0.3996             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_999.48       829.73     2_829.21       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_999.48       933.10     2_932.57       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_999.48     1_362.02     3_361.50       0.6790             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_999.48     4_450.01     6_449.48       0.6011             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_734.65     1_410.01     4_144.66       0.8491             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_734.65     1_589.87     4_324.52       0.8493             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_734.65     2_310.10     5_044.75       0.8493             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_734.65     7_713.22    10_447.87       0.8132             NaN         3.61
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
Exhaustive (query)                                        20.63     9_614.07     9_634.71       1.0000          1.0000        97.66
Exhaustive (self)                                         20.63    32_085.90    32_106.53       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_227.77       682.56     1_910.33       0.2283             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_227.77     2_264.35     3_492.12       0.1668             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_268.06     1_510.61     3_778.67       0.3094             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_268.06     4_975.26     7_243.32       0.2205             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_618.91     3_907.29     6_526.20       0.4023             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_618.91    13_023.61    15_642.51       0.3046             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_925.35       376.80     4_302.15       0.3654             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_925.35       624.89     4_550.24       0.3656             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_925.35       876.55     4_801.90       0.3656             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_925.35     2_937.09     6_862.44       0.2381             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_156.78       541.80     5_698.58       0.4716             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_156.78       911.67     6_068.45       0.4725             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_156.78     1_283.16     6_439.94       0.4725             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_156.78     4_280.16     9_436.94       0.3566             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_175.39       849.87     6_025.25       0.6182             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_175.39     1_431.91     6_607.29       0.6202             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_175.39     2_035.12     7_210.50       0.6202             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_175.39     6_736.86    11_912.24       0.5652             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_199.07       557.58     2_756.65       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_199.07       709.03     2_908.10       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_199.07     1_028.27     3_227.34       0.3634             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_199.07     3_418.36     5_617.43       0.2326             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_484.61       787.15     4_271.76       0.4690             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_484.61       991.26     4_475.87       0.4692             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_484.61     1_464.67     4_949.28       0.4692             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_484.61     4_921.63     8_406.24       0.3466             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_540.14     1_262.76     4_802.90       0.6175             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_540.14     1_590.56     5_130.69       0.6181             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_540.14     2_352.75     5_892.89       0.6181             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_540.14     7_839.54    11_379.67       0.5601             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_523.21       747.43     3_270.63       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_523.21       856.42     3_379.63       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_523.21     1_215.14     3_738.35       0.3590             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_523.21     3_925.04     6_448.24       0.2239             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_745.42     1_057.54     4_802.96       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_745.42     1_191.15     4_936.57       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_745.42     1_729.66     5_475.08       0.4674             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_745.42     5_697.01     9_442.43       0.3349             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_805.28     1_610.76     5_416.04       0.6202             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_805.28     1_823.11     5_628.39       0.6204             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_805.28     2_649.81     6_455.10       0.6204             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_805.28     8_845.05    12_650.33       0.5565             NaN         4.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         2.88     1_812.23     1_815.10       1.0000          1.0000        24.41
Exhaustive (self)                                          2.88     5_964.25     5_967.12       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                628.60       692.03     1_320.64       0.1806             NaN         0.89
Exhaustive-PQ-m16 (self)                                 628.60     2_151.08     2_779.68       0.3284             NaN         0.89
Exhaustive-PQ-m32 (query)                              1_002.85     1_449.12     2_451.97       0.2764             NaN         1.65
Exhaustive-PQ-m32 (self)                               1_002.85     4_872.11     5_874.96       0.4254             NaN         1.65
Exhaustive-PQ-m64 (query)                              2_016.57     3_915.28     5_931.85       0.5303             NaN         3.18
Exhaustive-PQ-m64 (self)                               2_016.57    13_054.45    15_071.02       0.6555             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_087.76       281.05     1_368.81       0.4309             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_087.76       494.07     1_581.83       0.4309             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_087.76       658.56     1_746.32       0.4310             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_087.76     2_209.74     3_297.50       0.5727             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_455.43       527.60     1_983.04       0.6301             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_455.43       934.90     2_390.34       0.6303             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_455.43     1_305.47     2_760.90       0.6303             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_455.43     4_345.29     5_800.72       0.7406             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_417.79     1_154.54     3_572.32       0.8659             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_417.79     2_009.74     4_427.52       0.8663             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_417.79     2_769.20     5_186.98       0.8664             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_417.79     9_261.25    11_679.03       0.9113             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            694.88       343.61     1_038.49       0.4801             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            694.88       428.30     1_123.18       0.4801             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            694.88       638.60     1_333.48       0.4802             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  694.88     2_123.39     2_818.27       0.6156             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_059.84       582.14     1_641.98       0.6748             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_059.84       744.25     1_804.08       0.6748             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_059.84     1_117.93     2_177.76       0.6748             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_059.84     3_719.04     4_778.88       0.7747             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_025.52     1_251.88     3_277.41       0.8839             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_025.52     1_581.31     3_606.84       0.8839             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_025.52     2_386.47     4_412.00       0.8839             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_025.52     7_900.86     9_926.39       0.9208             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                            715.93       448.53     1_164.47       0.4907             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                            715.93       503.33     1_219.27       0.4907             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                            715.93       703.10     1_419.03       0.4907             NaN         1.05
IVF-PQ-nl316-m16 (self)                                  715.93     2_357.02     3_072.95       0.6197             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_107.13       772.63     1_879.76       0.6833             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_107.13       877.65     1_984.79       0.6833             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_107.13     1_292.21     2_399.34       0.6833             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_107.13     4_296.16     5_403.30       0.7782             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_078.59     1_831.32     3_909.90       0.8879             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_078.59     1_976.46     4_055.04       0.8879             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_078.59     2_807.69     4_886.27       0.8879             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_078.59     9_191.64    11_270.22       0.9227             NaN         3.34
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.58     4_133.97     4_140.55       1.0000          1.0000        48.83
Exhaustive (self)                                          6.58    14_365.47    14_372.04       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_189.84       664.47     1_854.31       0.1170             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_189.84     2_202.93     3_392.77       0.3086             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_241.74     1_486.90     2_728.64       0.1723             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_241.74     4_887.89     6_129.63       0.4012             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_003.97     3_949.34     5_953.31       0.2817             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_003.97    13_060.29    15_064.26       0.5337             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_306.03       302.41     2_608.44       0.3362             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_306.03       507.20     2_813.23       0.3363             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_306.03       715.53     3_021.56       0.3363             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_306.03     2_316.20     4_622.23       0.5860             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_246.14       495.11     2_741.25       0.4530             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_246.14       838.32     3_084.47       0.4534             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_246.14     1_190.56     3_436.70       0.4534             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_246.14     3_998.08     6_244.22       0.6918             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_215.17       923.41     4_138.58       0.6565             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_215.17     1_689.88     4_905.05       0.6573             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_215.17     2_239.70     5_454.88       0.6574             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_215.17     7_477.62    10_692.79       0.8277             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_392.10       404.07     1_796.17       0.3576             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_392.10       511.53     1_903.63       0.3577             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_392.10       756.07     2_148.17       0.3578             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_392.10     2_480.23     3_872.34       0.6017             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_347.95       617.85     1_965.80       0.4765             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_347.95       782.22     2_130.17       0.4767             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_347.95     1_163.37     2_511.32       0.4768             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_347.95     3_859.66     5_207.61       0.7014             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_156.04     1_089.28     3_245.32       0.6772             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_156.04     1_373.30     3_529.35       0.6775             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_156.04     2_046.63     4_202.68       0.6777             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_156.04     6_814.39     8_970.44       0.8351             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_430.25       543.90     1_974.15       0.3667             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_430.25       606.88     2_037.13       0.3667             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_430.25       876.77     2_307.02       0.3668             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_430.25     2_906.76     4_337.01       0.6029             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_383.74       809.35     2_193.10       0.4863             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_383.74       913.62     2_297.36       0.4864             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_383.74     1_327.22     2_710.96       0.4865             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_383.74     4_385.37     5_769.12       0.7000             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_147.54     1_400.87     3_548.41       0.6836             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_147.54     1_585.93     3_733.47       0.6837             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_147.54     2_325.10     4_472.64       0.6839             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_147.54     7_700.08     9_847.62       0.8354             NaN         3.61
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.94     9_570.36     9_590.30       1.0000          1.0000        97.66
Exhaustive (self)                                         19.94    32_290.07    32_310.01       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_208.53       672.54     1_881.07       0.0871             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_208.53     2_236.37     3_444.91       0.2928             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_348.62     1_480.84     3_829.45       0.1140             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_348.62     4_948.78     7_297.40       0.3714             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_489.62     3_908.52     6_398.14       0.1674             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_489.62    13_465.96    15_955.58       0.4781             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_599.76       431.36     4_031.12       0.2565             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_599.76       741.46     4_341.22       0.2566             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_599.76     1_099.96     4_699.72       0.2566             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_599.76     3_367.35     6_967.11       0.5497             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_526.32       585.39     5_111.71       0.3199             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_526.32       990.43     5_516.74       0.3202             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_526.32     1_386.14     5_912.45       0.3202             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_526.32     4_673.79     9_200.11       0.6347             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_554.40       971.33     5_525.73       0.4381             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_554.40     1_648.85     6_203.25       0.4387             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_554.40     2_327.29     6_881.69       0.4387             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_554.40     7_791.45    12_345.85       0.7341             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_376.79       558.06     1_934.85       0.2752             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_376.79       698.67     2_075.45       0.2753             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_376.79     1_030.23     2_407.02       0.2754             NaN         1.70
IVF-PQ-nl223-m16 (self)                                1_376.79     3_402.79     4_779.58       0.5672             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_503.82       796.10     3_299.92       0.3374             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_503.82       986.85     3_490.68       0.3375             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_503.82     1_460.52     3_964.34       0.3376             NaN         2.46
IVF-PQ-nl223-m32 (self)                                2_503.82     5_005.71     7_509.54       0.6430             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_667.04     1_242.76     3_909.80       0.4532             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_667.04     1_572.37     4_239.41       0.4536             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_667.04     2_343.96     5_010.99       0.4538             NaN         3.99
IVF-PQ-nl223-m64 (self)                                2_667.04     7_803.19    10_470.23       0.7363             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_413.86       729.60     2_143.46       0.2874             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_413.86       810.07     2_223.93       0.2874             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_413.86     1_162.13     2_575.99       0.2874             NaN         1.88
IVF-PQ-nl316-m16 (self)                                1_413.86     3_859.54     5_273.40       0.5775             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_515.83     1_043.21     3_559.03       0.3516             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_515.83     1_168.64     3_684.47       0.3516             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_515.83     1_689.61     4_205.44       0.3516             NaN         2.65
IVF-PQ-nl316-m32 (self)                                2_515.83     5_548.75     8_064.58       0.6509             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_683.90     1_608.98     4_292.88       0.4730             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_683.90     1_824.22     4_508.13       0.4732             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_683.90     2_652.88     5_336.79       0.4733             NaN         4.17
IVF-PQ-nl316-m64 (self)                                2_683.90     8_841.49    11_525.40       0.7432             NaN         4.17
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
Exhaustive (query)                                         4.39     1_754.24     1_758.63       1.0000          1.0000        24.41
Exhaustive (self)                                          4.39     5_919.25     5_923.64       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_052.71       309.71     3_362.43       0.2822             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_052.71     1_107.96     4_160.67       0.2114             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_191.80       649.33     3_841.13       0.4067             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_191.80     2_239.42     5_431.22       0.3210             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_632.49       161.42     3_793.91       0.4342             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_632.49       245.34     3_877.83       0.4358             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_632.49       335.61     3_968.10       0.4360             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_632.49     1_203.47     4_835.96       0.3371             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_794.83       251.54     4_046.37       0.5962             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_794.83       408.14     4_202.97       0.5997             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_794.83       561.94     4_356.77       0.6001             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_794.83     1_908.96     5_703.79       0.5153             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_457.27       200.49     3_657.76       0.4435             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_457.27       254.67     3_711.94       0.4438             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_457.27       374.45     3_831.73       0.4438             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_457.27     1_314.49     4_771.76       0.3447             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_559.35       339.78     3_899.13       0.6053             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_559.35       421.16     3_980.50       0.6059             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_559.35       626.99     4_186.33       0.6061             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_559.35     2_141.14     5_700.48       0.5192             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_500.82       278.10     3_778.92       0.4460             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_500.82       305.58     3_806.40       0.4461             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_500.82       438.22     3_939.04       0.4462             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_500.82     1_534.45     5_035.27       0.3473             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_602.57       443.58     4_046.15       0.6056             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_602.57       504.25     4_106.82       0.6061             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_602.57       709.65     4_312.21       0.6064             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_602.57     2_466.36     6_068.93       0.5205             NaN         1.11
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
Exhaustive (query)                                        10.05     4_191.91     4_201.96       1.0000          1.0000        48.83
Exhaustive (self)                                         10.05    14_395.01    14_405.06       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             6_251.72       662.15     6_913.87       0.2867             NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_251.72     2_528.86     8_780.58       0.2105             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_488.25     1_473.10     7_961.35       0.4040             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_488.25     5_175.27    11_663.52       0.3190             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_063.14     3_958.56    14_021.70       0.5310             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_063.14    13_599.57    23_662.71       0.4550             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_333.85       269.80     7_603.65       0.4315             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_333.85       445.16     7_779.02       0.4362             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_333.85       618.89     7_952.74       0.4371             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_333.85     2_399.77     9_733.62       0.3417             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_708.76       442.59     8_151.35       0.5861             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_708.76       736.32     8_445.07       0.5948             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_708.76     1_023.39     8_732.15       0.5966             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_708.76     3_740.51    11_449.27       0.5161             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_052.43       796.78    11_849.21       0.7226             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_052.43     1_330.66    12_383.09       0.7358             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_052.43     1_825.71    12_878.13       0.7389             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_052.43     6_454.59    17_507.02       0.6854             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_693.07       424.97     7_118.04       0.4394             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_693.07       518.94     7_212.02       0.4402             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_693.07       763.43     7_456.50       0.4406             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_693.07     2_882.10     9_575.18       0.3454             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_961.99       636.81     7_598.80       0.5968             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_961.99       816.09     7_778.08       0.5987             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_961.99     1_218.65     8_180.64       0.5997             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_961.99     4_326.67    11_288.66       0.5179             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_575.25     1_116.68    11_691.93       0.7377             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_575.25     1_412.50    11_987.75       0.7407             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_575.25     2_104.06    12_679.30       0.7422             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_575.25     7_407.51    17_982.76       0.6880             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_907.78       535.52     7_443.30       0.4415             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_907.78       606.38     7_514.16       0.4421             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_907.78       877.69     7_785.46       0.4427             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_907.78     3_292.76    10_200.54       0.3486             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_197.68       824.55     8_022.22       0.5990             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_197.68       934.84     8_132.52       0.6001             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_197.68     1_358.51     8_556.18       0.6015             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_197.68     4_895.35    12_093.03       0.5204             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_769.18     1_423.00    12_192.18       0.7378             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_769.18     1_605.37    12_374.55       0.7399             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_769.18     2_369.27    13_138.45       0.7419             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_769.18     8_201.60    18_970.78       0.6884             NaN         3.86
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
Exhaustive (query)                                        20.20     9_609.59     9_629.79       1.0000          1.0000        97.66
Exhaustive (self)                                         20.20    32_292.69    32_312.89       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_482.11       673.79     8_155.90       0.2174             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_482.11     3_636.85    11_118.95       0.1625             NaN         2.26
Exhaustive-OPQ-m32 (query)                            12_936.99     1_504.13    14_441.12       0.2960             NaN         3.03
Exhaustive-OPQ-m32 (self)                             12_936.99     6_395.15    19_332.14       0.2226             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_395.31     4_025.76    17_421.07       0.4149             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_395.31    14_712.45    28_107.76       0.3338             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_566.68     9_137.44    28_704.12       0.5365             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_566.68    31_804.84    51_371.52       0.4666             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_455.61       372.04    10_827.65       0.3111             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_455.61       629.10    11_084.71       0.3136             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_455.61       876.57    11_332.18       0.3143             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_455.61     4_293.68    14_749.28       0.2313             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         15_923.51       553.88    16_477.39       0.4428             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        15_923.51       915.92    16_839.43       0.4490             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        15_923.51     1_274.54    17_198.05       0.4505             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              15_923.51     5_646.35    21_569.86       0.3609             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_800.52       923.60    16_724.12       0.5974             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_800.52     1_511.28    17_311.81       0.6084             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_800.52     2_093.18    17_893.70       0.6114             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_800.52     8_431.59    24_232.12       0.5373             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        21_761.09     1_679.12    23_440.21       0.7241             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       21_761.09     2_743.76    24_504.85       0.7399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       21_761.09     3_805.55    25_566.64       0.7442             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             21_761.09    14_067.54    35_828.63       0.6963             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_608.09       551.77     9_159.85       0.3160             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_608.09       697.89     9_305.98       0.3164             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_608.09     1_033.49     9_641.57       0.3164             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_608.09     4_811.05    13_419.14       0.2317             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        14_323.47       776.62    15_100.09       0.4532             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        14_323.47       978.13    15_301.60       0.4543             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        14_323.47     1_435.66    15_759.13       0.4545             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              14_323.47     6_098.66    20_422.12       0.3663             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_820.41     1_251.01    15_071.42       0.6102             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_820.41     1_558.59    15_379.00       0.6126             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_820.41     2_345.24    16_165.65       0.6131             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_820.41     9_210.77    23_031.18       0.5387             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_348.50     2_221.88    22_570.37       0.7440             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_348.50     2_833.29    23_181.79       0.7475             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_348.50     4_247.82    24_596.32       0.7482             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_348.50    15_563.05    35_911.55       0.7007             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_905.67       761.94     9_667.61       0.3181             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_905.67       862.98     9_768.65       0.3183             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_905.67     1_238.91    10_144.58       0.3185             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_905.67     5_478.90    14_384.56       0.2335             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        14_327.41     1_010.23    15_337.65       0.4546             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        14_327.41     1_149.57    15_476.98       0.4555             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        14_327.41     1_640.22    15_967.63       0.4561             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              14_327.41     6_848.70    21_176.11       0.3671             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_300.95     1_613.21    15_914.16       0.6113             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_300.95     1_835.05    16_136.00       0.6128             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_300.95     2_697.48    16_998.43       0.6139             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_300.95    10_370.58    24_671.53       0.5401             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       20_743.13     2_857.53    23_600.66       0.7452             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       20_743.13     3_238.53    23_981.66       0.7476             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       20_743.13     4_758.46    25_501.59       0.7490             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             20_743.13    17_247.18    37_990.31       0.7015             NaN         8.23
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
Exhaustive (query)                                         5.11     1_792.47     1_797.58       1.0000          1.0000        24.41
Exhaustive (self)                                          5.11     6_039.31     6_044.41       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_154.65       320.06     3_474.70       0.3320             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_154.65     1_095.14     4_249.78       0.2247             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_069.25       657.33     3_726.58       0.4615             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_069.25     2_225.31     5_294.57       0.3241             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_616.76       142.77     3_759.53       0.6714             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_616.76       244.84     3_861.60       0.6716             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_616.76       333.90     3_950.66       0.6716             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_616.76     1_175.57     4_792.33       0.5856             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_636.07       232.20     3_868.27       0.7834             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_636.07       390.90     4_026.97       0.7838             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_636.07       550.49     4_186.56       0.7838             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_636.07     1_877.92     5_513.99       0.7165             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_342.90       209.32     3_552.22       0.6740             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_342.90       264.56     3_607.46       0.6741             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_342.90       391.81     3_734.71       0.6741             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_342.90     1_382.01     4_724.91       0.5895             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_309.65       327.68     3_637.33       0.7857             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_309.65       424.82     3_734.47       0.7860             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_309.65       605.50     3_915.16       0.7860             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_309.65     2_111.93     5_421.59       0.7198             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_466.18       277.50     3_743.68       0.6774             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_466.18       306.02     3_772.20       0.6774             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_466.18       441.32     3_907.50       0.6774             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_466.18     1_532.40     4_998.58       0.5947             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_521.70       431.55     3_953.25       0.7865             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_521.70       484.34     4_006.04       0.7866             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_521.70       739.03     4_260.73       0.7866             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_521.70     2_442.04     5_963.74       0.7209             NaN         1.11
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
Exhaustive (query)                                         9.87     4_205.16     4_215.02       1.0000          1.0000        48.83
Exhaustive (self)                                          9.87    14_371.45    14_381.32       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             6_411.58       680.84     7_092.42       0.3278             NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_411.58     2_501.23     8_912.81       0.2142             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_547.56     1_458.67     8_006.23       0.4531             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_547.56     5_215.56    11_763.12       0.3215             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_944.07     3_956.97    13_901.04       0.5830             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_944.07    13_567.82    23_511.89       0.4945             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_366.50       259.11     7_625.61       0.6369             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_366.50       431.15     7_797.65       0.6385             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_366.50       602.76     7_969.26       0.6385             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_366.50     2_376.34     9_742.83       0.5499             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_672.45       430.63     8_103.09       0.7638             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_672.45       718.53     8_390.99       0.7665             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_672.45     1_003.52     8_675.97       0.7665             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_672.45     3_663.02    11_335.47       0.6995             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_250.45       753.40    12_003.84       0.8571             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_250.45     1_290.06    12_540.50       0.8616             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_250.45     1_824.63    13_075.08       0.8616             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_250.45     6_487.96    17_738.40       0.8239             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_945.91       397.54     7_343.45       0.6384             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_945.91       505.40     7_451.31       0.6385             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_945.91       747.35     7_693.25       0.6385             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_945.91     2_829.68     9_775.58       0.5521             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_974.96       617.76     7_592.72       0.7653             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_974.96       788.77     7_763.74       0.7656             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_974.96     1_169.12     8_144.08       0.7656             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_974.96     4_225.17    11_200.14       0.6990             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_449.75     1_102.36    11_552.10       0.8626             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_449.75     1_393.53    11_843.27       0.8631             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_449.75     2_076.65    12_526.39       0.8631             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_449.75     7_314.97    17_764.72       0.8249             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_980.22       528.95     7_509.16       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_980.22       597.29     7_577.51       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_980.22       868.35     7_848.57       0.6381             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_980.22     3_259.02    10_239.23       0.5527             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_216.31       805.55     8_021.86       0.7646             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_216.31       905.90     8_122.21       0.7647             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_216.31     1_325.93     8_542.24       0.7647             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_216.31     4_767.51    11_983.81       0.7001             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_793.53     1_411.32    12_204.85       0.8632             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_793.53     1_600.00    12_393.54       0.8633             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_793.53     2_354.23    13_147.76       0.8634             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_793.53     8_292.71    19_086.25       0.8255             NaN         3.86
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
Exhaustive (query)                                        19.79     9_622.44     9_642.23       1.0000          1.0000        97.66
Exhaustive (self)                                         19.79    32_154.66    32_174.45       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_496.22       672.61     8_168.83       0.2170             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_496.22     3_645.78    11_142.00       0.1636             NaN         2.26
Exhaustive-OPQ-m32 (query)                            12_743.94     1_479.03    14_222.96       0.3206             NaN         3.03
Exhaustive-OPQ-m32 (self)                             12_743.94     6_330.44    19_074.38       0.2264             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_010.63     4_033.18    17_043.81       0.4637             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_010.63    14_803.72    27_814.35       0.3377             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_097.68     9_084.55    28_182.23       0.6010             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_097.68    31_743.93    50_841.61       0.5072             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_571.40       374.25    10_945.64       0.4556             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_571.40       664.85    11_236.25       0.4564             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_571.40       894.76    11_466.15       0.4564             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_571.40     4_438.21    15_009.61       0.3662             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         15_540.06       544.48    16_084.55       0.6078             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        15_540.06       918.41    16_458.47       0.6096             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        15_540.06     1_307.97    16_848.03       0.6096             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              15_540.06     5_603.35    21_143.41       0.5228             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_682.64       864.69    16_547.32       0.7427             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_682.64     1_457.82    17_140.45       0.7458             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_682.64     2_055.28    17_737.92       0.7458             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_682.64     8_264.98    23_947.62       0.6764             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        21_701.83     1_539.20    23_241.03       0.8356             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       21_701.83     2_596.92    24_298.75       0.8399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       21_701.83     3_694.83    25_396.66       0.8399             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             21_701.83    13_862.75    35_564.58       0.8116             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_346.94       554.95     8_901.89       0.4573             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_346.94       701.74     9_048.68       0.4576             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_346.94     1_038.41     9_385.35       0.4576             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_346.94     4_844.92    13_191.86       0.3672             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        13_981.40       797.92    14_779.32       0.6053             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        13_981.40     1_004.71    14_986.12       0.6059             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        13_981.40     1_467.40    15_448.80       0.6059             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              13_981.40     6_332.90    20_314.30       0.5220             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_605.51     1_259.74    15_865.25       0.7434             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_605.51     1_602.36    16_207.87       0.7445             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_605.51     2_397.42    17_002.93       0.7445             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_605.51     9_444.88    24_050.39       0.6776             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_092.26     2_280.14    22_372.39       0.8391             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_092.26     2_900.91    22_993.17       0.8408             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_092.26     4_338.45    24_430.71       0.8408             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_092.26    15_955.97    36_048.22       0.8119             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_763.02       745.77     9_508.79       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_763.02       847.58     9_610.60       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_763.02     1_215.99     9_979.01       0.4569             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_763.02     5_512.91    14_275.92       0.3687             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        14_189.82     1_038.50    15_228.32       0.6054             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        14_189.82     1_171.90    15_361.72       0.6055             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        14_189.82     1_695.20    15_885.02       0.6055             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              14_189.82     7_003.50    21_193.32       0.5215             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_248.43     1_610.76    15_859.18       0.7429             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_248.43     1_824.02    16_072.45       0.7432             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_248.43     2_656.85    16_905.28       0.7432             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_248.43    10_410.97    24_659.40       0.6751             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       20_983.34     2_898.16    23_881.50       0.8407             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       20_983.34     3_264.00    24_247.34       0.8413             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       20_983.34     4_801.97    25_785.31       0.8413             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             20_983.34    17_600.73    38_584.07       0.8127             NaN         8.23
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

##### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 128 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         2.88     1_814.08     1_816.95       1.0000          1.0000        24.41
Exhaustive (self)                                          2.88     6_207.89     6_210.77       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_036.35       305.26     3_341.61       0.1253             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_036.35     1_098.14     4_134.48       0.2565             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_057.38       650.98     3_708.36       0.1716             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_057.38     2_233.97     5_291.35       0.3354             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_456.41       166.30     3_622.71       0.3011             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_456.41       281.72     3_738.13       0.3010             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_456.41       387.97     3_844.39       0.3010             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_456.41     1_357.29     4_813.70       0.5401             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_489.12       273.03     3_762.16       0.3990             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_489.12       469.49     3_958.62       0.3990             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_489.12       652.12     4_141.24       0.3990             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_489.12     2_227.77     5_716.89       0.6569             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_061.43       213.21     3_274.64       0.3275             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_061.43       272.17     3_333.60       0.3275             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_061.43       401.46     3_462.89       0.3275             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_061.43     1_421.61     4_483.04       0.5946             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_109.17       345.33     3_454.50       0.4273             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_109.17       450.90     3_560.07       0.4273             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_109.17       658.02     3_767.18       0.4273             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_109.17     2_339.61     5_448.78       0.6983             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_254.20       293.12     3_547.32       0.3369             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_254.20       331.00     3_585.19       0.3369             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_254.20       477.96     3_732.16       0.3369             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_254.20     1_670.12     4_924.32       0.6012             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_252.72       446.41     3_699.13       0.4370             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_252.72       499.77     3_752.49       0.4370             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_252.72       728.05     3_980.76       0.4370             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_252.72     2_561.44     5_814.15       0.7007             NaN         1.11
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.60     4_439.82     4_446.42       1.0000          1.0000        48.83
Exhaustive (self)                                          6.60    14_204.03    14_210.63       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             6_457.51       672.40     7_129.91       0.1141             NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_457.51     2_501.45     8_958.97       0.3027             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_433.79     1_458.68     7_892.47       0.1529             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_433.79     5_280.60    11_714.40       0.4066             NaN         2.03
Exhaustive-OPQ-m64 (query)                             9_917.82     3_961.63    13_879.45       0.2357             NaN         3.55
Exhaustive-OPQ-m64 (self)                              9_917.82    13_572.11    23_489.93       0.5577             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_159.05       302.62     7_461.66       0.2893             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_159.05       505.74     7_664.79       0.2895             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_159.05       711.57     7_870.62       0.2895             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_159.05     2_717.59     9_876.64       0.6663             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_414.63       480.50     7_895.13       0.3951             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_414.63       824.98     8_239.60       0.3953             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_414.63     1_237.00     8_651.63       0.3953             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_414.63     4_217.16    11_631.78       0.7601             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         10_908.21       924.76    11_832.97       0.6249             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        10_908.21     1_605.41    12_513.62       0.6257             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        10_908.21     2_270.84    13_179.05       0.6258             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              10_908.21     7_978.64    18_886.85       0.8432             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_269.69       398.05     6_667.74       0.3024             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_269.69       513.97     6_783.67       0.3024             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_269.69       745.37     7_015.06       0.3024             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_269.69     2_821.11     9_090.81       0.6832             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_870.29       623.35     7_493.64       0.4094             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_870.29       797.38     7_667.67       0.4095             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_870.29     1_177.13     8_047.42       0.4096             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_870.29     4_299.18    11_169.47       0.7704             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_407.97     1_084.21    11_492.18       0.6372             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_407.97     1_385.95    11_793.91       0.6375             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_407.97     2_308.83    12_716.80       0.6376             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_407.97     7_689.94    18_097.91       0.8497             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_854.73       522.02     7_376.75       0.3114             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_854.73       598.69     7_453.42       0.3114             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_854.73       871.46     7_726.19       0.3114             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_854.73     3_267.35    10_122.08       0.6864             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         6_646.21       832.24     7_478.46       0.4173             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         6_646.21       947.57     7_593.78       0.4173             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         6_646.21     1_381.63     8_027.84       0.4174             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               6_646.21     4_928.39    11_574.60       0.7705             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_110.81     1_520.85    11_631.66       0.6493             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_110.81     1_606.80    11_717.61       0.6495             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_110.81     2_352.40    12_463.22       0.6496             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_110.81     8_812.60    18_923.42       0.8495             NaN         3.86
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        18.96     9_577.06     9_596.02       1.0000          1.0000        97.66
Exhaustive (self)                                         18.96    32_381.32    32_400.28       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_509.35       690.12     8_199.47       0.0880             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_509.35     3_682.96    11_192.31       0.2699             NaN         2.26
Exhaustive-OPQ-m32 (query)                            12_964.53     1_481.17    14_445.70       0.1079             NaN         3.03
Exhaustive-OPQ-m32 (self)                             12_964.53     6_324.98    19_289.51       0.3567             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_874.34     3_977.97    16_852.31       0.1462             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_874.34    14_629.96    27_504.30       0.4750             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_178.09     9_088.24    28_266.33       0.2377             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_178.09    34_166.26    53_344.35       0.6470             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                          9_815.74       397.94    10_213.68       0.2178             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                         9_815.74       668.38    10_484.12       0.2178             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                         9_815.74       936.86    10_752.60       0.2178             NaN         2.57
IVF-OPQ-nl158-m16 (self)                               9_815.74     4_520.04    14_335.78       0.6326             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         14_966.65       564.69    15_531.34       0.2705             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        14_966.65       975.00    15_941.65       0.2707             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        14_966.65     1_351.89    16_318.54       0.2707             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              14_966.65     5_930.45    20_897.10       0.7168             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_032.09       973.24    16_005.33       0.3706             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_032.09     1_681.85    16_713.94       0.3710             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_032.09     2_381.75    17_413.84       0.3709             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_032.09     9_384.65    24_416.74       0.8013             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        21_143.51     1_815.19    22_958.71       0.6139             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       21_143.51     3_127.47    24_270.99       0.6159             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       21_143.51     4_427.33    25_570.85       0.6160             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             21_143.51    16_443.99    37_587.51       0.8737             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         7_597.37       545.99     8_143.36       0.2313             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         7_597.37       689.53     8_286.90       0.2313             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         7_597.37     1_014.88     8_612.25       0.2313             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               7_597.37     4_734.21    12_331.58       0.6444             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        12_931.19       778.68    13_709.87       0.2807             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        12_931.19       979.35    13_910.54       0.2807             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        12_931.19     1_448.57    14_379.76       0.2807             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              12_931.19     6_228.73    19_159.92       0.7233             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_079.59     1_288.41    14_368.01       0.3785             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_079.59     1_571.09    14_650.68       0.3787             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_079.59     2_367.70    15_447.29       0.3787             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_079.59     9_295.52    22_375.11       0.8020             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       19_416.34     2_243.22    21_659.56       0.6148             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       19_416.34     2_855.04    22_271.38       0.6155             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       19_416.34     4_315.24    23_731.58       0.6161             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             19_416.34    15_753.23    35_169.57       0.8738             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         7_760.50       777.39     8_537.89       0.2416             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         7_760.50       861.10     8_621.60       0.2415             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         7_760.50     1_239.10     8_999.60       0.2415             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               7_760.50     5_545.25    13_305.75       0.6599             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        13_011.03     1_032.37    14_043.40       0.2952             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        13_011.03     1_163.53    14_174.56       0.2952             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        13_011.03     1_692.88    14_703.91       0.2951             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              13_011.03     6_961.51    19_972.54       0.7381             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        13_091.35     1_615.62    14_706.97       0.3962             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        13_091.35     1_848.42    14_939.78       0.3964             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        13_091.35     2_675.60    15_766.95       0.3963             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              13_091.35    10_408.60    23_499.95       0.8135             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       19_374.67     2_901.83    22_276.50       0.6401             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       19_374.67     3_270.42    22_645.09       0.6404             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       19_374.67     4_796.09    24_170.76       0.6409             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             19_374.67    17_546.73    36_921.41       0.8804             NaN         8.23
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
