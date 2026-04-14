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
Exhaustive (query)                                         3.35     1_513.05     1_516.40       1.0000          1.0000        18.31
Exhaustive (self)                                          3.35    15_582.42    15_585.77       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.75     1_170.13     1_175.88       0.9867          1.0000         9.16
Exhaustive-BF16 (self)                                     5.75    15_904.89    15_910.63       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              390.94        85.63       476.57       0.9758          1.0010         9.19
IVF-BF16-nl273-np16 (query)                              390.94       102.90       493.84       0.9845          1.0002         9.19
IVF-BF16-nl273-np23 (query)                              390.94       145.38       536.32       0.9867          1.0000         9.19
IVF-BF16-nl273 (self)                                    390.94     1_449.22     1_840.15       0.9830          1.0001         9.19
IVF-BF16-nl387-np19 (query)                              744.55        88.56       833.11       0.9802          1.0006         9.21
IVF-BF16-nl387-np27 (query)                              744.55       117.35       861.90       0.9865          1.0000         9.21
IVF-BF16-nl387 (self)                                    744.55     1_190.25     1_934.80       0.9828          1.0001         9.21
IVF-BF16-nl547-np23 (query)                            1_443.58        80.27     1_523.85       0.9773          1.0008         9.23
IVF-BF16-nl547-np27 (query)                            1_443.58        90.08     1_533.66       0.9842          1.0002         9.23
IVF-BF16-nl547-np33 (query)                            1_443.58       107.82     1_551.40       0.9866          1.0000         9.23
IVF-BF16-nl547 (self)                                  1_443.58     1_077.06     2_520.64       0.9828          1.0001         9.23
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
Exhaustive (query)                                         4.04     1_504.22     1_508.26       1.0000          1.0000        18.88
Exhaustive (self)                                          4.04    15_216.69    15_220.72       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                    5.82     1_238.71     1_244.54       0.9240          0.9976         9.44
Exhaustive-BF16 (self)                                     5.82    15_047.77    15_053.59       1.0000          1.0000         9.44
IVF-BF16-nl273-np13 (query)                              388.64        89.09       477.73       0.9163          0.9986         9.48
IVF-BF16-nl273-np16 (query)                              388.64       106.16       494.80       0.9222          0.9978         9.48
IVF-BF16-nl273-np23 (query)                              388.64       149.44       538.08       0.9240          0.9976         9.48
IVF-BF16-nl273 (self)                                    388.64     1_502.82     1_891.46       0.9229          0.9974         9.48
IVF-BF16-nl387-np19 (query)                              741.09        93.53       834.62       0.9200          0.9982         9.49
IVF-BF16-nl387-np27 (query)                              741.09       126.91       868.00       0.9239          0.9976         9.49
IVF-BF16-nl387 (self)                                    741.09     1_275.10     2_016.19       0.9228          0.9974         9.49
IVF-BF16-nl547-np23 (query)                            1_445.83        83.60     1_529.43       0.9183          0.9984         9.51
IVF-BF16-nl547-np27 (query)                            1_445.83        94.61     1_540.44       0.9224          0.9978         9.51
IVF-BF16-nl547-np33 (query)                            1_445.83       111.70     1_557.53       0.9239          0.9976         9.51
IVF-BF16-nl547 (self)                                  1_445.83     1_132.31     2_578.14       0.9228          0.9974         9.51
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
Exhaustive (query)                                         3.09     1_528.02     1_531.12       1.0000          1.0000        18.31
Exhaustive (self)                                          3.09    15_366.77    15_369.86       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.50     1_166.78     1_172.28       0.9649          1.0011         9.16
Exhaustive-BF16 (self)                                     5.50    15_382.75    15_388.25       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              384.37        87.31       471.68       0.9649          1.0011         9.19
IVF-BF16-nl273-np16 (query)                              384.37       103.03       487.41       0.9649          1.0011         9.19
IVF-BF16-nl273-np23 (query)                              384.37       139.01       523.38       0.9649          1.0011         9.19
IVF-BF16-nl273 (self)                                    384.37     1_427.95     1_812.33       0.9561          1.0024         9.19
IVF-BF16-nl387-np19 (query)                              733.87        90.53       824.40       0.9649          1.0011         9.21
IVF-BF16-nl387-np27 (query)                              733.87       119.69       853.56       0.9649          1.0011         9.21
IVF-BF16-nl387 (self)                                    733.87     1_208.87     1_942.74       0.9561          1.0024         9.21
IVF-BF16-nl547-np23 (query)                            1_419.59        82.98     1_502.56       0.9649          1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_419.59        93.39     1_512.98       0.9649          1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_419.59       112.31     1_531.90       0.9649          1.0011         9.23
IVF-BF16-nl547 (self)                                  1_419.59     1_108.05     2_527.64       0.9561          1.0024         9.23
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
Exhaustive (query)                                         3.10     1_520.35     1_523.45       1.0000          1.0000        18.31
Exhaustive (self)                                          3.10    15_280.37    15_283.47       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.62     1_159.61     1_164.24       0.9348          1.0021         9.16
Exhaustive-BF16 (self)                                     4.62    15_487.46    15_492.08       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              386.79        86.16       472.95       0.9348          1.0021         9.19
IVF-BF16-nl273-np16 (query)                              386.79       104.31       491.10       0.9348          1.0021         9.19
IVF-BF16-nl273-np23 (query)                              386.79       144.48       531.27       0.9348          1.0021         9.19
IVF-BF16-nl273 (self)                                    386.79     1_451.23     1_838.02       0.9174          1.0042         9.19
IVF-BF16-nl387-np19 (query)                              733.30        88.67       821.96       0.9348          1.0021         9.21
IVF-BF16-nl387-np27 (query)                              733.30       120.21       853.50       0.9348          1.0021         9.21
IVF-BF16-nl387 (self)                                    733.30     1_202.38     1_935.68       0.9174          1.0042         9.21
IVF-BF16-nl547-np23 (query)                            1_418.23        81.35     1_499.58       0.9348          1.0021         9.23
IVF-BF16-nl547-np27 (query)                            1_418.23        91.28     1_509.51       0.9348          1.0021         9.23
IVF-BF16-nl547-np33 (query)                            1_418.23       109.75     1_527.98       0.9348          1.0021         9.23
IVF-BF16-nl547 (self)                                  1_418.23     1_105.44     2_523.67       0.9174          1.0042         9.23
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
Exhaustive (query)                                        14.22     5_952.54     5_966.76       1.0000          1.0000        73.24
Exhaustive (self)                                         14.22    60_517.57    60_531.79       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   23.15     5_178.43     5_201.58       0.9714          1.0025        36.62
Exhaustive-BF16 (self)                                    23.15    60_698.38    60_721.53       1.0000          1.0000        36.62
IVF-BF16-nl273-np13 (query)                              477.47       307.19       784.66       0.9712          1.0025        36.76
IVF-BF16-nl273-np16 (query)                              477.47       364.37       841.84       0.9713          1.0025        36.76
IVF-BF16-nl273-np23 (query)                              477.47       539.01     1_016.48       0.9714          1.0025        36.76
IVF-BF16-nl273 (self)                                    477.47     5_358.30     5_835.76       0.9637          1.0047        36.76
IVF-BF16-nl387-np19 (query)                              812.55       305.92     1_118.47       0.9713          1.0025        36.81
IVF-BF16-nl387-np27 (query)                              812.55       422.84     1_235.39       0.9714          1.0025        36.81
IVF-BF16-nl387 (self)                                    812.55     4_306.95     5_119.50       0.9637          1.0047        36.81
IVF-BF16-nl547-np23 (query)                            1_600.97       279.11     1_880.08       0.9713          1.0025        36.89
IVF-BF16-nl547-np27 (query)                            1_600.97       320.12     1_921.09       0.9714          1.0025        36.89
IVF-BF16-nl547-np33 (query)                            1_600.97       380.66     1_981.63       0.9714          1.0025        36.89
IVF-BF16-nl547 (self)                                  1_600.97     3_899.98     5_500.94       0.9637          1.0047        36.89
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
Exhaustive (query)                                         3.21     1_534.71     1_537.91       1.0000          1.0000        18.31
Exhaustive (self)                                          3.21    15_389.82    15_393.03       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     9.26       760.64       769.90       0.8011             NaN         4.58
Exhaustive-SQ8 (self)                                      9.26     7_816.13     7_825.39       0.8007             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               434.83        50.30       485.14       0.7779             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               434.83        60.26       495.09       0.7813             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               434.83        81.55       516.38       0.7822             NaN         4.61
IVF-SQ8-nl273 (self)                                     434.83       822.13     1_256.96       0.7819             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               766.08        52.41       818.48       0.7853             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               766.08        70.37       836.45       0.7878             NaN         4.63
IVF-SQ8-nl387 (self)                                     766.08       696.61     1_462.68       0.7872             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_493.31        49.25     1_542.56       0.7972             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_493.31        54.47     1_547.79       0.8002             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_493.31        65.36     1_558.68       0.8012             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_493.31       640.84     2_134.15       0.8007             NaN         4.65
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
Exhaustive (query)                                         4.02     1_509.98     1_514.00       1.0000          1.0000        18.88
Exhaustive (self)                                          4.02    15_357.39    15_361.41       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                    17.29       706.15       723.45       0.8501             NaN         5.15
Exhaustive-SQ8 (self)                                     17.29     7_758.77     7_776.06       0.8497             NaN         5.15
IVF-SQ8-nl273-np13 (query)                               385.17        49.27       434.44       0.8423             NaN         5.19
IVF-SQ8-nl273-np16 (query)                               385.17        57.63       442.80       0.8463             NaN         5.19
IVF-SQ8-nl273-np23 (query)                               385.17        77.71       462.88       0.8473             NaN         5.19
IVF-SQ8-nl273 (self)                                     385.17       789.92     1_175.09       0.8467             NaN         5.19
IVF-SQ8-nl387-np19 (query)                               723.62        51.88       775.50       0.8420             NaN         5.20
IVF-SQ8-nl387-np27 (query)                               723.62        67.49       791.12       0.8449             NaN         5.20
IVF-SQ8-nl387 (self)                                     723.62       677.71     1_401.33       0.8446             NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_420.54        48.27     1_468.81       0.8437             NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_420.54        53.66     1_474.20       0.8467             NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_420.54        61.72     1_482.26       0.8477             NaN         5.22
IVF-SQ8-nl547 (self)                                   1_420.54       627.37     2_047.92       0.8473             NaN         5.22
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
Exhaustive (query)                                         3.13     1_508.33     1_511.46       1.0000          1.0000        18.31
Exhaustive (self)                                          3.13    15_119.97    15_123.10       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     7.54       734.40       741.94       0.6828             NaN         4.58
Exhaustive-SQ8 (self)                                      7.54     7_801.37     7_808.91       0.6835             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               413.31        51.33       464.64       0.6821             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               413.31        63.60       476.91       0.6821             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               413.31        81.18       494.49       0.6820             NaN         4.61
IVF-SQ8-nl273 (self)                                     413.31       807.20     1_220.51       0.6832             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               774.31        56.73       831.04       0.6842             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               774.31        69.98       844.29       0.6842             NaN         4.63
IVF-SQ8-nl387 (self)                                     774.31       701.92     1_476.23       0.6851             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_482.78        51.80     1_534.58       0.6826             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_482.78        61.57     1_544.35       0.6826             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_482.78        67.37     1_550.15       0.6826             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_482.78       660.51     2_143.29       0.6832             NaN         4.65
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
Exhaustive (query)                                         3.28     1_491.66     1_494.95       1.0000          1.0000        18.31
Exhaustive (self)                                          3.28    15_158.13    15_161.41       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    11.34       780.57       791.91       0.4800             NaN         4.58
Exhaustive-SQ8 (self)                                     11.34     8_439.35     8_450.69       0.4862             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               501.79        59.91       561.70       0.4788             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               501.79        97.18       598.97       0.4787             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               501.79        86.20       587.99       0.4786             NaN         4.61
IVF-SQ8-nl273 (self)                                     501.79       869.98     1_371.77       0.4863             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               999.66        59.49     1_059.15       0.4790             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               999.66       109.12     1_108.78       0.4790             NaN         4.63
IVF-SQ8-nl387 (self)                                     999.66       722.88     1_722.54       0.4861             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_547.70        48.89     1_596.58       0.4800             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_547.70        55.09     1_602.79       0.4799             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_547.70        65.56     1_613.26       0.4799             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_547.70       648.09     2_195.78       0.4865             NaN         4.65
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
Exhaustive (query)                                        14.44     6_134.38     6_148.82       1.0000          1.0000        73.24
Exhaustive (self)                                         14.44    63_269.18    63_283.62       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    51.32     1_664.08     1_715.40       0.8081             NaN        18.31
Exhaustive-SQ8 (self)                                     51.32    17_262.50    17_313.82       0.8095             NaN        18.31
IVF-SQ8-nl273-np13 (query)                               518.69       123.70       642.39       0.8062             NaN        18.45
IVF-SQ8-nl273-np16 (query)                               518.69       128.18       646.87       0.8062             NaN        18.45
IVF-SQ8-nl273-np23 (query)                               518.69       178.85       697.54       0.8062             NaN        18.45
IVF-SQ8-nl273 (self)                                     518.69     1_637.00     2_155.69       0.8082             NaN        18.45
IVF-SQ8-nl387-np19 (query)                               798.94       110.02       908.96       0.8062             NaN        18.50
IVF-SQ8-nl387-np27 (query)                               798.94       146.17       945.11       0.8062             NaN        18.50
IVF-SQ8-nl387 (self)                                     798.94     1_373.49     2_172.43       0.8086             NaN        18.50
IVF-SQ8-nl547-np23 (query)                             1_607.19       105.83     1_713.02       0.8078             NaN        18.58
IVF-SQ8-nl547-np27 (query)                             1_607.19       118.00     1_725.19       0.8078             NaN        18.58
IVF-SQ8-nl547-np33 (query)                             1_607.19       142.84     1_750.03       0.8078             NaN        18.58
IVF-SQ8-nl547 (self)                                   1_607.19     1_308.52     2_915.70       0.8097             NaN        18.58
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
Exhaustive (query)                                         4.77     1_794.76     1_799.53       1.0000          1.0000        24.41
Exhaustive (self)                                          4.77     6_033.04     6_037.81       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                664.97       674.67     1_339.64       0.4099             NaN         0.89
Exhaustive-PQ-m16 (self)                                 664.97     2_252.46     2_917.43       0.3229             NaN         0.89
Exhaustive-PQ-m32 (query)                              1_195.63     1_527.02     2_722.66       0.5325             NaN         1.65
Exhaustive-PQ-m32 (self)                               1_195.63     5_061.07     6_256.71       0.4486             NaN         1.65
Exhaustive-PQ-m64 (query)                              2_083.31     4_089.20     6_172.51       0.6898             NaN         3.18
Exhaustive-PQ-m64 (self)                               2_083.31    13_841.40    15_924.71       0.6265             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_258.55       261.15     1_519.70       0.5976             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_258.55       428.86     1_687.41       0.6008             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_258.55       580.57     1_839.12       0.6013             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_258.55     1_915.25     3_173.80       0.5141             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_670.27       451.67     2_121.93       0.7401             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_670.27       768.32     2_438.59       0.7456             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_670.27       994.07     2_664.34       0.7463             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_670.27     3_330.54     5_000.81       0.6861             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_755.69       984.29     3_739.98       0.8664             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_755.69     1_585.14     4_340.82       0.8745             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_755.69     2_173.40     4_929.09       0.8754             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_755.69     8_609.49    11_365.18       0.8464             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                          1_033.41       423.24     1_456.65       0.6040             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                          1_033.41       475.27     1_508.68       0.6047             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                          1_033.41       699.66     1_733.07       0.6049             NaN         1.00
IVF-PQ-nl223-m16 (self)                                1_033.41     2_198.48     3_231.89       0.5193             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_407.09       602.27     2_009.36       0.7472             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_407.09       765.22     2_172.31       0.7487             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_407.09     1_208.43     2_615.52       0.7491             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_407.09     3_786.56     5_193.66       0.6891             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_584.76     1_320.14     3_904.90       0.8733             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_584.76     1_627.66     4_212.42       0.8757             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_584.76     2_503.05     5_087.81       0.8761             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_584.76     7_941.80    10_526.57       0.8474             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_087.51       431.89     1_519.40       0.6065             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_087.51       481.50     1_569.00       0.6070             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_087.51       699.51     1_787.02       0.6072             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_087.51     2_326.51     3_414.02       0.5206             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_418.45       730.94     2_149.39       0.7489             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_418.45       814.65     2_233.11       0.7498             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_418.45     1_173.96     2_592.42       0.7504             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_418.45     4_021.31     5_439.76       0.6916             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_385.35     1_607.09     3_992.44       0.8743             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_385.35     1_818.02     4_203.37       0.8757             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_385.35     2_682.62     5_067.96       0.8767             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_385.35     8_873.79    11_259.13       0.8492             NaN         3.34
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
Exhaustive (query)                                        10.44     4_153.28     4_163.72       1.0000          1.0000        48.83
Exhaustive (self)                                         10.44    14_304.32    14_314.77       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_235.60       703.60     1_939.20       0.2875             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_235.60     2_256.05     3_491.65       0.2108             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_318.24     1_478.07     2_796.31       0.4059             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_318.24     4_913.15     6_231.39       0.3192             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_967.32     3_908.26     5_875.57       0.5236             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_967.32    13_366.11    15_333.43       0.4429             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_395.63       282.08     2_677.71       0.4315             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_395.63       468.80     2_864.43       0.4362             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_395.63       648.40     3_044.03       0.4371             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_395.63     2_134.86     4_530.49       0.3418             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_521.43       446.02     2_967.45       0.5867             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_521.43       758.17     3_279.60       0.5953             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_521.43     1_019.06     3_540.49       0.5973             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_521.43     3_373.55     5_894.98       0.5151             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_248.52       807.36     4_055.89       0.7200             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_248.52     1_316.73     4_565.25       0.7333             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_248.52     1_840.09     5_088.62       0.7362             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_248.52     6_066.29     9_314.81       0.6796             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_649.10       442.58     2_091.68       0.4376             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_649.10       549.48     2_198.58       0.4385             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_649.10       804.31     2_453.41       0.4390             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_649.10     2_600.24     4_249.34       0.3448             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_729.17       640.23     2_369.40       0.5980             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_729.17       811.65     2_540.82       0.6000             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_729.17     1_208.22     2_937.39       0.6009             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_729.17     3_960.74     5_689.91       0.5176             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_662.47     1_199.81     3_862.28       0.7338             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_662.47     1_407.56     4_070.03       0.7369             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_662.47     2_100.71     4_763.18       0.7384             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_662.47     6_930.49     9_592.96       0.6823             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_891.43       546.25     2_437.68       0.4419             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_891.43       649.18     2_540.61       0.4425             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_891.43       914.36     2_805.79       0.4430             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_891.43     2_989.49     4_880.92       0.3475             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_980.07       809.55     2_789.62       0.5995             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_980.07       914.63     2_894.69       0.6008             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_980.07     1_325.80     3_305.86       0.6020             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_980.07     4_439.19     6_419.26       0.5196             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_866.49     1_425.64     4_292.13       0.7357             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_866.49     1_634.91     4_501.40       0.7377             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_866.49     2_396.14     5_262.63       0.7396             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_866.49     7_922.07    10_788.56       0.6836             NaN         3.61
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
Exhaustive (query)                                        20.39     9_807.88     9_828.27       1.0000          1.0000        97.66
Exhaustive (self)                                         20.39    32_400.61    32_421.00       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_183.48       680.25     1_863.73       0.2182             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_183.48     2_248.56     3_432.04       0.1623             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_237.00     1_496.09     3_733.09       0.2969             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_237.00     4_990.84     7_227.85       0.2221             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_468.99     3_932.43     6_401.42       0.4160             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_468.99    13_189.76    15_658.75       0.3341             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_918.78       392.97     4_311.75       0.3110             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_918.78       653.72     4_572.50       0.3141             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_918.78       898.71     4_817.49       0.3147             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_918.78     2_972.09     6_890.87       0.2306             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_937.81       548.35     5_486.16       0.4430             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_937.81       908.18     5_845.99       0.4491             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_937.81     1_257.98     6_195.79       0.4506             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_937.81     4_168.69     9_106.50       0.3611             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_157.83       909.54     6_067.37       0.5978             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_157.83     1_485.75     6_643.58       0.6085             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_157.83     2_048.35     7_206.18       0.6114             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_157.83     6_831.92    11_989.75       0.5369             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_118.48       554.00     2_672.49       0.3161             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_118.48       699.36     2_817.84       0.3165             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_118.48     1_029.40     3_147.88       0.3165             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_118.48     3_405.25     5_523.73       0.2310             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_178.21       791.69     3_969.90       0.4524             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_178.21       999.86     4_178.07       0.4534             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_178.21     1_469.04     4_647.25       0.4536             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_178.21     4_880.24     8_058.45       0.3641             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_407.69     1_242.18     4_649.86       0.6095             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_407.69     1_557.15     4_964.83       0.6119             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_407.69     2_350.83     5_758.52       0.6123             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_407.69     7_723.67    11_131.36       0.5375             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_564.62       734.50     3_299.12       0.3181             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_564.62       823.76     3_388.38       0.3184             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_564.62     1_185.76     3_750.38       0.3187             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_564.62     3_917.31     6_481.94       0.2325             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_853.71     1_077.26     4_930.96       0.4545             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_853.71     1_219.10     5_072.81       0.4553             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_853.71     1_742.65     5_596.36       0.4557             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_853.71     5_848.04     9_701.75       0.3661             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          4_020.14     1_672.33     5_692.47       0.6122             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          4_020.14     1_877.98     5_898.12       0.6139             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          4_020.14     2_716.45     6_736.58       0.6150             NaN         4.17
IVF-PQ-nl316-m64 (self)                                4_020.14     9_034.34    13_054.48       0.5393             NaN         4.17
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
Exhaustive (query)                                         4.61     1_749.20     1_753.81       1.0000          1.0000        24.41
Exhaustive (self)                                          4.61     5_975.90     5_980.50       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                629.81       645.72     1_275.52       0.4611             NaN         0.89
Exhaustive-PQ-m16 (self)                                 629.81     2_161.66     2_791.47       0.3589             NaN         0.89
Exhaustive-PQ-m32 (query)                                992.55     1_454.01     2_446.57       0.6024             NaN         1.65
Exhaustive-PQ-m32 (self)                                 992.55     4_846.98     5_839.54       0.5108             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_961.52     3_919.32     5_880.84       0.7713             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_961.52    13_082.37    15_043.89       0.7164             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_214.83       237.85     1_452.68       0.7301             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_214.83       404.30     1_619.13       0.7305             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_214.83       563.22     1_778.05       0.7305             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_214.83     1_861.43     3_076.26       0.6446             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_590.39       409.91     2_000.30       0.8602             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_590.39       715.55     2_305.94       0.8609             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_590.39     1_040.62     2_631.01       0.8609             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_590.39     3_548.18     5_138.57       0.8170             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_566.55       841.87     3_408.41       0.9526             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_566.55     1_436.17     4_002.72       0.9539             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_566.55     2_034.69     4_601.24       0.9539             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_566.55     6_726.33     9_292.88       0.9398             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            901.22       336.35     1_237.57       0.7326             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            901.22       424.10     1_325.32       0.7329             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            901.22       624.63     1_525.85       0.7329             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  901.22     2_068.18     2_969.39       0.6467             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_290.94       610.12     1_901.07       0.8623             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_290.94       794.73     2_085.67       0.8628             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_290.94     1_177.05     2_468.00       0.8629             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_290.94     3_918.30     5_209.25       0.8195             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_231.40     1_236.92     3_468.32       0.9538             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_231.40     1_578.33     3_809.73       0.9546             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_231.40     2_319.39     4_550.79       0.9547             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_231.40     7_834.52    10_065.92       0.9408             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_176.39       472.61     1_649.00       0.7357             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_176.39       528.05     1_704.44       0.7358             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_176.39       757.01     1_933.40       0.7359             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_176.39     2_512.70     3_689.09       0.6465             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_410.98       759.60     2_170.58       0.8647             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_410.98       869.56     2_280.54       0.8649             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_410.98     1_251.47     2_662.45       0.8650             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_410.98     4_182.72     5_593.70       0.8212             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_364.98     1_607.39     3_972.37       0.9557             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_364.98     1_834.49     4_199.47       0.9561             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_364.98     2_713.63     5_078.61       0.9563             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_364.98     9_042.01    11_406.99       0.9417             NaN         3.34
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
Exhaustive (query)                                        10.14     4_702.94     4_713.07       1.0000          1.0000        48.83
Exhaustive (self)                                         10.14    15_058.41    15_068.55       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_370.55       670.86     2_041.41       0.3249             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_370.55     2_198.72     3_569.27       0.2378             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_220.11     1_468.22     2_688.33       0.4336             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_220.11     4_889.50     6_109.61       0.3345             NaN         1.78
Exhaustive-PQ-m64 (query)                              1_981.61     3_924.96     5_906.57       0.5527             NaN         3.30
Exhaustive-PQ-m64 (self)                               1_981.61    13_260.46    15_242.07       0.4781             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_347.84       269.47     2_617.31       0.5294             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_347.84       453.67     2_801.51       0.5303             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_347.84       633.97     2_981.81       0.5303             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_347.84     2_153.06     4_500.90       0.4148             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_593.53       428.22     3_021.75       0.6733             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_593.53       726.43     3_319.96       0.6755             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_593.53     1_023.16     3_616.69       0.6755             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_593.53     3_401.88     5_995.41       0.6057             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_225.74       763.28     3_989.01       0.8414             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_225.74     1_309.85     4_535.59       0.8457             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_225.74     1_833.61     5_059.35       0.8457             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_225.74     6_108.25     9_333.98       0.8109             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_611.27       417.49     2_028.76       0.5299             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_611.27       529.70     2_140.97       0.5300             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_611.27       778.44     2_389.71       0.5300             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_611.27     2_623.22     4_234.49       0.4077             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_785.38       636.06     2_421.44       0.6775             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_785.38       818.06     2_603.44       0.6776             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_785.38     1_166.73     2_952.11       0.6776             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_785.38     3_862.92     5_648.30       0.6039             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_608.32     1_092.07     3_700.39       0.8475             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_608.32     1_386.30     3_994.62       0.8480             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_608.32     2_054.97     4_663.29       0.8480             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_608.32     6_886.48     9_494.80       0.8127             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_821.77       548.08     2_369.85       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_821.77       616.81     2_438.58       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_821.77       891.18     2_712.95       0.5307             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_821.77     2_821.47     4_643.25       0.3996             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_964.92       805.18     2_770.10       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_964.92       905.35     2_870.27       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_964.92     1_318.67     3_283.59       0.6790             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_964.92     4_479.65     6_444.57       0.6011             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_772.03     1_440.90     4_212.93       0.8491             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_772.03     1_596.78     4_368.81       0.8493             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_772.03     2_338.75     5_110.79       0.8493             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_772.03     7_849.71    10_621.74       0.8132             NaN         3.61
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
Exhaustive (query)                                        20.50    10_038.17    10_058.67       1.0000          1.0000        97.66
Exhaustive (self)                                         20.50    33_807.17    33_827.67       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_946.58       689.22     2_635.80       0.2283             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_946.58     2_288.12     4_234.70       0.1668             NaN         1.26
Exhaustive-PQ-m32 (query)                              3_630.95     1_552.56     5_183.52       0.3094             NaN         2.03
Exhaustive-PQ-m32 (self)                               3_630.95     5_251.17     8_882.12       0.2205             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_590.62     4_103.84     6_694.46       0.4023             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_590.62    13_635.04    16_225.66       0.3046             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           4_582.73       397.11     4_979.84       0.3654             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          4_582.73       645.99     5_228.73       0.3656             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          4_582.73       915.23     5_497.96       0.3656             NaN         1.57
IVF-PQ-nl158-m16 (self)                                4_582.73     3_027.39     7_610.12       0.2381             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           6_306.79       557.03     6_863.82       0.4716             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          6_306.79       933.69     7_240.47       0.4725             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          6_306.79     1_296.53     7_603.32       0.4725             NaN         2.34
IVF-PQ-nl158-m32 (self)                                6_306.79     4_340.47    10_647.26       0.3566             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_342.38       895.70     6_238.08       0.6182             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_342.38     1_514.23     6_856.61       0.6202             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_342.38     2_143.40     7_485.78       0.6202             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_342.38     7_104.44    12_446.81       0.5652             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_317.81       641.82     2_959.62       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_317.81       768.07     3_085.88       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_317.81     1_149.93     3_467.74       0.3634             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_317.81     3_697.43     6_015.23       0.2326             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_426.39       811.11     4_237.50       0.4690             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_426.39     1_031.92     4_458.31       0.4692             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_426.39     1_523.34     4_949.73       0.4692             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_426.39     5_064.13     8_490.52       0.3466             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_590.15     1_298.79     4_888.94       0.6175             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_590.15     1_650.07     5_240.23       0.6181             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_590.15     2_438.68     6_028.83       0.6181             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_590.15     8_113.61    11_703.76       0.5601             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_689.27       761.20     3_450.47       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_689.27       847.52     3_536.79       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_689.27     1_224.26     3_913.54       0.3590             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_689.27     4_008.82     6_698.09       0.2239             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_889.55     1_060.83     4_950.39       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_889.55     1_189.76     5_079.32       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_889.55     1_720.22     5_609.77       0.4674             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_889.55     5_709.29     9_598.84       0.3349             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_883.67     1_663.02     5_546.69       0.6202             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_883.67     1_877.60     5_761.26       0.6204             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_883.67     2_743.07     6_626.73       0.6204             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_883.67     9_116.14    12_999.81       0.5565             NaN         4.17
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
Exhaustive (query)                                         3.00     1_739.35     1_742.34       1.0000          1.0000        24.41
Exhaustive (self)                                          3.00     5_837.11     5_840.11       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                617.62       646.78     1_264.40       0.1806             NaN         0.89
Exhaustive-PQ-m16 (self)                                 617.62     2_198.26     2_815.88       0.3284             NaN         0.89
Exhaustive-PQ-m32 (query)                                990.27     1_453.36     2_443.63       0.2764             NaN         1.65
Exhaustive-PQ-m32 (self)                                 990.27     4_864.77     5_855.04       0.4254             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_940.97     3_958.70     5_899.67       0.5303             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_940.97    13_199.28    15_140.25       0.6555             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_080.33       289.84     1_370.18       0.4309             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_080.33       493.52     1_573.85       0.4309             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_080.33       684.76     1_765.09       0.4310             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_080.33     2_303.84     3_384.18       0.5727             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_490.95       550.90     2_041.85       0.6301             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_490.95     1_023.87     2_514.82       0.6303             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_490.95     1_309.93     2_800.88       0.6303             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_490.95     4_313.90     5_804.85       0.7406             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_403.15     1_155.48     3_558.63       0.8659             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_403.15     2_025.42     4_428.57       0.8663             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_403.15     2_783.09     5_186.24       0.8664             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_403.15     9_265.29    11_668.44       0.9113             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            682.65       349.11     1_031.76       0.4801             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            682.65       435.33     1_117.98       0.4801             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            682.65       627.43     1_310.08       0.4802             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  682.65     2_145.29     2_827.94       0.6156             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_076.48       607.15     1_683.63       0.6748             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_076.48       784.97     1_861.45       0.6748             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_076.48     1_135.58     2_212.06       0.6748             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_076.48     3_858.86     4_935.34       0.7747             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_191.69     1_345.89     3_537.58       0.8839             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_191.69     1_645.97     3_837.66       0.8839             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_191.69     2_482.71     4_674.40       0.8839             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_191.69     8_448.14    10_639.82       0.9208             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                            792.84       471.34     1_264.18       0.4907             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                            792.84       539.44     1_332.29       0.4907             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                            792.84       820.54     1_613.39       0.4907             NaN         1.05
IVF-PQ-nl316-m16 (self)                                  792.84     2_626.31     3_419.16       0.6197             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_272.84       792.56     2_065.40       0.6833             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_272.84       901.40     2_174.23       0.6833             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_272.84     1_313.89     2_586.73       0.6833             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_272.84     4_251.47     5_524.31       0.7782             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_064.22     1_609.84     3_674.06       0.8879             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_064.22     1_822.81     3_887.03       0.8879             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_064.22     2_712.50     4_776.72       0.8879             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_064.22     9_010.10    11_074.32       0.9227             NaN         3.34
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
Exhaustive (query)                                         6.58     4_149.50     4_156.08       1.0000          1.0000        48.83
Exhaustive (self)                                          6.58    14_004.91    14_011.49       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_109.71       660.01     1_769.72       0.1170             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_109.71     2_194.96     3_304.66       0.3086             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_218.52     1_485.95     2_704.47       0.1723             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_218.52     4_896.46     6_114.98       0.4012             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_000.35     3_919.05     5_919.39       0.2817             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_000.35    13_058.69    15_059.04       0.5337             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_086.18       296.35     2_382.52       0.3362             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_086.18       499.67     2_585.85       0.3363             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_086.18       703.80     2_789.98       0.3363             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_086.18     2_341.38     4_427.56       0.5860             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_202.65       482.81     2_685.46       0.4530             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_202.65       824.16     3_026.81       0.4534             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_202.65     1_162.00     3_364.65       0.4534             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_202.65     3_964.83     6_167.48       0.6918             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_997.79       924.11     3_921.89       0.6565             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_997.79     1_586.74     4_584.53       0.6573             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_997.79     2_256.21     5_253.99       0.6574             NaN         3.46
IVF-PQ-nl158-m64 (self)                                2_997.79     7_655.72    10_653.51       0.8277             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_201.66       413.20     1_614.86       0.3576             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_201.66       516.08     1_717.73       0.3577             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_201.66       767.49     1_969.15       0.3578             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_201.66     2_504.80     3_706.45       0.6017             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_319.50       640.64     1_960.14       0.4765             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_319.50       809.27     2_128.77       0.4767             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_319.50     1_200.59     2_520.09       0.4768             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_319.50     3_985.89     5_305.39       0.7014             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_062.61     1_086.04     3_148.64       0.6772             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_062.61     1_375.67     3_438.28       0.6775             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_062.61     2_066.32     4_128.93       0.6777             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_062.61     6_830.23     8_892.84       0.8351             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_227.72       517.68     1_745.40       0.3667             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_227.72       591.77     1_819.49       0.3667             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_227.72       843.78     2_071.50       0.3668             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_227.72     2_816.14     4_043.85       0.6029             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_345.98       837.61     2_183.59       0.4863             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_345.98       941.96     2_287.94       0.4864             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_345.98     1_367.02     2_713.00       0.4865             NaN         2.09
IVF-PQ-nl316-m32 (self)                                1_345.98     4_586.92     5_932.90       0.7000             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_080.39     1_635.83     3_716.22       0.6836             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_080.39     1_824.79     3_905.18       0.6837             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_080.39     2_453.44     4_533.82       0.6839             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_080.39     8_360.26    10_440.64       0.8354             NaN         3.61
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
Exhaustive (query)                                        20.86    10_020.57    10_041.43       1.0000          1.0000        97.66
Exhaustive (self)                                         20.86    33_536.44    33_557.30       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_337.40       709.45     2_046.85       0.0871             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_337.40     2_362.46     3_699.86       0.2928             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_749.77     1_575.29     4_325.07       0.1140             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_749.77     5_268.29     8_018.06       0.3714             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_584.18     4_101.62     6_685.80       0.1674             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_584.18    13_665.22    16_249.40       0.4781             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_558.98       428.52     3_987.50       0.2565             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_558.98       741.17     4_300.15       0.2566             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_558.98     1_004.54     4_563.52       0.2566             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_558.98     3_269.55     6_828.53       0.5497             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_687.35       598.56     5_285.91       0.3199             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_687.35     1_033.05     5_720.40       0.3202             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_687.35     1_441.59     6_128.94       0.3202             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_687.35     4_721.43     9_408.78       0.6347             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_806.15     1_000.11     5_806.27       0.4381             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_806.15     1_710.62     6_516.77       0.4387             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_806.15     2_437.91     7_244.06       0.4387             NaN         3.86
IVF-PQ-nl158-m64 (self)                                4_806.15     8_094.31    12_900.46       0.7341             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_766.29       574.01     2_340.30       0.2752             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_766.29       736.46     2_502.75       0.2753             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_766.29     1_040.90     2_807.19       0.2754             NaN         1.70
IVF-PQ-nl223-m16 (self)                                1_766.29     3_431.42     5_197.71       0.5672             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_934.87       813.02     3_747.89       0.3374             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_934.87     1_035.88     3_970.74       0.3375             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_934.87     1_515.37     4_450.24       0.3376             NaN         2.46
IVF-PQ-nl223-m32 (self)                                2_934.87     5_024.54     7_959.40       0.6430             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_763.78     1_286.47     4_050.25       0.4532             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_763.78     1_615.58     4_379.36       0.4536             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_763.78     2_415.26     5_179.04       0.4538             NaN         3.99
IVF-PQ-nl223-m64 (self)                                2_763.78     8_153.53    10_917.31       0.7363             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_482.14       753.39     2_235.53       0.2874             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_482.14       837.26     2_319.40       0.2874             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_482.14     1_214.79     2_696.93       0.2874             NaN         1.88
IVF-PQ-nl316-m16 (self)                                1_482.14     3_965.93     5_448.07       0.5775             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_851.60     1_049.77     3_901.36       0.3516             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_851.60     1_180.74     4_032.34       0.3516             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_851.60     1_755.64     4_607.24       0.3516             NaN         2.65
IVF-PQ-nl316-m32 (self)                                2_851.60     5_746.20     8_597.79       0.6509             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_784.49     1_659.77     4_444.26       0.4730             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_784.49     1_872.19     4_656.68       0.4732             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_784.49     2_738.71     5_523.20       0.4733             NaN         4.17
IVF-PQ-nl316-m64 (self)                                2_784.49     9_189.65    11_974.15       0.7432             NaN         4.17
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
Exhaustive (query)                                         4.92     1_745.08     1_750.00       1.0000          1.0000        24.41
Exhaustive (self)                                          4.92     5_943.07     5_947.99       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_355.71       313.45     3_669.16       0.2822             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_355.71     1_137.84     4_493.55       0.2114             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_166.27       680.03     3_846.30       0.4067             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_166.27     2_344.71     5_510.98       0.3210             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_984.24       175.52     4_159.76       0.4342             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_984.24       252.87     4_237.11       0.4358             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_984.24       357.01     4_341.25       0.4360             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_984.24     1_215.07     5_199.30       0.3371             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_718.41       242.05     3_960.46       0.5962             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_718.41       391.24     4_109.65       0.5997             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_718.41       539.67     4_258.08       0.6001             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_718.41     1_866.98     5_585.39       0.5153             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          4_751.36       220.06     4_971.42       0.4435             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          4_751.36       282.35     5_033.71       0.4438             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          4_751.36       418.37     5_169.73       0.4438             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                4_751.36     1_434.07     6_185.43       0.3447             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_461.02       351.15     3_812.17       0.6053             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_461.02       450.80     3_911.82       0.6059             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_461.02       670.35     4_131.37       0.6061             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_461.02     2_271.56     5_732.58       0.5192             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_712.81       291.84     4_004.65       0.4460             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_712.81       320.76     4_033.57       0.4461             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_712.81       466.71     4_179.52       0.4462             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_712.81     1_642.32     5_355.13       0.3473             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_598.95       448.79     4_047.74       0.6056             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_598.95       507.47     4_106.42       0.6061             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_598.95       743.86     4_342.81       0.6064             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_598.95     2_573.65     6_172.60       0.5205             NaN         1.11
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
Exhaustive (query)                                        10.18     4_449.02     4_459.20       1.0000          1.0000        48.83
Exhaustive (self)                                         10.18    15_156.20    15_166.38       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             7_296.60       675.88     7_972.48       0.2867             NaN         1.26
Exhaustive-OPQ-m16 (self)                              7_296.60     2_564.37     9_860.96       0.2105             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_659.36     1_522.26     8_181.62       0.4040             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_659.36     5_331.08    11_990.44       0.3190             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_256.57     4_139.12    14_395.70       0.5310             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_256.57    14_148.38    24_404.95       0.4550             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_957.44       276.70     8_234.15       0.4315             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_957.44       462.02     8_419.47       0.4362             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_957.44       645.85     8_603.30       0.4371             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_957.44     2_856.98    10_814.42       0.3417             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          8_694.35       445.63     9_139.98       0.5861             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         8_694.35       745.35     9_439.70       0.5948             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         8_694.35     1_039.17     9_733.52       0.5966             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               8_694.35     3_759.64    12_453.99       0.5161             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_548.61       826.08    12_374.69       0.7226             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_548.61     1_378.23    12_926.84       0.7358             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_548.61     1_898.45    13_447.06       0.7389             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_548.61     6_720.63    18_269.24       0.6854             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         7_376.53       414.43     7_790.96       0.4394             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         7_376.53       524.52     7_901.05       0.4402             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         7_376.53       782.76     8_159.29       0.4406             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               7_376.53     2_945.85    10_322.38       0.3454             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_275.16       643.32     7_918.48       0.5968             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_275.16       820.45     8_095.61       0.5987             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_275.16     1_218.44     8_493.60       0.5997             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_275.16     4_397.30    11_672.46       0.5179             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        11_012.85     1_150.05    12_162.91       0.7377             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        11_012.85     1_457.68    12_470.53       0.7407             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        11_012.85     2_185.67    13_198.53       0.7422             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              11_012.85     7_658.51    18_671.36       0.6880             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         7_543.28       558.84     8_102.12       0.4415             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         7_543.28       637.42     8_180.70       0.4421             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         7_543.28       917.04     8_460.32       0.4427             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               7_543.28     3_447.66    10_990.94       0.3486             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_400.61       838.53     8_239.13       0.5990             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_400.61       958.85     8_359.46       0.6001             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_400.61     1_386.35     8_786.95       0.6015             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_400.61     4_948.32    12_348.93       0.5204             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_078.56     1_469.81    12_548.38       0.7378             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_078.56     1_683.40    12_761.96       0.7399             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_078.56     2_447.31    13_525.87       0.7419             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_078.56     8_556.78    19_635.35       0.6884             NaN         3.86
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
Exhaustive (query)                                        20.52    10_076.18    10_096.70       1.0000          1.0000        97.66
Exhaustive (self)                                         20.52    33_646.97    33_667.49       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             8_093.14       692.50     8_785.64       0.2174             NaN         2.26
Exhaustive-OPQ-m16 (self)                              8_093.14     3_785.18    11_878.32       0.1625             NaN         2.26
Exhaustive-OPQ-m32 (query)                            14_090.22     1_541.82    15_632.04       0.2960             NaN         3.03
Exhaustive-OPQ-m32 (self)                             14_090.22     6_506.17    20_596.40       0.2226             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_394.37     4_156.44    17_550.81       0.4149             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_394.37    15_096.04    28_490.41       0.3338             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_824.02     9_447.89    29_271.91       0.5365             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_824.02    32_859.59    52_683.61       0.4666             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         13_186.41       416.17    13_602.59       0.3111             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        13_186.41       702.37    13_888.79       0.3136             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        13_186.41       962.04    14_148.45       0.3143             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              13_186.41     4_628.37    17_814.78       0.2313             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         19_701.17       566.83    20_268.00       0.4428             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        19_701.17       948.50    20_649.67       0.4490             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        19_701.17     1_308.08    21_009.25       0.4505             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              19_701.17     5_825.03    25_526.20       0.3609             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_395.87       961.28    17_357.16       0.5974             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_395.87     1_579.03    17_974.90       0.6084             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_395.87     2_186.86    18_582.74       0.6114             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_395.87     8_835.99    25_231.86       0.5373             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_558.29     1_730.08    24_288.37       0.7241             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_558.29     2_814.31    25_372.59       0.7399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_558.29     3_892.34    26_450.62       0.7442             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_558.29    14_506.49    37_064.77       0.6963             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         9_696.54       615.58    10_312.12       0.3160             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_696.54       810.71    10_507.26       0.3164             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_696.54     1_203.66    10_900.21       0.3164             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_696.54     5_221.11    14_917.65       0.2317             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        16_689.81       787.92    17_477.73       0.4532             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        16_689.81       998.43    17_688.24       0.4543             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        16_689.81     1_463.57    18_153.38       0.4545             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              16_689.81     6_346.05    23_035.86       0.3663             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_375.96     1_282.73    15_658.69       0.6102             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_375.96     1_625.75    16_001.72       0.6126             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_375.96     2_439.97    16_815.93       0.6131             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_375.96     9_666.80    24_042.76       0.5387             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_738.99     2_307.46    23_046.45       0.7440             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_738.99     2_932.26    23_671.25       0.7475             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_738.99     4_397.02    25_136.01       0.7482             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_738.99    16_118.31    36_857.30       0.7007             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_501.65       763.70    10_265.35       0.3181             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_501.65       865.73    10_367.38       0.3183             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_501.65     1_248.91    10_750.56       0.3185             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_501.65     5_607.93    15_109.58       0.2335             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        15_549.75     1_054.74    16_604.49       0.4546             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        15_549.75     1_185.17    16_734.92       0.4555             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        15_549.75     1_716.74    17_266.49       0.4561             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              15_549.75     7_124.50    22_674.25       0.3671             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_826.47     1_671.73    16_498.20       0.6113             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_826.47     1_879.19    16_705.66       0.6128             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_826.47     2_755.44    17_581.91       0.6139             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_826.47    10_856.49    25_682.96       0.5401             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       21_215.18     2_957.08    24_172.26       0.7452             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       21_215.18     3_368.30    24_583.48       0.7476             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       21_215.18     4_953.03    26_168.22       0.7490             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             21_215.18    17_974.04    39_189.22       0.7015             NaN         8.23
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
Exhaustive (query)                                         5.22     1_811.77     1_817.00       1.0000          1.0000        24.41
Exhaustive (self)                                          5.22     6_154.77     6_159.99       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_382.33       313.41     3_695.74       0.3320             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_382.33     1_129.38     4_511.71       0.2247             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_128.00       678.33     3_806.33       0.4615             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_128.00     2_312.40     5_440.40       0.3241             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_935.71       144.56     4_080.26       0.6714             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_935.71       237.98     4_173.69       0.6716             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_935.71       336.88     4_272.58       0.6716             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_935.71     1_214.31     5_150.02       0.5856             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_797.69       223.84     4_021.53       0.7834             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_797.69       390.44     4_188.13       0.7838             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_797.69       563.61     4_361.30       0.7838             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_797.69     2_013.66     5_811.35       0.7165             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_787.16       213.06     4_000.22       0.6740             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_787.16       273.63     4_060.78       0.6741             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_787.16       406.61     4_193.77       0.6741             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_787.16     1_395.87     5_183.02       0.5895             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_414.08       329.08     3_743.16       0.7857             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_414.08       416.23     3_830.31       0.7860             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_414.08       614.60     4_028.68       0.7860             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_414.08     2_169.34     5_583.41       0.7198             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_702.64       294.55     3_997.19       0.6774             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_702.64       327.95     4_030.59       0.6774             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_702.64       467.07     4_169.71       0.6774             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_702.64     1_654.23     5_356.87       0.5947             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_657.02       456.08     4_113.10       0.7865             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_657.02       515.88     4_172.90       0.7866             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_657.02       763.96     4_420.97       0.7866             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_657.02     2_525.95     6_182.96       0.7209             NaN         1.11
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
Exhaustive (query)                                        10.22     4_401.57     4_411.79       1.0000          1.0000        48.83
Exhaustive (self)                                         10.22    15_665.50    15_675.72       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             6_860.41       682.73     7_543.14       0.3278             NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_860.41     2_572.04     9_432.45       0.2142             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_721.50     1_525.06     8_246.56       0.4531             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_721.50     5_388.13    12_109.63       0.3215             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_303.24     4_190.84    14_494.08       0.5830             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_303.24    14_234.07    24_537.31       0.4945             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          8_013.60       277.57     8_291.17       0.6369             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         8_013.60       464.33     8_477.93       0.6385             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         8_013.60       651.28     8_664.88       0.6385             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               8_013.60     2_530.03    10_543.63       0.5499             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_983.55       467.34     8_450.90       0.7638             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_983.55       730.84     8_714.40       0.7665             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_983.55     1_041.62     9_025.17       0.7665             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_983.55     3_837.23    11_820.79       0.6995             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_528.79       788.39    12_317.17       0.8571             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_528.79     1_349.29    12_878.08       0.8616             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_528.79     1_905.08    13_433.87       0.8616             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_528.79     6_715.95    18_244.74       0.8239             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         7_235.44       411.67     7_647.11       0.6384             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         7_235.44       528.96     7_764.40       0.6385             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         7_235.44       782.93     8_018.37       0.6385             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               7_235.44     2_938.07    10_173.51       0.5521             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_188.85       639.65     7_828.50       0.7653             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_188.85       806.92     7_995.77       0.7656             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_188.85     1_193.16     8_382.01       0.7656             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_188.85     4_342.77    11_531.63       0.6990             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_803.72     1_133.24    11_936.97       0.8626             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_803.72     1_443.89    12_247.61       0.8631             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_803.72     2_154.74    12_958.47       0.8631             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_803.72     7_564.53    18_368.26       0.8249             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         7_429.99       541.37     7_971.36       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         7_429.99       623.11     8_053.11       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         7_429.99       906.39     8_336.38       0.6381             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               7_429.99     3_333.45    10_763.44       0.5527             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_523.73       856.40     8_380.13       0.7646             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_523.73       959.08     8_482.81       0.7647             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_523.73     1_405.92     8_929.65       0.7647             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_523.73     5_032.63    12_556.36       0.7001             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_102.96     1_460.20    12_563.16       0.8632             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_102.96     1_661.08    12_764.04       0.8633             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_102.96     2_437.98    13_540.94       0.8634             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_102.96     8_552.85    19_655.81       0.8255             NaN         3.86
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
Exhaustive (query)                                        20.16    10_041.85    10_062.01       1.0000          1.0000        97.66
Exhaustive (self)                                         20.16    33_739.53    33_759.69       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             8_019.40       709.34     8_728.74       0.2170             NaN         2.26
Exhaustive-OPQ-m16 (self)                              8_019.40     3_752.99    11_772.38       0.1636             NaN         2.26
Exhaustive-OPQ-m32 (query)                            13_996.47     1_542.92    15_539.39       0.3206             NaN         3.03
Exhaustive-OPQ-m32 (self)                             13_996.47     6_576.63    20_573.10       0.2264             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_279.48     4_161.85    17_441.33       0.4637             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_279.48    15_286.28    28_565.76       0.3377             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_711.65     9_405.23    29_116.88       0.6010             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_711.65    32_751.41    52_463.05       0.5072             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         11_029.29       378.58    11_407.87       0.4556             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        11_029.29       640.10    11_669.40       0.4564             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        11_029.29       900.36    11_929.65       0.4564             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              11_029.29     4_437.71    15_467.01       0.3662             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         17_040.93       546.66    17_587.60       0.6078             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        17_040.93       917.29    17_958.23       0.6096             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        17_040.93     1_301.28    18_342.21       0.6096             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              17_040.93     5_847.08    22_888.01       0.5228             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_129.46       889.65    17_019.11       0.7427             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_129.46     1_513.54    17_643.00       0.7458             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_129.46     2_139.93    18_269.39       0.7458             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_129.46     8_630.82    24_760.28       0.6764             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_456.08     1_586.25    24_042.33       0.8356             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_456.08     2_690.22    25_146.30       0.8399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_456.08     3_824.68    26_280.76       0.8399             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_456.08    14_261.46    36_717.54       0.8116             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         9_116.01       577.45     9_693.47       0.4573             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_116.01       725.66     9_841.68       0.4576             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_116.01     1_075.98    10_192.00       0.4576             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_116.01     4_987.73    14_103.75       0.3672             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        15_400.21       816.09    16_216.30       0.6053             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        15_400.21     1_034.55    16_434.76       0.6059             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        15_400.21     1_542.86    16_943.07       0.6059             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              15_400.21     6_487.15    21_887.36       0.5220             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_445.32     1_316.69    15_762.02       0.7434             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_445.32     1_676.03    16_121.35       0.7445             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_445.32     2_511.19    16_956.51       0.7445             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_445.32     9_834.36    24_279.68       0.6776             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_703.50     2_373.53    23_077.04       0.8391             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_703.50     3_031.23    23_734.73       0.8408             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_703.50     4_530.26    25_233.76       0.8408             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_703.50    16_408.77    37_112.27       0.8119             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_309.51       760.14    10_069.65       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_309.51       855.30    10_164.81       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_309.51     1_223.44    10_532.95       0.4569             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_309.51     5_549.64    14_859.15       0.3687             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        16_723.60     1_057.80    17_781.41       0.6054             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        16_723.60     1_192.74    17_916.34       0.6055             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        16_723.60     1_732.76    18_456.36       0.6055             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              16_723.60     7_207.09    23_930.69       0.5215             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_701.22     1_668.38    16_369.59       0.7429             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_701.22     1_892.79    16_594.01       0.7432             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_701.22     2_772.86    17_474.07       0.7432             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_701.22    10_914.42    25_615.63       0.6751             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       21_045.97     2_971.29    24_017.26       0.8407             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       21_045.97     3_399.54    24_445.51       0.8413             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       21_045.97     5_005.94    26_051.91       0.8413             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             21_045.97    18_072.50    39_118.48       0.8127             NaN         8.23
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
Exhaustive (query)                                         3.32     1_865.59     1_868.91       1.0000          1.0000        24.41
Exhaustive (self)                                          3.32     6_395.63     6_398.94       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              3_339.08       315.77     3_654.85       0.1253             NaN         0.57
Exhaustive-OPQ-m8 (self)                               3_339.08     1_147.24     4_486.32       0.2565             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_152.53       672.72     3_825.25       0.1716             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_152.53     2_303.18     5_455.70       0.3354             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_756.66       187.13     3_943.79       0.3011             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_756.66       307.02     4_063.68       0.3010             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_756.66       455.04     4_211.69       0.3010             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_756.66     1_488.55     5_245.21       0.5401             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_634.34       286.28     3_920.62       0.3990             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_634.34       493.95     4_128.29       0.3990             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_634.34       687.17     4_321.50       0.3990             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_634.34     2_422.47     6_056.81       0.6569             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_445.56       213.03     3_658.59       0.3275             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_445.56       273.10     3_718.65       0.3275             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_445.56       405.85     3_851.41       0.3275             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_445.56     1_430.12     4_875.68       0.5946             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_188.72       342.07     3_530.79       0.4273             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_188.72       434.18     3_622.90       0.4273             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_188.72       651.20     3_839.91       0.4273             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_188.72     2_271.51     5_460.23       0.6983             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_428.39       284.99     3_713.37       0.3369             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_428.39       330.78     3_759.17       0.3369             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_428.39       482.73     3_911.12       0.3369             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_428.39     1_664.12     5_092.50       0.6012             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_230.79       465.42     3_696.20       0.4370             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_230.79       524.41     3_755.19       0.4370             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_230.79       780.47     4_011.26       0.4370             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_230.79     2_659.61     5_890.40       0.7007             NaN         1.11
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
Exhaustive (query)                                         6.79     4_469.07     4_475.86       1.0000          1.0000        48.83
Exhaustive (self)                                          6.79    14_889.02    14_895.81       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             6_817.67       681.59     7_499.26       0.1141             NaN         1.26
Exhaustive-OPQ-m16 (self)                              6_817.67     2_758.07     9_575.74       0.3027             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_678.19     1_528.51     8_206.70       0.1529             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_678.19     5_341.73    12_019.92       0.4066             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_304.58     4_149.98    14_454.56       0.2357             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_304.58    14_268.12    24_572.71       0.5577             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          7_782.04       296.50     8_078.54       0.2893             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         7_782.04       509.10     8_291.14       0.2895             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         7_782.04       708.57     8_490.61       0.2895             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               7_782.04     2_723.57    10_505.61       0.6663             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_615.60       495.31     8_110.90       0.3951             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_615.60       849.10     8_464.70       0.3953             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_615.60     1_202.54     8_818.14       0.3953             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_615.60     4_368.46    11_984.06       0.7601             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_143.99       957.07    12_101.05       0.6249             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_143.99     1_661.89    12_805.88       0.6257             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_143.99     2_360.83    13_504.82       0.6258             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_143.99     8_241.56    19_385.55       0.8432             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_995.55       399.54     7_395.09       0.3024             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_995.55       511.65     7_507.20       0.3024             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_995.55       756.93     7_752.48       0.3024             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_995.55     2_888.47     9_884.02       0.6832             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         6_757.90       642.35     7_400.25       0.4094             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         6_757.90       813.85     7_571.75       0.4095             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         6_757.90     1_209.09     7_966.99       0.4096             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               6_757.90     4_417.13    11_175.03       0.7704             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_371.36     1_123.43    11_494.79       0.6372             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_371.36     1_434.20    11_805.56       0.6375             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_371.36     2_157.39    12_528.75       0.6376             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_371.36     7_525.43    17_896.79       0.8497             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_852.70       548.86     7_401.56       0.3114             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_852.70       618.99     7_471.69       0.3114             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_852.70       900.41     7_753.11       0.3114             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_852.70     3_343.49    10_196.19       0.6864             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         6_750.63       835.92     7_586.55       0.4173             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         6_750.63       955.17     7_705.81       0.4173             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         6_750.63     1_389.73     8_140.37       0.4174             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               6_750.63     4_951.78    11_702.41       0.7705             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        10_389.42     1_455.78    11_845.19       0.6493             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        10_389.42     1_656.48    12_045.89       0.6495             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        10_389.42     2_430.52    12_819.93       0.6496             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              10_389.42     8_503.81    18_893.23       0.8495             NaN         3.86
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
Exhaustive (query)                                        19.86    10_259.82    10_279.69       1.0000          1.0000        97.66
Exhaustive (self)                                         19.86    32_631.32    32_651.18       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_296.57       686.33     7_982.90       0.0880             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_296.57     3_653.21    10_949.79       0.2699             NaN         2.26
Exhaustive-OPQ-m32 (query)                            12_182.28     1_485.20    13_667.48       0.1079             NaN         3.03
Exhaustive-OPQ-m32 (self)                             12_182.28     6_330.06    18_512.34       0.3567             NaN         3.03
Exhaustive-OPQ-m64 (query)                            12_668.22     3_998.53    16_666.75       0.1462             NaN         4.55
Exhaustive-OPQ-m64 (self)                             12_668.22    14_960.90    27_629.12       0.4750             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_594.28     9_123.40    28_717.68       0.2377             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_594.28    32_174.45    51_768.73       0.6470             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                          9_595.96       377.80     9_973.76       0.2178             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                         9_595.96       649.37    10_245.33       0.2178             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                         9_595.96       907.89    10_503.85       0.2178             NaN         2.57
IVF-OPQ-nl158-m16 (self)                               9_595.96     4_460.20    14_056.16       0.6326             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         14_268.34       569.92    14_838.26       0.2705             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        14_268.34       958.73    15_227.07       0.2707             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        14_268.34     1_350.80    15_619.14       0.2707             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              14_268.34     5_965.64    20_233.98       0.7168             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         14_791.11       975.31    15_766.41       0.3706             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        14_791.11     1_683.75    16_474.85       0.3710             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        14_791.11     2_390.44    17_181.55       0.3709             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              14_791.11     9_400.10    24_191.20       0.8013             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        20_740.19     1_822.85    22_563.04       0.6139             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       20_740.19     3_140.79    23_880.98       0.6159             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       20_740.19     4_455.07    25_195.26       0.6160             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             20_740.19    16_365.82    37_106.01       0.8737             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         7_431.73       555.41     7_987.13       0.2313             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         7_431.73       709.08     8_140.80       0.2313             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         7_431.73     1_044.57     8_476.30       0.2313             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               7_431.73     4_869.82    12_301.55       0.6444             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        14_357.19       792.41    15_149.60       0.2807             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        14_357.19       996.75    15_353.95       0.2807             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        14_357.19     1_492.75    15_849.94       0.2807             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              14_357.19     6_317.25    20_674.44       0.7233             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        13_691.16     1_286.79    14_977.94       0.3785             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        13_691.16     1_618.64    15_309.80       0.3787             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        13_691.16     2_442.41    16_133.56       0.3787             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              13_691.16     9_698.08    23_389.23       0.8020             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_438.76     2_380.16    22_818.92       0.6148             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_438.76     2_948.88    23_387.64       0.6155             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_438.76     4_447.41    24_886.17       0.6161             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_438.76    16_566.63    37_005.39       0.8738             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         8_224.71       759.82     8_984.54       0.2416             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         8_224.71       908.60     9_133.32       0.2415             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         8_224.71     1_236.14     9_460.86       0.2415             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               8_224.71     5_576.59    13_801.30       0.6599             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        13_317.20     1_052.34    14_369.55       0.2952             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        13_317.20     1_180.87    14_498.08       0.2952             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        13_317.20     1_728.85    15_046.05       0.2951             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              13_317.20     7_173.45    20_490.65       0.7381             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        13_711.74     1_673.43    15_385.17       0.3962             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        13_711.74     1_909.41    15_621.15       0.3964             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        13_711.74     2_795.76    16_507.50       0.3963             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              13_711.74    10_878.43    24_590.17       0.8135             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       20_478.10     2_953.99    23_432.09       0.6401             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       20_478.10     3_401.17    23_879.27       0.6404             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       20_478.10     4_982.46    25_460.56       0.6409             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             20_478.10    18_090.65    38_568.75       0.8804             NaN         8.23
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
