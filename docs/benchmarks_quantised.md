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
Exhaustive (query)                                         3.03     1_505.21     1_508.24       1.0000          1.0000        18.31
Exhaustive (self)                                          3.03    16_789.71    16_792.74       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    6.15     1_289.93     1_296.09       0.9867          1.0000         9.16
Exhaustive-BF16 (self)                                     6.15    16_674.60    16_680.75       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              414.40        88.06       502.45       0.9758          1.0010         9.19
IVF-BF16-nl273-np16 (query)                              414.40       106.44       520.84       0.9845          1.0002         9.19
IVF-BF16-nl273-np23 (query)                              414.40       154.00       568.40       0.9867          1.0000         9.19
IVF-BF16-nl273 (self)                                    414.40     1_479.94     1_894.34       0.9830          1.0001         9.19
IVF-BF16-nl387-np19 (query)                              784.27        90.78       875.05       0.9802          1.0006         9.21
IVF-BF16-nl387-np27 (query)                              784.27       132.74       917.01       0.9865          1.0000         9.21
IVF-BF16-nl387 (self)                                    784.27     1_225.97     2_010.25       0.9828          1.0001         9.21
IVF-BF16-nl547-np23 (query)                            1_482.93        82.08     1_565.01       0.9773          1.0008         9.23
IVF-BF16-nl547-np27 (query)                            1_482.93        91.61     1_574.54       0.9842          1.0002         9.23
IVF-BF16-nl547-np33 (query)                            1_482.93       108.36     1_591.30       0.9866          1.0000         9.23
IVF-BF16-nl547 (self)                                  1_482.93     1_095.92     2_578.85       0.9828          1.0001         9.23
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
Exhaustive (query)                                         4.03     1_604.20     1_608.22       1.0000          1.0000        18.88
Exhaustive (self)                                          4.03    16_142.07    16_146.09       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                    5.94     1_247.40     1_253.34       0.9240          0.9976         9.44
Exhaustive-BF16 (self)                                     5.94    15_931.87    15_937.82       1.0000          1.0000         9.44
IVF-BF16-nl273-np13 (query)                              376.53        91.54       468.07       0.9163          0.9986         9.48
IVF-BF16-nl273-np16 (query)                              376.53       111.43       487.96       0.9222          0.9978         9.48
IVF-BF16-nl273-np23 (query)                              376.53       151.85       528.38       0.9240          0.9976         9.48
IVF-BF16-nl273 (self)                                    376.53     1_508.53     1_885.07       0.9229          0.9974         9.48
IVF-BF16-nl387-np19 (query)                              706.51        94.92       801.43       0.9200          0.9982         9.49
IVF-BF16-nl387-np27 (query)                              706.51       129.21       835.72       0.9239          0.9976         9.49
IVF-BF16-nl387 (self)                                    706.51     1_276.33     1_982.83       0.9228          0.9974         9.49
IVF-BF16-nl547-np23 (query)                            1_377.39        93.37     1_470.77       0.9183          0.9984         9.51
IVF-BF16-nl547-np27 (query)                            1_377.39        95.66     1_473.06       0.9224          0.9978         9.51
IVF-BF16-nl547-np33 (query)                            1_377.39       112.89     1_490.29       0.9239          0.9976         9.51
IVF-BF16-nl547 (self)                                  1_377.39     1_136.98     2_514.37       0.9228          0.9974         9.51
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
Exhaustive (query)                                         3.26     1_532.48     1_535.74       1.0000          1.0000        18.31
Exhaustive (self)                                          3.26    15_503.48    15_506.74       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    5.21     1_158.66     1_163.86       0.9649          1.0011         9.16
Exhaustive-BF16 (self)                                     5.21    15_661.29    15_666.50       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              391.35        88.04       479.39       0.9649          1.0011         9.19
IVF-BF16-nl273-np16 (query)                              391.35       105.59       496.94       0.9649          1.0011         9.19
IVF-BF16-nl273-np23 (query)                              391.35       140.02       531.37       0.9649          1.0011         9.19
IVF-BF16-nl273 (self)                                    391.35     1_406.78     1_798.13       0.9561          1.0024         9.19
IVF-BF16-nl387-np19 (query)                              741.20        91.14       832.34       0.9649          1.0011         9.21
IVF-BF16-nl387-np27 (query)                              741.20       120.37       861.57       0.9649          1.0011         9.21
IVF-BF16-nl387 (self)                                    741.20     1_209.77     1_950.97       0.9561          1.0024         9.21
IVF-BF16-nl547-np23 (query)                            1_432.65        83.38     1_516.02       0.9649          1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_432.65        95.02     1_527.67       0.9649          1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_432.65       110.82     1_543.47       0.9649          1.0011         9.23
IVF-BF16-nl547 (self)                                  1_432.65     1_110.98     2_543.62       0.9561          1.0024         9.23
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
Exhaustive (query)                                         3.17     1_506.02     1_509.18       1.0000          1.0000        18.31
Exhaustive (self)                                          3.17    15_389.83    15_393.00       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                    4.53     1_158.27     1_162.79       0.9348          1.0021         9.16
Exhaustive-BF16 (self)                                     4.53    15_542.21    15_546.74       1.0000          1.0000         9.16
IVF-BF16-nl273-np13 (query)                              388.77        87.22       475.99       0.9348          1.0021         9.19
IVF-BF16-nl273-np16 (query)                              388.77       107.82       496.59       0.9348          1.0021         9.19
IVF-BF16-nl273-np23 (query)                              388.77       147.00       535.77       0.9348          1.0021         9.19
IVF-BF16-nl273 (self)                                    388.77     1_452.28     1_841.05       0.9174          1.0042         9.19
IVF-BF16-nl387-np19 (query)                              744.09        90.15       834.24       0.9348          1.0021         9.21
IVF-BF16-nl387-np27 (query)                              744.09       122.75       866.83       0.9348          1.0021         9.21
IVF-BF16-nl387 (self)                                    744.09     1_207.01     1_951.09       0.9174          1.0042         9.21
IVF-BF16-nl547-np23 (query)                            1_426.38        80.61     1_506.99       0.9348          1.0021         9.23
IVF-BF16-nl547-np27 (query)                            1_426.38        92.08     1_518.46       0.9348          1.0021         9.23
IVF-BF16-nl547-np33 (query)                            1_426.38       109.81     1_536.19       0.9348          1.0021         9.23
IVF-BF16-nl547 (self)                                  1_426.38     1_122.45     2_548.83       0.9174          1.0042         9.23
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
Exhaustive (query)                                        14.39     5_829.38     5_843.77       1.0000          1.0000        73.24
Exhaustive (self)                                         14.39    58_794.49    58_808.88       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   23.26     4_997.33     5_020.59       0.9714          1.0025        36.62
Exhaustive-BF16 (self)                                    23.26    58_904.52    58_927.78       1.0000          1.0000        36.62
IVF-BF16-nl273-np13 (query)                              491.47       293.05       784.52       0.9712          1.0025        36.76
IVF-BF16-nl273-np16 (query)                              491.47       354.17       845.64       0.9713          1.0025        36.76
IVF-BF16-nl273-np23 (query)                              491.47       504.43       995.90       0.9714          1.0025        36.76
IVF-BF16-nl273 (self)                                    491.47     5_164.33     5_655.80       0.9637          1.0047        36.76
IVF-BF16-nl387-np19 (query)                              909.23       298.90     1_208.13       0.9713          1.0025        36.81
IVF-BF16-nl387-np27 (query)                              909.23       412.42     1_321.64       0.9714          1.0025        36.81
IVF-BF16-nl387 (self)                                    909.23     4_180.43     5_089.66       0.9637          1.0047        36.81
IVF-BF16-nl547-np23 (query)                            1_843.23       272.43     2_115.66       0.9713          1.0025        36.89
IVF-BF16-nl547-np27 (query)                            1_843.23       310.45     2_153.68       0.9714          1.0025        36.89
IVF-BF16-nl547-np33 (query)                            1_843.23       374.69     2_217.92       0.9714          1.0025        36.89
IVF-BF16-nl547 (self)                                  1_843.23     3_766.25     5_609.48       0.9637          1.0047        36.89
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
Exhaustive (query)                                         3.11     1_504.31     1_507.42       1.0000          1.0000        18.31
Exhaustive (self)                                          3.11    15_223.57    15_226.68       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     7.38       721.03       728.41       0.8011             NaN         4.58
Exhaustive-SQ8 (self)                                      7.38     7_964.87     7_972.24       0.8007             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               388.22        49.55       437.77       0.7779             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               388.22        58.15       446.37       0.7813             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               388.22        79.68       467.90       0.7822             NaN         4.61
IVF-SQ8-nl273 (self)                                     388.22       820.16     1_208.38       0.7819             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               737.82        51.80       789.62       0.7853             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               737.82        67.20       805.02       0.7878             NaN         4.63
IVF-SQ8-nl387 (self)                                     737.82       680.90     1_418.72       0.7872             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_458.56        49.42     1_507.98       0.7972             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_458.56        55.55     1_514.11       0.8002             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_458.56        63.35     1_521.91       0.8012             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_458.56       639.72     2_098.27       0.8007             NaN         4.65
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
Exhaustive (query)                                         4.06     1_592.76     1_596.83       1.0000          1.0000        18.88
Exhaustive (self)                                          4.06    15_761.86    15_765.92       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                     7.51       685.71       693.22       0.8501             NaN         5.15
Exhaustive-SQ8 (self)                                      7.51     7_216.83     7_224.34       0.8497             NaN         5.15
IVF-SQ8-nl273-np13 (query)                               372.63        47.98       420.60       0.8423             NaN         5.19
IVF-SQ8-nl273-np16 (query)                               372.63        57.63       430.25       0.8463             NaN         5.19
IVF-SQ8-nl273-np23 (query)                               372.63        76.29       448.92       0.8473             NaN         5.19
IVF-SQ8-nl273 (self)                                     372.63       766.39     1_139.02       0.8467             NaN         5.19
IVF-SQ8-nl387-np19 (query)                               701.67        53.71       755.38       0.8420             NaN         5.20
IVF-SQ8-nl387-np27 (query)                               701.67        73.99       775.66       0.8449             NaN         5.20
IVF-SQ8-nl387 (self)                                     701.67       664.45     1_366.12       0.8446             NaN         5.20
IVF-SQ8-nl547-np23 (query)                             1_374.87        48.25     1_423.12       0.8437             NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_374.87        52.39     1_427.26       0.8467             NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_374.87        60.94     1_435.81       0.8477             NaN         5.22
IVF-SQ8-nl547 (self)                                   1_374.87       604.46     1_979.33       0.8473             NaN         5.22
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
Exhaustive (query)                                         3.31     1_518.88     1_522.19       1.0000          1.0000        18.31
Exhaustive (self)                                          3.31    15_327.55    15_330.86       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     6.97       711.44       718.41       0.6828             NaN         4.58
Exhaustive-SQ8 (self)                                      6.97     8_599.36     8_606.33       0.6835             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               479.13        51.54       530.67       0.6821             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               479.13        60.42       539.55       0.6821             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               479.13        91.25       570.38       0.6820             NaN         4.61
IVF-SQ8-nl273 (self)                                     479.13       860.96     1_340.09       0.6832             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               807.32        67.64       874.96       0.6842             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               807.32        77.07       884.39       0.6842             NaN         4.63
IVF-SQ8-nl387 (self)                                     807.32       723.03     1_530.35       0.6851             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_466.15        50.60     1_516.75       0.6826             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_466.15        59.02     1_525.17       0.6826             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_466.15        68.34     1_534.49       0.6826             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_466.15       687.82     2_153.98       0.6832             NaN         4.65
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
Exhaustive (query)                                         3.39     1_617.77     1_621.16       1.0000          1.0000        18.31
Exhaustive (self)                                          3.39    16_428.10    16_431.49       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                     8.12       771.24       779.37       0.4800             NaN         4.58
Exhaustive-SQ8 (self)                                      8.12     8_704.79     8_712.91       0.4862             NaN         4.58
IVF-SQ8-nl273-np13 (query)                               437.26        55.93       493.18       0.4788             NaN         4.61
IVF-SQ8-nl273-np16 (query)                               437.26        61.39       498.65       0.4787             NaN         4.61
IVF-SQ8-nl273-np23 (query)                               437.26        83.39       520.65       0.4786             NaN         4.61
IVF-SQ8-nl273 (self)                                     437.26       829.48     1_266.74       0.4863             NaN         4.61
IVF-SQ8-nl387-np19 (query)                               742.43        52.23       794.66       0.4790             NaN         4.63
IVF-SQ8-nl387-np27 (query)                               742.43        68.05       810.48       0.4790             NaN         4.63
IVF-SQ8-nl387 (self)                                     742.43       732.73     1_475.16       0.4861             NaN         4.63
IVF-SQ8-nl547-np23 (query)                             1_587.12        55.36     1_642.49       0.4800             NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_587.12        60.79     1_647.91       0.4799             NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_587.12        72.07     1_659.19       0.4799             NaN         4.65
IVF-SQ8-nl547 (self)                                   1_587.12       712.35     2_299.47       0.4865             NaN         4.65
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
Exhaustive (query)                                        16.24     7_138.85     7_155.10       1.0000          1.0000        73.24
Exhaustive (self)                                         16.24    63_384.18    63_400.42       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    41.41     1_629.03     1_670.44       0.8081             NaN        18.31
Exhaustive-SQ8 (self)                                     41.41    17_021.27    17_062.68       0.8095             NaN        18.31
IVF-SQ8-nl273-np13 (query)                               503.72       102.55       606.27       0.8062             NaN        18.45
IVF-SQ8-nl273-np16 (query)                               503.72       129.74       633.46       0.8062             NaN        18.45
IVF-SQ8-nl273-np23 (query)                               503.72       165.98       669.69       0.8062             NaN        18.45
IVF-SQ8-nl273 (self)                                     503.72     1_584.41     2_088.13       0.8082             NaN        18.45
IVF-SQ8-nl387-np19 (query)                               945.37       108.72     1_054.09       0.8062             NaN        18.50
IVF-SQ8-nl387-np27 (query)                               945.37       141.41     1_086.78       0.8062             NaN        18.50
IVF-SQ8-nl387 (self)                                     945.37     1_334.88     2_280.25       0.8086             NaN        18.50
IVF-SQ8-nl547-np23 (query)                             1_867.90       103.65     1_971.55       0.8078             NaN        18.58
IVF-SQ8-nl547-np27 (query)                             1_867.90       116.11     1_984.01       0.8078             NaN        18.58
IVF-SQ8-nl547-np33 (query)                             1_867.90       135.32     2_003.22       0.8078             NaN        18.58
IVF-SQ8-nl547 (self)                                   1_867.90     1_255.53     3_123.43       0.8097             NaN        18.58
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
Exhaustive (query)                                         4.44     1_733.78     1_738.22       1.0000          1.0000        24.41
Exhaustive (self)                                          4.44     5_876.99     5_881.42       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                628.53       646.59     1_275.12       0.4099             NaN         0.89
Exhaustive-PQ-m16 (self)                                 628.53     2_145.21     2_773.73       0.3229             NaN         0.89
Exhaustive-PQ-m32 (query)                                996.78     1_559.06     2_555.84       0.5325             NaN         1.65
Exhaustive-PQ-m32 (self)                                 996.78     4_834.64     5_831.42       0.4486             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_946.82     3_924.58     5_871.40       0.6898             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_946.82    13_122.33    15_069.15       0.6265             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_235.13       223.88     1_459.01       0.5971             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_235.13       349.83     1_584.96       0.6007             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_235.13       479.67     1_714.80       0.6012             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_235.13     1_659.85     2_894.98       0.5140             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_614.79       397.01     2_011.81       0.7400             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_614.79       630.24     2_245.03       0.7456             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_614.79       862.66     2_477.45       0.7464             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_614.79     2_852.97     4_467.76       0.6864             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_563.89       923.07     3_486.96       0.8666             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_563.89     1_480.78     4_044.67       0.8748             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_563.89     2_025.74     4_589.63       0.8757             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_563.89     6_818.50     9_382.39       0.8460             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            976.16       300.79     1_276.95       0.6043             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            976.16       379.19     1_355.35       0.6050             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            976.16       569.99     1_546.16       0.6052             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  976.16     1_872.07     2_848.23       0.5191             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_344.46       539.81     1_884.26       0.7474             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_344.46       668.02     2_012.47       0.7489             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_344.46       993.50     2_337.96       0.7492             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_344.46     3_291.79     4_636.25       0.6896             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_290.68     1_226.27     3_516.95       0.8729             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_290.68     1_558.22     3_848.90       0.8752             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_290.68     2_326.88     4_617.56       0.8757             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_290.68     7_746.24    10_036.92       0.8476             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_123.30       393.69     1_516.99       0.6064             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_123.30       454.30     1_577.60       0.6068             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_123.30       652.72     1_776.02       0.6071             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_123.30     2_208.07     3_331.36       0.5200             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_494.15       682.41     2_176.55       0.7499             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_494.15       780.37     2_274.52       0.7507             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_494.15     1_125.29     2_619.43       0.7513             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_494.15     3_755.19     5_249.34       0.6914             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_470.35     1_590.11     4_060.46       0.8746             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_470.35     1_805.55     4_275.90       0.8760             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_470.35     2_654.82     5_125.18       0.8770             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_470.35     8_789.11    11_259.47       0.8494             NaN         3.34
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
Exhaustive (query)                                        10.45     4_257.47     4_267.92       1.0000          1.0000        48.83
Exhaustive (self)                                         10.45    14_416.66    14_427.11       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_203.25       671.72     1_874.97       0.2875             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_203.25     2_226.36     3_429.61       0.2108             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_268.68     1_537.26     2_805.94       0.4059             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_268.68     5_054.46     6_323.15       0.3192             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_108.71     4_034.26     6_142.97       0.5236             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_108.71    13_477.49    15_586.20       0.4429             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_434.03       267.88     2_701.91       0.4316             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_434.03       447.21     2_881.23       0.4364             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_434.03       636.69     3_070.72       0.4373             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_434.03     2_076.05     4_510.08       0.3421             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_568.98       429.94     2_998.92       0.5865             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_568.98       722.34     3_291.32       0.5951             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_568.98     1_003.09     3_572.07       0.5970             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_568.98     3_321.73     5_890.71       0.5151             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_362.76       787.20     4_149.96       0.7201             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_362.76     1_305.18     4_667.94       0.7333             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_362.76     1_805.66     5_168.41       0.7363             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_362.76     6_053.31     9_416.07       0.6797             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_743.00       412.13     2_155.13       0.4376             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_743.00       499.20     2_242.19       0.4385             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_743.00       752.95     2_495.95       0.4390             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_743.00     2_463.77     4_206.76       0.3448             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_858.99       620.84     2_479.83       0.5980             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_858.99       785.98     2_644.96       0.6000             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_858.99     1_164.37     3_023.36       0.6009             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_858.99     3_894.61     5_753.59       0.5176             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_695.31     1_113.92     3_809.23       0.7338             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_695.31     1_404.60     4_099.92       0.7369             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_695.31     2_071.73     4_767.04       0.7384             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_695.31     6_885.97     9_581.28       0.6823             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_932.54       517.15     2_449.69       0.4419             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_932.54       582.69     2_515.23       0.4425             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_932.54       841.87     2_774.41       0.4430             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_932.54     2_822.47     4_755.00       0.3475             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_112.29       800.13     2_912.42       0.5995             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_112.29       903.34     3_015.62       0.6008             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_112.29     1_333.17     3_445.46       0.6020             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_112.29     4_374.21     6_486.50       0.5196             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_944.95     1_416.84     4_361.79       0.7357             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_944.95     1_595.75     4_540.70       0.7377             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_944.95     2_351.45     5_296.40       0.7396             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_944.95     7_787.67    10_732.63       0.6836             NaN         3.61
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
Exhaustive (query)                                        20.02     9_997.18    10_017.20       1.0000          1.0000        97.66
Exhaustive (self)                                         20.02    33_418.41    33_438.43       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_316.76       678.22     1_994.98       0.2182             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_316.76     2_257.31     3_574.08       0.1623             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_249.67     1_524.08     3_773.75       0.2969             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_249.67     5_072.93     7_322.60       0.2221             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_563.96     4_030.38     6_594.34       0.4160             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_563.96    13_662.10    16_226.06       0.3341             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           4_036.86       381.79     4_418.65       0.3110             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          4_036.86       634.41     4_671.27       0.3141             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          4_036.86       879.56     4_916.42       0.3147             NaN         1.57
IVF-PQ-nl158-m16 (self)                                4_036.86     2_984.66     7_021.52       0.2306             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           5_037.79       562.33     5_600.12       0.4430             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          5_037.79       920.80     5_958.59       0.4491             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          5_037.79     1_276.48     6_314.27       0.4506             NaN         2.34
IVF-PQ-nl158-m32 (self)                                5_037.79     4_236.20     9_273.99       0.3611             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_305.79       908.09     6_213.89       0.5978             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_305.79     1_487.94     6_793.73       0.6085             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_305.79     2_159.15     7_464.95       0.6114             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_305.79     6_784.29    12_090.08       0.5369             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_339.21       548.95     2_888.16       0.3161             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_339.21       690.94     3_030.14       0.3165             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_339.21     1_013.26     3_352.47       0.3165             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_339.21     3_377.75     5_716.96       0.2310             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_251.93       777.47     4_029.40       0.4524             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_251.93       975.05     4_226.98       0.4534             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_251.93     1_452.04     4_703.97       0.4536             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_251.93     4_788.69     8_040.62       0.3641             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_598.06     1_246.06     4_844.13       0.6095             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_598.06     1_565.61     5_163.68       0.6119             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_598.06     2_334.94     5_933.01       0.6123             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_598.06     7_753.62    11_351.69       0.5375             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_738.62       714.84     3_453.46       0.3181             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_738.62       806.95     3_545.57       0.3184             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_738.62     1_152.32     3_890.94       0.3187             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_738.62     3_878.54     6_617.16       0.2325             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_720.07     1_027.27     4_747.34       0.4545             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_720.07     1_161.06     4_881.13       0.4553             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_720.07     1_664.77     5_384.83       0.4557             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_720.07     5_548.42     9_268.49       0.3661             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_974.10     1_643.14     5_617.24       0.6122             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_974.10     1_835.16     5_809.26       0.6139             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_974.10     2_655.82     6_629.92       0.6150             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_974.10     8_860.63    12_834.73       0.5393             NaN         4.17
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
Exhaustive (query)                                         4.70     1_781.65     1_786.35       1.0000          1.0000        24.41
Exhaustive (self)                                          4.70     5_972.00     5_976.70       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                627.34       651.26     1_278.59       0.4611             NaN         0.89
Exhaustive-PQ-m16 (self)                                 627.34     2_145.38     2_772.72       0.3589             NaN         0.89
Exhaustive-PQ-m32 (query)                                992.62     1_454.74     2_447.36       0.6024             NaN         1.65
Exhaustive-PQ-m32 (self)                                 992.62     4_838.74     5_831.36       0.5108             NaN         1.65
Exhaustive-PQ-m64 (query)                              1_935.96     3_929.91     5_865.87       0.7713             NaN         3.18
Exhaustive-PQ-m64 (self)                               1_935.96    13_119.88    15_055.84       0.7164             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_248.72       201.57     1_450.29       0.7304             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_248.72       339.44     1_588.17       0.7308             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_248.72       502.57     1_751.29       0.7308             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_248.72     1_561.99     2_810.71       0.6444             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_610.05       355.05     1_965.10       0.8598             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_610.05       599.78     2_209.83       0.8606             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_610.05       849.40     2_459.45       0.8606             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_610.05     2_791.20     4_401.25       0.8171             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_545.36       825.60     3_370.96       0.9531             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_545.36     1_401.46     3_946.83       0.9543             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_545.36     1_978.35     4_523.72       0.9543             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_545.36     6_589.84     9_135.20       0.9398             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            924.18       305.48     1_229.67       0.7321             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            924.18       394.03     1_318.21       0.7324             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            924.18       567.89     1_492.07       0.7324             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  924.18     1_879.84     2_804.02       0.6473             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_301.22       525.90     1_827.12       0.8624             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_301.22       663.91     1_965.13       0.8629             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_301.22       992.72     2_293.94       0.8630             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_301.22     3_273.35     4_574.57       0.8191             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_249.27     1_217.91     3_467.18       0.9537             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_249.27     1_542.34     3_791.61       0.9545             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_249.27     2_286.27     4_535.54       0.9546             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_249.27     7_619.42     9_868.69       0.9407             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_065.72       397.98     1_463.70       0.7354             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_065.72       446.58     1_512.30       0.7354             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_065.72       643.28     1_709.00       0.7355             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_065.72     2_264.47     3_330.19       0.6462             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_592.30       705.27     2_297.57       0.8644             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_592.30       789.76     2_382.06       0.8646             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_592.30     1_154.00     2_746.30       0.8646             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_592.30     3_818.17     5_410.47       0.8214             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_530.32     1_633.42     4_163.73       0.9558             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_530.32     1_841.74     4_372.05       0.9562             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_530.32     2_685.70     5_216.02       0.9564             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_530.32     8_965.49    11_495.80       0.9416             NaN         3.34
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
Exhaustive (query)                                        10.11     4_309.21     4_319.33       1.0000          1.0000        48.83
Exhaustive (self)                                         10.11    14_342.04    14_352.15       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_145.24       666.37     1_811.61       0.3249             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_145.24     2_228.22     3_373.47       0.2378             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_267.43     1_512.19     2_779.62       0.4336             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_267.43     5_384.44     6_651.87       0.3345             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_125.24     4_085.35     6_210.59       0.5527             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_125.24    13_480.18    15_605.42       0.4781             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_445.63       264.37     2_710.00       0.5294             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_445.63       441.83     2_887.47       0.5303             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_445.63       617.91     3_063.54       0.5303             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_445.63     2_044.48     4_490.12       0.4148             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_586.82       413.53     3_000.35       0.6733             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_586.82       707.04     3_293.86       0.6755             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_586.82       992.64     3_579.46       0.6755             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_586.82     3_341.86     5_928.68       0.6057             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_361.29       749.37     4_110.66       0.8414             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_361.29     1_285.99     4_647.28       0.8457             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_361.29     1_809.21     5_170.50       0.8457             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_361.29     6_000.15     9_361.44       0.8109             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_680.57       393.90     2_074.48       0.5299             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_680.57       500.36     2_180.93       0.5300             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_680.57       741.16     2_421.74       0.5300             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_680.57     2_457.85     4_138.43       0.4077             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_811.52       612.87     2_424.39       0.6775             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_811.52       779.30     2_590.83       0.6776             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_811.52     1_156.31     2_967.83       0.6776             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_811.52     3_874.76     5_686.28       0.6039             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_603.79     1_088.54     3_692.34       0.8475             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_603.79     1_382.36     3_986.15       0.8480             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_603.79     2_047.98     4_651.77       0.8480             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_603.79     6_850.20     9_453.99       0.8127             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_826.90       517.55     2_344.45       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_826.90       589.48     2_416.39       0.5307             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_826.90       842.88     2_669.78       0.5307             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_826.90     2_800.00     4_626.90       0.3996             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_044.02       796.45     2_840.47       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_044.02       906.70     2_950.72       0.6790             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_044.02     1_304.88     3_348.90       0.6790             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_044.02     4_355.16     6_399.18       0.6011             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_797.93     1_404.59     4_202.52       0.8491             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_797.93     1_585.96     4_383.89       0.8493             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_797.93     2_322.38     5_120.31       0.8493             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_797.93     7_712.23    10_510.16       0.8132             NaN         3.61
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
Exhaustive (query)                                        20.58     9_927.10     9_947.68       1.0000          1.0000        97.66
Exhaustive (self)                                         20.58    33_259.77    33_280.35       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_519.21       675.88     2_195.09       0.2283             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_519.21     2_272.62     3_791.82       0.1668             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_285.19     1_513.79     3_798.98       0.3094             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_285.19     5_066.95     7_352.14       0.2205             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_565.49     4_048.15     6_613.64       0.4023             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_565.49    13_488.04    16_053.53       0.3046             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           4_009.09       361.30     4_370.39       0.3654             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          4_009.09       604.39     4_613.47       0.3656             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          4_009.09       875.47     4_884.56       0.3656             NaN         1.57
IVF-PQ-nl158-m16 (self)                                4_009.09     2_846.97     6_856.06       0.2381             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_821.74       529.50     5_351.24       0.4716             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_821.74       891.17     5_712.91       0.4725             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_821.74     1_246.59     6_068.33       0.4725             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_821.74     4_155.40     8_977.14       0.3566             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_229.47       849.29     6_078.76       0.6182             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_229.47     1_436.45     6_665.93       0.6202             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_229.47     2_014.62     7_244.10       0.6202             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_229.47     6_714.23    11_943.71       0.5652             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_293.47       552.01     2_845.49       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_293.47       691.36     2_984.83       0.3634             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_293.47     1_030.32     3_323.79       0.3634             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_293.47     3_388.51     5_681.98       0.2326             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_158.60       791.21     3_949.82       0.4690             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_158.60       984.44     4_143.05       0.4692             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_158.60     1_464.39     4_622.99       0.4692             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_158.60     4_869.58     8_028.18       0.3466             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_500.80     1_268.10     4_768.90       0.6175             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_500.80     1_599.30     5_100.10       0.6181             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_500.80     2_371.88     5_872.68       0.6181             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_500.80     7_896.44    11_397.24       0.5601             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_599.32       722.58     3_321.90       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_599.32       804.79     3_404.12       0.3590             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_599.32     1_164.59     3_763.92       0.3590             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_599.32     3_864.62     6_463.94       0.2239             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_714.07     1_035.35     4_749.42       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_714.07     1_173.71     4_887.78       0.4674             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_714.07     1_705.27     5_419.34       0.4674             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_714.07     5_610.08     9_324.15       0.3349             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_960.87     1_629.00     5_589.87       0.6202             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_960.87     1_971.08     5_931.95       0.6204             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_960.87     2_693.46     6_654.33       0.6204             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_960.87     9_104.33    13_065.20       0.5565             NaN         4.17
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
Exhaustive (query)                                         3.25     1_755.07     1_758.32       1.0000          1.0000        24.41
Exhaustive (self)                                          3.25     5_974.28     5_977.53       1.0000          1.0000        24.41
Exhaustive-PQ-m16 (query)                                646.15       659.76     1_305.92       0.1806             NaN         0.89
Exhaustive-PQ-m16 (self)                                 646.15     2_212.38     2_858.53       0.3284             NaN         0.89
Exhaustive-PQ-m32 (query)                              1_055.43     1_551.48     2_606.91       0.2764             NaN         1.65
Exhaustive-PQ-m32 (self)                               1_055.43     4_965.23     6_020.66       0.4254             NaN         1.65
Exhaustive-PQ-m64 (query)                              2_054.74     4_032.41     6_087.14       0.5303             NaN         3.18
Exhaustive-PQ-m64 (self)                               2_054.74    13_526.80    15_581.54       0.6555             NaN         3.18
IVF-PQ-nl158-m16-np7 (query)                           1_263.70       258.35     1_522.05       0.4381             NaN         0.97
IVF-PQ-nl158-m16-np12 (query)                          1_263.70       449.31     1_713.01       0.4382             NaN         0.97
IVF-PQ-nl158-m16-np17 (query)                          1_263.70       620.30     1_883.99       0.4382             NaN         0.97
IVF-PQ-nl158-m16 (self)                                1_263.70     2_071.58     3_335.28       0.5805             NaN         0.97
IVF-PQ-nl158-m32-np7 (query)                           1_686.39       477.44     2_163.83       0.6345             NaN         1.73
IVF-PQ-nl158-m32-np12 (query)                          1_686.39       839.29     2_525.68       0.6347             NaN         1.73
IVF-PQ-nl158-m32-np17 (query)                          1_686.39     1_188.56     2_874.96       0.6347             NaN         1.73
IVF-PQ-nl158-m32 (self)                                1_686.39     3_953.04     5_639.44       0.7460             NaN         1.73
IVF-PQ-nl158-m64-np7 (query)                           2_604.65     1_138.90     3_743.56       0.8675             NaN         3.26
IVF-PQ-nl158-m64-np12 (query)                          2_604.65     1_981.65     4_586.30       0.8679             NaN         3.26
IVF-PQ-nl158-m64-np17 (query)                          2_604.65     2_744.35     5_349.00       0.8679             NaN         3.26
IVF-PQ-nl158-m64 (self)                                2_604.65     9_613.35    12_218.00       0.9122             NaN         3.26
IVF-PQ-nl223-m16-np11 (query)                            946.51       310.21     1_256.72       0.4843             NaN         1.00
IVF-PQ-nl223-m16-np14 (query)                            946.51       397.39     1_343.90       0.4843             NaN         1.00
IVF-PQ-nl223-m16-np21 (query)                            946.51       604.11     1_550.61       0.4843             NaN         1.00
IVF-PQ-nl223-m16 (self)                                  946.51     1_957.68     2_904.18       0.6189             NaN         1.00
IVF-PQ-nl223-m32-np11 (query)                          1_332.37       556.30     1_888.67       0.6783             NaN         1.76
IVF-PQ-nl223-m32-np14 (query)                          1_332.37       694.58     2_026.95       0.6783             NaN         1.76
IVF-PQ-nl223-m32-np21 (query)                          1_332.37     1_031.58     2_363.95       0.6783             NaN         1.76
IVF-PQ-nl223-m32 (self)                                1_332.37     3_417.01     4_749.38       0.7762             NaN         1.76
IVF-PQ-nl223-m64-np11 (query)                          2_542.11     1_415.85     3_957.96       0.8864             NaN         3.29
IVF-PQ-nl223-m64-np14 (query)                          2_542.11     1_738.67     4_280.77       0.8864             NaN         3.29
IVF-PQ-nl223-m64-np21 (query)                          2_542.11     2_584.80     5_126.90       0.8865             NaN         3.29
IVF-PQ-nl223-m64 (self)                                2_542.11     8_618.68    11_160.79       0.9220             NaN         3.29
IVF-PQ-nl316-m16-np15 (query)                          1_182.05       412.09     1_594.14       0.4948             NaN         1.05
IVF-PQ-nl316-m16-np17 (query)                          1_182.05       481.61     1_663.66       0.4948             NaN         1.05
IVF-PQ-nl316-m16-np25 (query)                          1_182.05       698.89     1_880.94       0.4948             NaN         1.05
IVF-PQ-nl316-m16 (self)                                1_182.05     2_222.49     3_404.54       0.6215             NaN         1.05
IVF-PQ-nl316-m32-np15 (query)                          1_586.69       708.40     2_295.09       0.6885             NaN         1.81
IVF-PQ-nl316-m32-np17 (query)                          1_586.69       789.73     2_376.42       0.6885             NaN         1.81
IVF-PQ-nl316-m32-np25 (query)                          1_586.69     1_184.10     2_770.79       0.6885             NaN         1.81
IVF-PQ-nl316-m32 (self)                                1_586.69     3_868.45     5_455.14       0.7797             NaN         1.81
IVF-PQ-nl316-m64-np15 (query)                          2_508.44     1_634.00     4_142.43       0.8890             NaN         3.34
IVF-PQ-nl316-m64-np17 (query)                          2_508.44     1_853.33     4_361.77       0.8890             NaN         3.34
IVF-PQ-nl316-m64-np25 (query)                          2_508.44     2_730.31     5_238.74       0.8890             NaN         3.34
IVF-PQ-nl316-m64 (self)                                2_508.44     9_092.31    11_600.75       0.9237             NaN         3.34
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
Exhaustive (query)                                         6.68     4_326.10     4_332.78       1.0000          1.0000        48.83
Exhaustive (self)                                          6.68    14_455.87    14_462.55       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                              1_347.31       666.08     2_013.39       0.1170             NaN         1.01
Exhaustive-PQ-m16 (self)                               1_347.31     2_210.53     3_557.84       0.3086             NaN         1.01
Exhaustive-PQ-m32 (query)                              1_263.27     1_505.54     2_768.81       0.1723             NaN         1.78
Exhaustive-PQ-m32 (self)                               1_263.27     4_991.19     6_254.46       0.4012             NaN         1.78
Exhaustive-PQ-m64 (query)                              2_224.63     4_025.47     6_250.10       0.2817             NaN         3.30
Exhaustive-PQ-m64 (self)                               2_224.63    13_458.23    15_682.85       0.5337             NaN         3.30
IVF-PQ-nl158-m16-np7 (query)                           2_354.72       294.86     2_649.57       0.3409             NaN         1.17
IVF-PQ-nl158-m16-np12 (query)                          2_354.72       485.26     2_839.97       0.3411             NaN         1.17
IVF-PQ-nl158-m16-np17 (query)                          2_354.72       688.23     3_042.94       0.3411             NaN         1.17
IVF-PQ-nl158-m16 (self)                                2_354.72     2_287.96     4_642.68       0.5928             NaN         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_498.33       472.67     2_970.99       0.4593             NaN         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_498.33       811.05     3_309.38       0.4596             NaN         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_498.33     1_136.19     3_634.51       0.4597             NaN         1.93
IVF-PQ-nl158-m32 (self)                                2_498.33     3_821.58     6_319.91       0.6963             NaN         1.93
IVF-PQ-nl158-m64-np7 (query)                           3_292.17       902.71     4_194.88       0.6615             NaN         3.46
IVF-PQ-nl158-m64-np12 (query)                          3_292.17     1_552.82     4_845.00       0.6621             NaN         3.46
IVF-PQ-nl158-m64-np17 (query)                          3_292.17     2_201.36     5_493.54       0.6622             NaN         3.46
IVF-PQ-nl158-m64 (self)                                3_292.17     7_336.45    10_628.63       0.8315             NaN         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_608.36       413.07     2_021.44       0.3599             NaN         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_608.36       497.14     2_105.50       0.3600             NaN         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_608.36       765.81     2_374.17       0.3600             NaN         1.23
IVF-PQ-nl223-m16 (self)                                1_608.36     2_435.53     4_043.89       0.6039             NaN         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_755.31       606.66     2_361.97       0.4802             NaN         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_755.31       770.72     2_526.03       0.4803             NaN         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_755.31     1_150.28     2_905.59       0.4803             NaN         2.00
IVF-PQ-nl223-m32 (self)                                1_755.31     3_848.75     5_604.06       0.7018             NaN         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_646.31     1_105.27     3_751.58       0.6794             NaN         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_646.31     1_375.52     4_021.83       0.6797             NaN         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_646.31     2_073.60     4_719.92       0.6798             NaN         3.52
IVF-PQ-nl223-m64 (self)                                2_646.31     6_838.43     9_484.74       0.8364             NaN         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_832.15       525.40     2_357.56       0.3717             NaN         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_832.15       585.60     2_417.75       0.3718             NaN         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_832.15       845.16     2_677.32       0.3718             NaN         1.32
IVF-PQ-nl316-m16 (self)                                1_832.15     2_836.72     4_668.87       0.6044             NaN         1.32
IVF-PQ-nl316-m32-np15 (query)                          2_022.97       826.93     2_849.89       0.4907             NaN         2.09
IVF-PQ-nl316-m32-np17 (query)                          2_022.97       932.27     2_955.24       0.4907             NaN         2.09
IVF-PQ-nl316-m32-np25 (query)                          2_022.97     1_312.39     3_335.35       0.4907             NaN         2.09
IVF-PQ-nl316-m32 (self)                                2_022.97     4_353.39     6_376.35       0.7010             NaN         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_796.65     1_417.95     4_214.60       0.6884             NaN         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_796.65     1_585.64     4_382.29       0.6884             NaN         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_796.65     2_345.41     5_142.06       0.6886             NaN         3.61
IVF-PQ-nl316-m64 (self)                                2_796.65     7_731.44    10_528.09       0.8365             NaN         3.61
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
Exhaustive (query)                                        20.26     9_886.59     9_906.85       1.0000          1.0000        97.66
Exhaustive (self)                                         20.26    33_932.72    33_952.98       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_449.45       749.00     2_198.44       0.0871             NaN         1.26
Exhaustive-PQ-m16 (self)                               1_449.45     2_438.62     3_888.07       0.2928             NaN         1.26
Exhaustive-PQ-m32 (query)                              2_088.51     1_575.74     3_664.25       0.1140             NaN         2.03
Exhaustive-PQ-m32 (self)                               2_088.51     5_131.90     7_220.42       0.3714             NaN         2.03
Exhaustive-PQ-m64 (query)                              2_603.88     4_035.93     6_639.81       0.1674             NaN         3.55
Exhaustive-PQ-m64 (self)                               2_603.88    13_522.74    16_126.62       0.4781             NaN         3.55
IVF-PQ-nl158-m16-np7 (query)                           3_904.68       382.93     4_287.61       0.2674             NaN         1.57
IVF-PQ-nl158-m16-np12 (query)                          3_904.68       630.89     4_535.57       0.2676             NaN         1.57
IVF-PQ-nl158-m16-np17 (query)                          3_904.68       893.98     4_798.66       0.2676             NaN         1.57
IVF-PQ-nl158-m16 (self)                                3_904.68     2_996.45     6_901.13       0.5650             NaN         1.57
IVF-PQ-nl158-m32-np7 (query)                           4_895.64       572.91     5_468.55       0.3288             NaN         2.34
IVF-PQ-nl158-m32-np12 (query)                          4_895.64       965.89     5_861.52       0.3291             NaN         2.34
IVF-PQ-nl158-m32-np17 (query)                          4_895.64     1_357.75     6_253.39       0.3291             NaN         2.34
IVF-PQ-nl158-m32 (self)                                4_895.64     4_560.93     9_456.56       0.6477             NaN         2.34
IVF-PQ-nl158-m64-np7 (query)                           5_296.22       958.41     6_254.63       0.4497             NaN         3.86
IVF-PQ-nl158-m64-np12 (query)                          5_296.22     1_651.64     6_947.86       0.4503             NaN         3.86
IVF-PQ-nl158-m64-np17 (query)                          5_296.22     2_391.37     7_687.59       0.4503             NaN         3.86
IVF-PQ-nl158-m64 (self)                                5_296.22     7_694.17    12_990.39       0.7429             NaN         3.86
IVF-PQ-nl223-m16-np11 (query)                          2_076.35       546.98     2_623.33       0.2867             NaN         1.70
IVF-PQ-nl223-m16-np14 (query)                          2_076.35       695.48     2_771.83       0.2867             NaN         1.70
IVF-PQ-nl223-m16-np21 (query)                          2_076.35     1_025.29     3_101.64       0.2866             NaN         1.70
IVF-PQ-nl223-m16 (self)                                2_076.35     3_409.47     5_485.83       0.5824             NaN         1.70
IVF-PQ-nl223-m32-np11 (query)                          3_061.09       777.44     3_838.54       0.3520             NaN         2.46
IVF-PQ-nl223-m32-np14 (query)                          3_061.09       985.39     4_046.49       0.3520             NaN         2.46
IVF-PQ-nl223-m32-np21 (query)                          3_061.09     1_461.09     4_522.18       0.3520             NaN         2.46
IVF-PQ-nl223-m32 (self)                                3_061.09     4_846.88     7_907.98       0.6583             NaN         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_368.56     1_250.70     4_619.26       0.4703             NaN         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_368.56     1_577.04     4_945.61       0.4706             NaN         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_368.56     2_345.46     5_714.03       0.4707             NaN         3.99
IVF-PQ-nl223-m64 (self)                                3_368.56     7_825.09    11_193.65       0.7512             NaN         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_453.13       724.12     3_177.25       0.2956             NaN         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_453.13       821.84     3_274.97       0.2956             NaN         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_453.13     1_183.35     3_636.48       0.2956             NaN         1.88
IVF-PQ-nl316-m16 (self)                                2_453.13     3_876.29     6_329.42       0.5833             NaN         1.88
IVF-PQ-nl316-m32-np15 (query)                          3_372.64     1_039.24     4_411.88       0.3603             NaN         2.65
IVF-PQ-nl316-m32-np17 (query)                          3_372.64     1_171.42     4_544.06       0.3604             NaN         2.65
IVF-PQ-nl316-m32-np25 (query)                          3_372.64     1_796.30     5_168.94       0.3604             NaN         2.65
IVF-PQ-nl316-m32 (self)                                3_372.64     5_605.47     8_978.11       0.6551             NaN         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_687.81     1_658.28     5_346.09       0.4810             NaN         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_687.81     1_846.22     5_534.03       0.4811             NaN         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_687.81     2_700.77     6_388.57       0.4812             NaN         4.17
IVF-PQ-nl316-m64 (self)                                3_687.81     9_512.24    13_200.05       0.7458             NaN         4.17
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
Exhaustive (query)                                         4.57     1_796.36     1_800.93       1.0000          1.0000        24.41
Exhaustive (self)                                          4.57     5_826.55     5_831.12       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              2_689.97       324.59     3_014.56       0.2822             NaN         0.57
Exhaustive-OPQ-m8 (self)                               2_689.97     1_123.96     3_813.93       0.2114             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_109.20       661.10     3_770.31       0.4067             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_109.20     2_278.36     5_387.57       0.3210             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_345.83       144.77     3_490.60       0.4349             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_345.83       232.93     3_578.76       0.4367             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_345.83       317.81     3_663.64       0.4369             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_345.83     1_134.52     4_480.35       0.3373             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_711.03       232.05     3_943.08       0.5962             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_711.03       375.11     4_086.14       0.6001             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_711.03       513.38     4_224.42       0.6005             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_711.03     1_720.31     5_431.34       0.5149             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_041.25       202.15     3_243.40       0.4416             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_041.25       256.23     3_297.47       0.4418             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_041.25       374.71     3_415.95       0.4419             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_041.25     1_304.09     4_345.33       0.3438             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_513.33       309.73     3_823.06       0.6051             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_513.33       411.56     3_924.89       0.6057             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_513.33       571.16     4_084.50       0.6059             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_513.33     2_003.73     5_517.07       0.5193             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_192.69       271.32     3_464.01       0.4467             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_192.69       298.92     3_491.60       0.4469             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_192.69       445.36     3_638.05       0.4469             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_192.69     1_518.68     4_711.37       0.3476             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_655.75       427.21     4_082.96       0.6054             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_655.75       469.89     4_125.64       0.6059             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_655.75       695.39     4_351.14       0.6062             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_655.75     2_327.39     5_983.13       0.5204             NaN         1.11
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
Exhaustive (query)                                        10.00     4_318.27     4_328.26       1.0000          1.0000        48.83
Exhaustive (self)                                         10.00    14_641.57    14_651.57       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             5_553.63       676.57     6_230.21       0.2867             NaN         1.26
Exhaustive-OPQ-m16 (self)                              5_553.63     2_552.30     8_105.93       0.2105             NaN         1.26
Exhaustive-OPQ-m32 (query)                             7_001.11     1_556.55     8_557.67       0.4040             NaN         2.03
Exhaustive-OPQ-m32 (self)                              7_001.11     5_279.51    12_280.62       0.3190             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_523.69     4_030.99    14_554.68       0.5310             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_523.69    13_795.48    24_319.17       0.4550             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          6_965.00       266.97     7_231.97       0.4316             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         6_965.00       440.79     7_405.79       0.4364             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         6_965.00       616.91     7_581.91       0.4372             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               6_965.00     2_493.78     9_458.78       0.3418             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_844.93       431.00     8_275.93       0.5870             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_844.93       743.72     8_588.65       0.5959             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_844.93       986.96     8_831.89       0.5976             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_844.93     3_646.83    11_491.76       0.5161             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_562.78       785.20    12_347.98       0.7221             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_562.78     1_302.19    12_864.96       0.7354             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_562.78     1_813.79    13_376.56       0.7386             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_562.78     6_367.95    17_930.73       0.6850             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_247.72       407.27     6_654.99       0.4394             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_247.72       518.33     6_766.06       0.4402             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_247.72       755.91     7_003.64       0.4406             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_247.72     2_858.88     9_106.60       0.3454             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_194.71       634.74     7_829.45       0.5968             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_194.71       789.99     7_984.71       0.5987             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_194.71     1_185.63     8_380.34       0.5997             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_194.71     4_296.21    11_490.92       0.5179             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_779.12     1_111.00    11_890.12       0.7377             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_779.12     1_411.23    12_190.35       0.7407             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_779.12     2_119.07    12_898.19       0.7422             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_779.12     7_331.60    18_110.72       0.6880             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_404.27       537.65     6_941.92       0.4415             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_404.27       603.18     7_007.45       0.4421             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_404.27       872.79     7_277.06       0.4427             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_404.27     3_255.18     9_659.45       0.3486             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_399.20       819.17     8_218.37       0.5990             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_399.20       926.66     8_325.86       0.6001             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_399.20     1_359.19     8_758.39       0.6015             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_399.20     4_848.39    12_247.59       0.5204             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_085.79     1_421.56    12_507.35       0.7378             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_085.79     1_616.45    12_702.24       0.7399             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_085.79     2_365.41    13_451.20       0.7419             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_085.79     8_232.07    19_317.86       0.6884             NaN         3.86
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
Exhaustive (query)                                        20.38     9_914.66     9_935.04       1.0000          1.0000        97.66
Exhaustive (self)                                         20.38    33_298.45    33_318.83       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_971.46       679.20     8_650.66       0.2174             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_971.46     3_717.31    11_688.78       0.1625             NaN         2.26
Exhaustive-OPQ-m32 (query)                            11_561.32     1_538.96    13_100.27       0.2960             NaN         3.03
Exhaustive-OPQ-m32 (self)                             11_561.32     6_461.38    18_022.69       0.2226             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_182.22     4_044.73    17_226.96       0.4149             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_182.22    14_745.17    27_927.39       0.3338             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_769.76     9_405.09    29_174.85       0.5365             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_769.76    32_550.60    52_320.35       0.4666             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         11_185.60       377.67    11_563.27       0.3111             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        11_185.60       639.54    11_825.14       0.3136             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        11_185.60       884.88    12_070.47       0.3143             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              11_185.60     4_405.45    15_591.04       0.2313             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         14_564.74       550.85    15_115.59       0.4428             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        14_564.74       917.14    15_481.88       0.4490             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        14_564.74     1_287.25    15_851.99       0.4505             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              14_564.74     5_683.76    20_248.50       0.3609             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         16_297.64       910.78    17_208.43       0.5974             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        16_297.64     1_495.49    17_793.14       0.6084             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        16_297.64     2_076.55    18_374.19       0.6114             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              16_297.64     8_350.08    24_647.73       0.5373             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_487.91     1_630.19    24_118.09       0.7241             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_487.91     2_661.43    25_149.34       0.7399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_487.91     3_701.60    26_189.50       0.7442             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_487.91    13_814.84    36_302.74       0.6963             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_815.65       550.12     9_365.77       0.3160             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_815.65       700.76     9_516.41       0.3164             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_815.65     1_023.45     9_839.10       0.3164             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_815.65     4_819.32    13_634.97       0.2317             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        12_574.47       782.77    13_357.24       0.4532             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        12_574.47       980.76    13_555.23       0.4543             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        12_574.47     1_442.19    14_016.66       0.4545             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              12_574.47     6_232.14    18_806.61       0.3663             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_197.52     1_246.36    15_443.88       0.6102             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_197.52     1_581.64    15_779.16       0.6126             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_197.52     2_351.89    16_549.41       0.6131             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_197.52     9_274.33    23_471.86       0.5387             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_706.69     2_200.97    22_907.66       0.7440             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_706.69     2_840.09    23_546.79       0.7475             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_706.69     4_199.85    24_906.54       0.7482             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_706.69    15_525.98    36_232.67       0.7007             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_678.23       731.84    10_410.07       0.3181             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_678.23       830.17    10_508.40       0.3183             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_678.23     1_207.64    10_885.87       0.3185             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_678.23     5_408.34    15_086.57       0.2335             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        13_484.08     1_035.88    14_519.96       0.4546             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        13_484.08     1_177.87    14_661.95       0.4555             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        13_484.08     1_694.48    15_178.56       0.4561             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              13_484.08     7_025.62    20_509.70       0.3671             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        15_039.30     1_637.10    16_676.40       0.6113             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        15_039.30     1_871.15    16_910.46       0.6128             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        15_039.30     2_730.12    17_769.43       0.6139             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              15_039.30    10_549.02    25_588.32       0.5401             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       21_941.01     2_859.64    24_800.65       0.7452             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       21_941.01     3_241.59    25_182.61       0.7476             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       21_941.01     4_757.51    26_698.52       0.7490             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             21_941.01    17_338.77    39_279.78       0.7015             NaN         8.23
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
Exhaustive (query)                                         4.62     1_774.18     1_778.81       1.0000          1.0000        24.41
Exhaustive (self)                                          4.62     6_179.42     6_184.05       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              2_691.15       328.50     3_019.65       0.3320             NaN         0.57
Exhaustive-OPQ-m8 (self)                               2_691.15     1_106.96     3_798.11       0.2247             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_132.68       659.16     3_791.84       0.4615             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_132.68     2_265.76     5_398.44       0.3241             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_378.64       137.00     3_515.64       0.6717             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_378.64       226.11     3_604.75       0.6719             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_378.64       317.02     3_695.66       0.6719             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_378.64     1_113.18     4_491.82       0.5861             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_722.91       215.92     3_938.83       0.7837             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_722.91       391.85     4_114.76       0.7842             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_722.91       536.58     4_259.50       0.7842             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_722.91     1_746.60     5_469.51       0.7155             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_008.43       207.65     3_216.08       0.6744             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_008.43       255.54     3_263.97       0.6745             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_008.43       380.31     3_388.74       0.6746             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_008.43     1_341.71     4_350.14       0.5896             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_557.70       316.86     3_874.56       0.7859             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_557.70       417.50     3_975.20       0.7861             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_557.70       624.07     4_181.77       0.7862             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_557.70     2_006.52     5_564.21       0.7199             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_172.98       270.75     3_443.73       0.6777             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_172.98       336.53     3_509.51       0.6777             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_172.98       444.24     3_617.22       0.6778             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_172.98     1_544.51     4_717.48       0.5943             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_797.10       410.01     4_207.12       0.7854             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_797.10       484.92     4_282.02       0.7855             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_797.10       688.36     4_485.46       0.7855             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_797.10     2_353.23     6_150.33       0.7212             NaN         1.11
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
Exhaustive (query)                                        10.08     4_324.43     4_334.51       1.0000          1.0000        48.83
Exhaustive (self)                                         10.08    14_292.48    14_302.55       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             5_522.95       663.85     6_186.81       0.3278             NaN         1.26
Exhaustive-OPQ-m16 (self)                              5_522.95     2_546.02     8_068.98       0.2142             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_544.39     1_518.11     8_062.50       0.4531             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_544.39     5_313.71    11_858.10       0.3215             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_193.95     4_015.20    14_209.15       0.5830             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_193.95    13_796.98    23_990.93       0.4945             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          6_869.61       266.50     7_136.11       0.6369             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         6_869.61       440.58     7_310.19       0.6385             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         6_869.61       619.09     7_488.70       0.6385             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               6_869.61     2_394.49     9_264.10       0.5499             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_852.55       425.91     8_278.46       0.7638             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_852.55       711.21     8_563.76       0.7665             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_852.55       996.53     8_849.08       0.7665             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_852.55     3_679.18    11_531.73       0.6995             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_392.35       750.08    12_142.43       0.8571             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_392.35     1_288.42    12_680.77       0.8616             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_392.35     1_815.81    13_208.15       0.8616             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_392.35     6_427.20    17_819.55       0.8239             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_243.56       401.03     6_644.59       0.6384             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_243.56       516.39     6_759.95       0.6385             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_243.56       779.58     7_023.14       0.6385             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_243.56     3_005.85     9_249.41       0.5521             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_136.15       625.79     7_761.94       0.7653             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_136.15       794.44     7_930.59       0.7656             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_136.15     1_177.23     8_313.37       0.7656             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_136.15     4_257.45    11_393.60       0.6990             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_768.31     1_102.34    11_870.65       0.8626             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_768.31     1_392.53    12_160.84       0.8631             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_768.31     2_078.49    12_846.80       0.8631             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_768.31     7_309.36    18_077.67       0.8249             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_240.61       527.35     6_767.96       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_240.61       600.12     6_840.73       0.6380             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_240.61       865.97     7_106.58       0.6381             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_240.61     3_241.78     9_482.40       0.5527             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_327.94       813.96     8_141.90       0.7646             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_327.94       932.46     8_260.40       0.7647             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_327.94     1_341.91     8_669.85       0.7647             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_327.94     4_827.60    12_155.53       0.7001             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_042.74     1_417.65    12_460.39       0.8632             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_042.74     1_617.64    12_660.38       0.8633             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_042.74     2_366.71    13_409.45       0.8634             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_042.74     8_244.09    19_286.83       0.8255             NaN         3.86
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
Exhaustive (query)                                        20.32     9_846.71     9_867.03       1.0000          1.0000        97.66
Exhaustive (self)                                         20.32    33_468.84    33_489.16       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             7_909.02       685.11     8_594.13       0.2170             NaN         2.26
Exhaustive-OPQ-m16 (self)                              7_909.02     3_712.55    11_621.57       0.1636             NaN         2.26
Exhaustive-OPQ-m32 (query)                            11_485.50     1_525.52    13_011.02       0.3206             NaN         3.03
Exhaustive-OPQ-m32 (self)                             11_485.50     6_457.19    17_942.69       0.2264             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_208.67     4_029.49    17_238.16       0.4637             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_208.67    14_744.79    27_953.46       0.3377             NaN         4.55
Exhaustive-OPQ-m128 (query)                           19_724.49     9_295.03    29_019.52       0.6010             NaN         7.61
Exhaustive-OPQ-m128 (self)                            19_724.49    32_440.74    52_165.23       0.5072             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_730.49       368.06    11_098.56       0.4556             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_730.49       628.37    11_358.87       0.4564             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_730.49       880.29    11_610.78       0.4564             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_730.49     4_409.77    15_140.26       0.3662             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         14_460.04       529.62    14_989.66       0.6078             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        14_460.04       900.59    15_360.63       0.6096             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        14_460.04     1_273.51    15_733.55       0.6096             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              14_460.04     5_649.58    20_109.62       0.5228             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_984.53       853.63    16_838.16       0.7427             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_984.53     1_448.23    17_432.75       0.7458             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_984.53     2_043.49    18_028.01       0.7458             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_984.53     8_298.75    24_283.28       0.6764             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_361.25     1_513.61    23_874.86       0.8356             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_361.25     2_699.69    25_060.94       0.8399             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_361.25     3_992.05    26_353.30       0.8399             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_361.25    14_308.48    36_669.73       0.8116             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         9_307.59       583.47     9_891.07       0.4573             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         9_307.59       695.55    10_003.15       0.4576             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         9_307.59     1_027.87    10_335.46       0.4576             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               9_307.59     4_840.86    14_148.46       0.3672             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        12_838.31       789.14    13_627.45       0.6053             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        12_838.31       990.10    13_828.41       0.6059             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        12_838.31     1_465.23    14_303.54       0.6059             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              12_838.31     6_293.54    19_131.85       0.5220             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_184.04     1_293.18    15_477.22       0.7434             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_184.04     1_618.83    15_802.87       0.7445             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_184.04     2_422.52    16_606.57       0.7445             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_184.04     9_457.91    23_641.95       0.6776             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_719.98     2_247.65    22_967.63       0.8391             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_719.98     2_857.47    23_577.45       0.8408             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_719.98     4_275.39    24_995.37       0.8408             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_719.98    15_773.98    36_493.96       0.8119             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_188.67       751.27     9_939.94       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_188.67       877.43    10_066.10       0.4569             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_188.67     1_203.54    10_392.21       0.4569             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_188.67     5_399.05    14_587.72       0.3687             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        12_814.90     1_030.57    13_845.47       0.6054             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        12_814.90     1_169.98    13_984.88       0.6055             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        12_814.90     1_678.49    14_493.39       0.6055             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              12_814.90     7_037.59    19_852.49       0.5215             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_742.36     1_645.23    16_387.59       0.7429             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_742.36     1_885.00    16_627.36       0.7432             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_742.36     2_701.34    17_443.70       0.7432             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_742.36    10_499.88    25_242.24       0.6751             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       21_082.66     2_860.90    23_943.56       0.8407             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       21_082.66     3_263.17    24_345.82       0.8413             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       21_082.66     4_818.27    25_900.93       0.8413             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             21_082.66    17_443.29    38_525.95       0.8127             NaN         8.23
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
Exhaustive (query)                                         3.08     1_842.75     1_845.83       1.0000          1.0000        24.41
Exhaustive (self)                                          3.08     6_299.72     6_302.80       1.0000          1.0000        24.41
Exhaustive-OPQ-m8 (query)                              2_733.75       345.58     3_079.33       0.1253             NaN         0.57
Exhaustive-OPQ-m8 (self)                               2_733.75     1_117.59     3_851.34       0.2565             NaN         0.57
Exhaustive-OPQ-m16 (query)                             3_153.69       657.95     3_811.64       0.1716             NaN         0.95
Exhaustive-OPQ-m16 (self)                              3_153.69     2_286.97     5_440.66       0.3354             NaN         0.95
IVF-OPQ-nl158-m8-np7 (query)                           3_228.35       158.52     3_386.87       0.3049             NaN         0.65
IVF-OPQ-nl158-m8-np12 (query)                          3_228.35       276.25     3_504.61       0.3050             NaN         0.65
IVF-OPQ-nl158-m8-np17 (query)                          3_228.35       375.71     3_604.06       0.3050             NaN         0.65
IVF-OPQ-nl158-m8 (self)                                3_228.35     1_331.39     4_559.74       0.5490             NaN         0.65
IVF-OPQ-nl158-m16-np7 (query)                          3_715.54       261.55     3_977.09       0.4028             NaN         1.03
IVF-OPQ-nl158-m16-np12 (query)                         3_715.54       446.06     4_161.60       0.4028             NaN         1.03
IVF-OPQ-nl158-m16-np17 (query)                         3_715.54       639.39     4_354.93       0.4028             NaN         1.03
IVF-OPQ-nl158-m16 (self)                               3_715.54     2_167.77     5_883.31       0.6637             NaN         1.03
IVF-OPQ-nl223-m8-np11 (query)                          3_067.41       207.13     3_274.54       0.3313             NaN         0.68
IVF-OPQ-nl223-m8-np14 (query)                          3_067.41       262.88     3_330.29       0.3313             NaN         0.68
IVF-OPQ-nl223-m8-np21 (query)                          3_067.41       392.81     3_460.22       0.3313             NaN         0.68
IVF-OPQ-nl223-m8 (self)                                3_067.41     1_355.64     4_423.05       0.5994             NaN         0.68
IVF-OPQ-nl223-m16-np11 (query)                         3_439.19       315.23     3_754.42       0.4313             NaN         1.06
IVF-OPQ-nl223-m16-np14 (query)                         3_439.19       408.51     3_847.71       0.4313             NaN         1.06
IVF-OPQ-nl223-m16-np21 (query)                         3_439.19       615.95     4_055.15       0.4313             NaN         1.06
IVF-OPQ-nl223-m16 (self)                               3_439.19     2_045.32     5_484.51       0.7014             NaN         1.06
IVF-OPQ-nl316-m8-np15 (query)                          3_154.54       272.87     3_427.42       0.3408             NaN         0.73
IVF-OPQ-nl316-m8-np17 (query)                          3_154.54       312.06     3_466.61       0.3408             NaN         0.73
IVF-OPQ-nl316-m8-np25 (query)                          3_154.54       450.71     3_605.26       0.3408             NaN         0.73
IVF-OPQ-nl316-m8 (self)                                3_154.54     1_567.06     4_721.61       0.6036             NaN         0.73
IVF-OPQ-nl316-m16-np15 (query)                         3_632.09       429.39     4_061.48       0.4411             NaN         1.11
IVF-OPQ-nl316-m16-np17 (query)                         3_632.09       478.61     4_110.70       0.4411             NaN         1.11
IVF-OPQ-nl316-m16-np25 (query)                         3_632.09       699.50     4_331.59       0.4411             NaN         1.11
IVF-OPQ-nl316-m16 (self)                               3_632.09     2_375.72     6_007.80       0.7037             NaN         1.11
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
Exhaustive (query)                                         6.78     4_210.68     4_217.46       1.0000          1.0000        48.83
Exhaustive (self)                                          6.78    14_494.14    14_500.92       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             5_568.99       675.57     6_244.56       0.1141             NaN         1.26
Exhaustive-OPQ-m16 (self)                              5_568.99     2_535.14     8_104.13       0.3027             NaN         1.26
Exhaustive-OPQ-m32 (query)                             6_579.35     1_540.98     8_120.32       0.1529             NaN         2.03
Exhaustive-OPQ-m32 (self)                              6_579.35     5_280.36    11_859.71       0.4066             NaN         2.03
Exhaustive-OPQ-m64 (query)                            10_196.37     4_023.27    14_219.63       0.2357             NaN         3.55
Exhaustive-OPQ-m64 (self)                             10_196.37    13_936.51    24_132.88       0.5577             NaN         3.55
IVF-OPQ-nl158-m16-np7 (query)                          6_804.38       292.23     7_096.61       0.2959             NaN         1.42
IVF-OPQ-nl158-m16-np12 (query)                         6_804.38       497.06     7_301.44       0.2960             NaN         1.42
IVF-OPQ-nl158-m16-np17 (query)                         6_804.38       703.59     7_507.98       0.2960             NaN         1.42
IVF-OPQ-nl158-m16 (self)                               6_804.38     2_702.49     9_506.88       0.6756             NaN         1.42
IVF-OPQ-nl158-m32-np7 (query)                          7_785.28       474.80     8_260.07       0.4009             NaN         2.18
IVF-OPQ-nl158-m32-np12 (query)                         7_785.28       824.93     8_610.20       0.4011             NaN         2.18
IVF-OPQ-nl158-m32-np17 (query)                         7_785.28     1_165.25     8_950.53       0.4011             NaN         2.18
IVF-OPQ-nl158-m32 (self)                               7_785.28     4_238.51    12_023.78       0.7668             NaN         2.18
IVF-OPQ-nl158-m64-np7 (query)                         11_378.18       901.30    12_279.48       0.6302             NaN         3.71
IVF-OPQ-nl158-m64-np12 (query)                        11_378.18     1_884.75    13_262.93       0.6308             NaN         3.71
IVF-OPQ-nl158-m64-np17 (query)                        11_378.18     2_384.47    13_762.65       0.6309             NaN         3.71
IVF-OPQ-nl158-m64 (self)                              11_378.18     8_076.26    19_454.44       0.8460             NaN         3.71
IVF-OPQ-nl223-m16-np11 (query)                         6_161.28       399.10     6_560.38       0.3048             NaN         1.48
IVF-OPQ-nl223-m16-np14 (query)                         6_161.28       521.84     6_683.12       0.3049             NaN         1.48
IVF-OPQ-nl223-m16-np21 (query)                         6_161.28       798.20     6_959.48       0.3049             NaN         1.48
IVF-OPQ-nl223-m16 (self)                               6_161.28     2_871.58     9_032.86       0.6870             NaN         1.48
IVF-OPQ-nl223-m32-np11 (query)                         7_154.18       657.07     7_811.25       0.4103             NaN         2.25
IVF-OPQ-nl223-m32-np14 (query)                         7_154.18       786.21     7_940.40       0.4104             NaN         2.25
IVF-OPQ-nl223-m32-np21 (query)                         7_154.18     1_165.99     8_320.18       0.4105             NaN         2.25
IVF-OPQ-nl223-m32 (self)                               7_154.18     4_228.81    11_382.99       0.7716             NaN         2.25
IVF-OPQ-nl223-m64-np11 (query)                        10_816.14     1_092.74    11_908.88       0.6406             NaN         3.77
IVF-OPQ-nl223-m64-np14 (query)                        10_816.14     1_390.64    12_206.79       0.6408             NaN         3.77
IVF-OPQ-nl223-m64-np21 (query)                        10_816.14     2_075.30    12_891.44       0.6409             NaN         3.77
IVF-OPQ-nl223-m64 (self)                              10_816.14     7_347.17    18_163.31       0.8502             NaN         3.77
IVF-OPQ-nl316-m16-np15 (query)                         6_312.10       529.39     6_841.49       0.3158             NaN         1.57
IVF-OPQ-nl316-m16-np17 (query)                         6_312.10       592.63     6_904.73       0.3158             NaN         1.57
IVF-OPQ-nl316-m16-np25 (query)                         6_312.10       864.54     7_176.64       0.3158             NaN         1.57
IVF-OPQ-nl316-m16 (self)                               6_312.10     3_247.76     9_559.86       0.6884             NaN         1.57
IVF-OPQ-nl316-m32-np15 (query)                         7_485.36       836.90     8_322.27       0.4223             NaN         2.34
IVF-OPQ-nl316-m32-np17 (query)                         7_485.36       932.42     8_417.78       0.4223             NaN         2.34
IVF-OPQ-nl316-m32-np25 (query)                         7_485.36     1_362.97     8_848.33       0.4223             NaN         2.34
IVF-OPQ-nl316-m32 (self)                               7_485.36     5_172.58    12_657.95       0.7711             NaN         2.34
IVF-OPQ-nl316-m64-np15 (query)                        11_234.94     1_419.36    12_654.30       0.6543             NaN         3.86
IVF-OPQ-nl316-m64-np17 (query)                        11_234.94     1_617.00    12_851.93       0.6544             NaN         3.86
IVF-OPQ-nl316-m64-np25 (query)                        11_234.94     2_391.98    13_626.91       0.6545             NaN         3.86
IVF-OPQ-nl316-m64 (self)                              11_234.94     8_278.52    19_513.45       0.8497             NaN         3.86
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
Exhaustive (query)                                        19.73     9_987.05    10_006.78       1.0000          1.0000        97.66
Exhaustive (self)                                         19.73    33_315.98    33_335.71       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             8_842.72       690.19     9_532.90       0.0880             NaN         2.26
Exhaustive-OPQ-m16 (self)                              8_842.72     3_730.84    12_573.56       0.2699             NaN         2.26
Exhaustive-OPQ-m32 (query)                            12_076.11     1_532.15    13_608.27       0.1079             NaN         3.03
Exhaustive-OPQ-m32 (self)                             12_076.11     6_432.32    18_508.43       0.3567             NaN         3.03
Exhaustive-OPQ-m64 (query)                            13_829.57     4_046.65    17_876.22       0.1462             NaN         4.55
Exhaustive-OPQ-m64 (self)                             13_829.57    14_771.57    28_601.14       0.4750             NaN         4.55
Exhaustive-OPQ-m128 (query)                           20_370.05     9_300.19    29_670.24       0.2377             NaN         7.61
Exhaustive-OPQ-m128 (self)                            20_370.05    32_475.32    52_845.37       0.6470             NaN         7.61
IVF-OPQ-nl158-m16-np7 (query)                         10_781.79       382.71    11_164.50       0.2258             NaN         2.57
IVF-OPQ-nl158-m16-np12 (query)                        10_781.79       657.76    11_439.56       0.2258             NaN         2.57
IVF-OPQ-nl158-m16-np17 (query)                        10_781.79       928.17    11_709.96       0.2258             NaN         2.57
IVF-OPQ-nl158-m16 (self)                              10_781.79     4_499.71    15_281.51       0.6481             NaN         2.57
IVF-OPQ-nl158-m32-np7 (query)                         14_381.84       567.53    14_949.36       0.2801             NaN         3.34
IVF-OPQ-nl158-m32-np12 (query)                        14_381.84       963.84    15_345.68       0.2803             NaN         3.34
IVF-OPQ-nl158-m32-np17 (query)                        14_381.84     1_361.47    15_743.30       0.2803             NaN         3.34
IVF-OPQ-nl158-m32 (self)                              14_381.84     6_017.44    20_399.28       0.7308             NaN         3.34
IVF-OPQ-nl158-m64-np7 (query)                         15_967.18       960.54    16_927.73       0.3797             NaN         4.86
IVF-OPQ-nl158-m64-np12 (query)                        15_967.18     1_632.71    17_599.90       0.3799             NaN         4.86
IVF-OPQ-nl158-m64-np17 (query)                        15_967.18     2_325.40    18_292.58       0.3799             NaN         4.86
IVF-OPQ-nl158-m64 (self)                              15_967.18     9_181.76    25_148.94       0.8105             NaN         4.86
IVF-OPQ-nl158-m128-np7 (query)                        22_503.70     1_734.71    24_238.41       0.6241             NaN         7.92
IVF-OPQ-nl158-m128-np12 (query)                       22_503.70     2_981.15    25_484.86       0.6253             NaN         7.92
IVF-OPQ-nl158-m128-np17 (query)                       22_503.70     4_233.92    26_737.62       0.6254             NaN         7.92
IVF-OPQ-nl158-m128 (self)                             22_503.70    15_652.27    38_155.98       0.8783             NaN         7.92
IVF-OPQ-nl223-m16-np11 (query)                         8_731.23       544.89     9_276.12       0.2401             NaN         2.70
IVF-OPQ-nl223-m16-np14 (query)                         8_731.23       693.39     9_424.62       0.2401             NaN         2.70
IVF-OPQ-nl223-m16-np21 (query)                         8_731.23     1_027.56     9_758.78       0.2401             NaN         2.70
IVF-OPQ-nl223-m16 (self)                               8_731.23     4_827.61    13_558.84       0.6637             NaN         2.70
IVF-OPQ-nl223-m32-np11 (query)                        12_511.92       777.55    13_289.47       0.2925             NaN         3.46
IVF-OPQ-nl223-m32-np14 (query)                        12_511.92       993.91    13_505.84       0.2924             NaN         3.46
IVF-OPQ-nl223-m32-np21 (query)                        12_511.92     1_459.26    13_971.19       0.2924             NaN         3.46
IVF-OPQ-nl223-m32 (self)                              12_511.92     6_282.61    18_794.54       0.7423             NaN         3.46
IVF-OPQ-nl223-m64-np11 (query)                        14_031.41     1_261.04    15_292.44       0.3941             NaN         4.99
IVF-OPQ-nl223-m64-np14 (query)                        14_031.41     1_599.85    15_631.26       0.3942             NaN         4.99
IVF-OPQ-nl223-m64-np21 (query)                        14_031.41     2_510.12    16_541.53       0.3944             NaN         4.99
IVF-OPQ-nl223-m64 (self)                              14_031.41     9_472.46    23_503.87       0.8169             NaN         4.99
IVF-OPQ-nl223-m128-np11 (query)                       20_892.11     2_227.16    23_119.26       0.6364             NaN         8.04
IVF-OPQ-nl223-m128-np14 (query)                       20_892.11     2_833.65    23_725.76       0.6369             NaN         8.04
IVF-OPQ-nl223-m128-np21 (query)                       20_892.11     4_244.77    25_136.88       0.6374             NaN         8.04
IVF-OPQ-nl223-m128 (self)                             20_892.11    15_607.37    36_499.47       0.8824             NaN         8.04
IVF-OPQ-nl316-m16-np15 (query)                         9_172.83       731.99     9_904.82       0.2473             NaN         2.88
IVF-OPQ-nl316-m16-np17 (query)                         9_172.83       831.64    10_004.47       0.2473             NaN         2.88
IVF-OPQ-nl316-m16-np25 (query)                         9_172.83     1_205.55    10_378.38       0.2473             NaN         2.88
IVF-OPQ-nl316-m16 (self)                               9_172.83     5_416.94    14_589.76       0.6676             NaN         2.88
IVF-OPQ-nl316-m32-np15 (query)                        12_604.79     1_029.06    13_633.85       0.3011             NaN         3.65
IVF-OPQ-nl316-m32-np17 (query)                        12_604.79     1_165.30    13_770.09       0.3011             NaN         3.65
IVF-OPQ-nl316-m32-np25 (query)                        12_604.79     1_685.74    14_290.53       0.3011             NaN         3.65
IVF-OPQ-nl316-m32 (self)                              12_604.79     7_533.07    20_137.86       0.7414             NaN         3.65
IVF-OPQ-nl316-m64-np15 (query)                        14_339.59     1_625.66    15_965.26       0.4052             NaN         5.17
IVF-OPQ-nl316-m64-np17 (query)                        14_339.59     1_840.61    16_180.21       0.4052             NaN         5.17
IVF-OPQ-nl316-m64-np25 (query)                        14_339.59     2_699.96    17_039.55       0.4053             NaN         5.17
IVF-OPQ-nl316-m64 (self)                              14_339.59    10_537.83    24_877.43       0.8153             NaN         5.17
IVF-OPQ-nl316-m128-np15 (query)                       20_900.26     2_883.89    23_784.15       0.6503             NaN         8.23
IVF-OPQ-nl316-m128-np17 (query)                       20_900.26     3_451.05    24_351.31       0.6505             NaN         8.23
IVF-OPQ-nl316-m128-np25 (query)                       20_900.26     4_856.22    25_756.49       0.6508             NaN         8.23
IVF-OPQ-nl316-m128 (self)                             20_900.26    17_532.38    38_432.64       0.8807             NaN         8.23
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
