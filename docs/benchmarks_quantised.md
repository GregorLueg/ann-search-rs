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
Exhaustive (query)                                         3.48     1_676.82     1_680.31       1.0000     0.000000        18.31
Exhaustive (self)                                          3.48    16_975.92    16_979.41       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.50     1_247.98     1_253.47       0.9867     0.065388         9.16
Exhaustive-BF16 (self)                                     5.50    16_704.02    16_709.51       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              481.74        88.83       570.57       0.9758     0.137513         9.19
IVF-BF16-nl273-np16 (query)                              481.74       103.23       584.96       0.9845     0.088329         9.19
IVF-BF16-nl273-np23 (query)                              481.74       147.52       629.26       0.9867     0.065388         9.19
IVF-BF16-nl273 (self)                                    481.74     1_555.35     2_037.09       0.9830     0.094689         9.19
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              810.03        92.65       902.68       0.9802     0.091487         9.21
IVF-BF16-nl387-np27 (query)                              810.03       121.52       931.55       0.9865     0.065640         9.21
IVF-BF16-nl387 (self)                                    810.03     1_201.53     2_011.56       0.9828     0.094875         9.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_530.20        88.77     1_618.97       0.9773     0.105124         9.23
IVF-BF16-nl547-np27 (query)                            1_530.20       103.98     1_634.17       0.9842     0.077306         9.23
IVF-BF16-nl547-np33 (query)                            1_530.20       119.41     1_649.61       0.9866     0.065956         9.23
IVF-BF16-nl547 (self)                                  1_530.20     1_155.62     2_685.82       0.9828     0.095761         9.23
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
Exhaustive (query)                                         4.51     1_661.30     1_665.81       1.0000     0.000000        18.88
Exhaustive (self)                                          4.51    16_556.76    16_561.27       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    6.43     1_304.48     1_310.91       0.9240     0.000230         9.44
Exhaustive-BF16 (self)                                     6.43    15_974.33    15_980.76       1.0000     0.000000         9.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              510.39       100.53       610.92       0.9163     0.000265         9.48
IVF-BF16-nl273-np16 (query)                              510.39       108.43       618.82       0.9222     0.000241         9.48
IVF-BF16-nl273-np23 (query)                              510.39       159.94       670.32       0.9240     0.000230         9.48
IVF-BF16-nl273 (self)                                    510.39     1_548.05     2_058.44       0.9229     0.001251         9.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              762.15        96.30       858.45       0.9200     0.000243         9.49
IVF-BF16-nl387-np27 (query)                              762.15       130.27       892.43       0.9239     0.000230         9.49
IVF-BF16-nl387 (self)                                    762.15     1_285.21     2_047.37       0.9228     0.001251         9.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_429.45        85.71     1_515.16       0.9183     0.000248         9.51
IVF-BF16-nl547-np27 (query)                            1_429.45        98.57     1_528.02       0.9224     0.000235         9.51
IVF-BF16-nl547-np33 (query)                            1_429.45       113.47     1_542.92       0.9239     0.000231         9.51
IVF-BF16-nl547 (self)                                  1_429.45     1_152.31     2_581.76       0.9228     0.001251         9.51
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
Exhaustive (query)                                         3.57     1_723.68     1_727.25       1.0000     0.000000        18.31
Exhaustive (self)                                          3.57    16_934.02    16_937.59       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    6.07     1_242.01     1_248.08       0.9649     0.019029         9.16
Exhaustive-BF16 (self)                                     6.07    16_796.95    16_803.01       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              424.54        89.89       514.43       0.9649     0.019034         9.19
IVF-BF16-nl273-np16 (query)                              424.54       106.13       530.67       0.9649     0.019029         9.19
IVF-BF16-nl273-np23 (query)                              424.54       141.64       566.18       0.9649     0.019029         9.19
IVF-BF16-nl273 (self)                                    424.54     1_456.06     1_880.61       0.9561     0.026167         9.19
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              783.10        97.30       880.40       0.9649     0.019040         9.21
IVF-BF16-nl387-np27 (query)                              783.10       133.62       916.73       0.9649     0.019029         9.21
IVF-BF16-nl387 (self)                                    783.10     1_309.44     2_092.54       0.9561     0.026167         9.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_495.00        85.01     1_580.01       0.9649     0.019032         9.23
IVF-BF16-nl547-np27 (query)                            1_495.00       104.77     1_599.78       0.9649     0.019031         9.23
IVF-BF16-nl547-np33 (query)                            1_495.00       115.52     1_610.52       0.9649     0.019029         9.23
IVF-BF16-nl547 (self)                                  1_495.00     1_124.26     2_619.26       0.9561     0.026167         9.23
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
Exhaustive (query)                                         3.51     1_639.16     1_642.67       1.0000     0.000000        18.31
Exhaustive (self)                                          3.51    16_777.70    16_781.22       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                    5.79     1_235.14     1_240.93       0.9348     0.158338         9.16
Exhaustive-BF16 (self)                                     5.79    16_858.68    16_864.47       1.0000     0.000000         9.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              434.81        88.98       523.79       0.9348     0.158338         9.19
IVF-BF16-nl273-np16 (query)                              434.81       108.84       543.66       0.9348     0.158338         9.19
IVF-BF16-nl273-np23 (query)                              434.81       151.24       586.06       0.9348     0.158338         9.19
IVF-BF16-nl273 (self)                                    434.81     1_554.99     1_989.80       0.9174     0.221294         9.19
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              817.35        97.81       915.16       0.9348     0.158338         9.21
IVF-BF16-nl387-np27 (query)                              817.35       132.12       949.47       0.9348     0.158338         9.21
IVF-BF16-nl387 (self)                                    817.35     1_312.89     2_130.24       0.9174     0.221294         9.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_530.52        83.37     1_613.89       0.9348     0.158338         9.23
IVF-BF16-nl547-np27 (query)                            1_530.52        97.54     1_628.06       0.9348     0.158338         9.23
IVF-BF16-nl547-np33 (query)                            1_530.52       116.16     1_646.68       0.9348     0.158338         9.23
IVF-BF16-nl547 (self)                                  1_530.52     1_122.05     2_652.57       0.9174     0.221294         9.23
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
Exhaustive (query)                                        16.09     7_151.16     7_167.24       1.0000     0.000000        73.24
Exhaustive (self)                                         16.09    66_208.80    66_224.88       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-BF16 (query)                                   26.49     5_612.95     5_639.44       0.9714     0.573744        36.62
Exhaustive-BF16 (self)                                    26.49    69_458.12    69_484.61       1.0000     0.000000        36.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl273-np13 (query)                              415.26       297.41       712.67       0.9712     0.574446        36.76
IVF-BF16-nl273-np16 (query)                              415.26       361.67       776.93       0.9713     0.573771        36.76
IVF-BF16-nl273-np23 (query)                              415.26       519.28       934.54       0.9714     0.573744        36.76
IVF-BF16-nl273 (self)                                    415.26     5_298.64     5_713.91       0.9637     0.816885        36.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl387-np19 (query)                              735.49       323.99     1_059.48       0.9713     0.573817        36.81
IVF-BF16-nl387-np27 (query)                              735.49       422.18     1_157.67       0.9714     0.573744        36.81
IVF-BF16-nl387 (self)                                    735.49     4_273.22     5_008.71       0.9637     0.816885        36.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-BF16-nl547-np23 (query)                            1_539.70       278.14     1_817.84       0.9713     0.573810        36.89
IVF-BF16-nl547-np27 (query)                            1_539.70       316.45     1_856.15       0.9714     0.573744        36.89
IVF-BF16-nl547-np33 (query)                            1_539.70       384.10     1_923.80       0.9714     0.573744        36.89
IVF-BF16-nl547 (self)                                  1_539.70     3_904.67     5_444.37       0.9637     0.816885        36.89
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
Exhaustive (query)                                         3.56     1_623.78     1_627.34       1.0000     0.000000        18.31
Exhaustive (self)                                          3.56    17_326.48    17_330.04       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     8.30       747.08       755.38       0.8011          NaN         4.58
Exhaustive-SQ8 (self)                                      8.30     7_904.08     7_912.38       0.8007          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               572.51        60.69       633.19       0.7779          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               572.51        66.64       639.15       0.7813          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               572.51        91.08       663.59       0.7822          NaN         4.61
IVF-SQ8-nl273 (self)                                     572.51       848.63     1_421.13       0.7819          NaN         4.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               795.05        53.38       848.43       0.7853          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               795.05        71.68       866.72       0.7878          NaN         4.63
IVF-SQ8-nl387 (self)                                     795.05       703.02     1_498.06       0.7872          NaN         4.63
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_610.35        52.33     1_662.69       0.7972          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_610.35        56.76     1_667.12       0.8002          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_610.35        65.55     1_675.90       0.8012          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_610.35       646.86     2_257.22       0.8007          NaN         4.65
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
Exhaustive (query)                                         4.42     1_682.22     1_686.64       1.0000     0.000000        18.88
Exhaustive (self)                                          4.42    16_861.57    16_866.00       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     7.60       725.54       733.14       0.8501          NaN         5.15
Exhaustive-SQ8 (self)                                      7.60     7_642.83     7_650.43       0.8497          NaN         5.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               443.16        50.72       493.88       0.8423          NaN         5.19
IVF-SQ8-nl273-np16 (query)                               443.16        57.55       500.71       0.8463          NaN         5.19
IVF-SQ8-nl273-np23 (query)                               443.16        78.93       522.09       0.8473          NaN         5.19
IVF-SQ8-nl273 (self)                                     443.16       793.94     1_237.09       0.8467          NaN         5.19
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               796.15        52.50       848.65       0.8420          NaN         5.20
IVF-SQ8-nl387-np27 (query)                               796.15        67.98       864.13       0.8449          NaN         5.20
IVF-SQ8-nl387 (self)                                     796.15       682.00     1_478.14       0.8446          NaN         5.20
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_450.60        49.76     1_500.36       0.8437          NaN         5.22
IVF-SQ8-nl547-np27 (query)                             1_450.60        53.98     1_504.58       0.8467          NaN         5.22
IVF-SQ8-nl547-np33 (query)                             1_450.60        65.24     1_515.84       0.8477          NaN         5.22
IVF-SQ8-nl547 (self)                                   1_450.60       623.46     2_074.07       0.8473          NaN         5.22
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
Exhaustive (query)                                         3.57     1_754.73     1_758.30       1.0000     0.000000        18.31
Exhaustive (self)                                          3.57    16_799.48    16_803.05       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                     8.15       753.10       761.25       0.6828          NaN         4.58
Exhaustive-SQ8 (self)                                      8.15     8_011.02     8_019.17       0.6835          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               602.64        57.50       660.14       0.6821          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               602.64        60.49       663.13       0.6821          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               602.64        91.66       694.30       0.6820          NaN         4.61
IVF-SQ8-nl273 (self)                                     602.64       828.32     1_430.96       0.6832          NaN         4.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               825.04        58.06       883.10       0.6842          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               825.04        81.52       906.56       0.6842          NaN         4.63
IVF-SQ8-nl387 (self)                                     825.04       736.67     1_561.70       0.6851          NaN         4.63
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_585.41        54.93     1_640.35       0.6826          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_585.41        64.95     1_650.36       0.6826          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_585.41        69.99     1_655.41       0.6826          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_585.41       693.76     2_279.17       0.6832          NaN         4.65
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
Exhaustive (query)                                         3.42     1_663.50     1_666.92       1.0000     0.000000        18.31
Exhaustive (self)                                          3.42    18_597.15    18_600.58       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    10.14       811.35       821.49       0.4800          NaN         4.58
Exhaustive-SQ8 (self)                                     10.14     8_378.47     8_388.61       0.4862          NaN         4.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               443.05        53.62       496.67       0.4788          NaN         4.61
IVF-SQ8-nl273-np16 (query)                               443.05        71.60       514.65       0.4787          NaN         4.61
IVF-SQ8-nl273-np23 (query)                               443.05        99.38       542.42       0.4786          NaN         4.61
IVF-SQ8-nl273 (self)                                     443.05       889.19     1_332.23       0.4863          NaN         4.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               826.63        56.65       883.29       0.4790          NaN         4.63
IVF-SQ8-nl387-np27 (query)                               826.63        75.93       902.56       0.4790          NaN         4.63
IVF-SQ8-nl387 (self)                                     826.63       773.05     1_599.68       0.4861          NaN         4.63
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_601.80        67.68     1_669.48       0.4800          NaN         4.65
IVF-SQ8-nl547-np27 (query)                             1_601.80        59.79     1_661.60       0.4799          NaN         4.65
IVF-SQ8-nl547-np33 (query)                             1_601.80        68.33     1_670.13       0.4799          NaN         4.65
IVF-SQ8-nl547 (self)                                   1_601.80       692.75     2_294.55       0.4865          NaN         4.65
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
Exhaustive (query)                                        15.42     7_643.28     7_658.70       1.0000     0.000000        73.24
Exhaustive (self)                                         15.42    75_928.07    75_943.50       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-SQ8 (query)                                    43.89     1_729.51     1_773.39       0.8081          NaN        18.31
Exhaustive-SQ8 (self)                                     43.89    17_747.31    17_791.19       0.8095          NaN        18.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl273-np13 (query)                               410.90       103.11       514.01       0.8062          NaN        18.45
IVF-SQ8-nl273-np16 (query)                               410.90       121.90       532.80       0.8062          NaN        18.45
IVF-SQ8-nl273-np23 (query)                               410.90       166.62       577.51       0.8062          NaN        18.45
IVF-SQ8-nl273 (self)                                     410.90     1_603.78     2_014.68       0.8082          NaN        18.45
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl387-np19 (query)                               750.34       110.99       861.33       0.8062          NaN        18.50
IVF-SQ8-nl387-np27 (query)                               750.34       152.44       902.77       0.8062          NaN        18.50
IVF-SQ8-nl387 (self)                                     750.34     1_340.80     2_091.13       0.8086          NaN        18.50
--------------------------------------------------------------------------------------------------------------------------------
IVF-SQ8-nl547-np23 (query)                             1_575.56       108.02     1_683.58       0.8078          NaN        18.58
IVF-SQ8-nl547-np27 (query)                             1_575.56       120.20     1_695.76       0.8078          NaN        18.58
IVF-SQ8-nl547-np33 (query)                             1_575.56       136.91     1_712.47       0.8078          NaN        18.58
IVF-SQ8-nl547 (self)                                   1_575.56     1_271.48     2_847.04       0.8097          NaN        18.58
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
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.29     6_877.18     6_892.48       1.0000     0.000000        73.24
Exhaustive (self)                                         15.29    67_250.30    67_265.60       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 702.84       881.05     1_583.89       0.0969          NaN         1.27
Exhaustive-PQ-m8 (self)                                  702.84     8_782.63     9_485.47       0.0835          NaN         1.27
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_059.80     1_969.56     3_029.37       0.1328          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_059.80    20_384.05    21_443.85       0.0967          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_460.70       290.96     1_751.67       0.1530          NaN         1.41
IVF-PQ-nl273-m8-np16 (query)                           1_460.70       323.43     1_784.13       0.1532          NaN         1.41
IVF-PQ-nl273-m8-np23 (query)                           1_460.70       477.95     1_938.65       0.1533          NaN         1.41
IVF-PQ-nl273-m8 (self)                                 1_460.70     4_700.73     6_161.44       0.1037          NaN         1.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_648.20       433.08     2_081.27       0.2676          NaN         2.55
IVF-PQ-nl273-m16-np16 (query)                          1_648.20       508.58     2_156.78       0.2689          NaN         2.55
IVF-PQ-nl273-m16-np23 (query)                          1_648.20       729.59     2_377.79       0.2696          NaN         2.55
IVF-PQ-nl273-m16 (self)                                1_648.20     7_131.18     8_779.37       0.1781          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_757.11       356.97     2_114.08       0.1545          NaN         1.46
IVF-PQ-nl387-m8-np27 (query)                           1_757.11       499.40     2_256.51       0.1545          NaN         1.46
IVF-PQ-nl387-m8 (self)                                 1_757.11     4_977.94     6_735.05       0.1042          NaN         1.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          2_106.06       557.47     2_663.53       0.2703          NaN         2.61
IVF-PQ-nl387-m16-np27 (query)                          2_106.06       781.35     2_887.41       0.2711          NaN         2.61
IVF-PQ-nl387-m16 (self)                                2_106.06     8_081.50    10_187.56       0.1807          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           3_200.28       417.98     3_618.25       0.1570          NaN         1.54
IVF-PQ-nl547-m8-np27 (query)                           3_200.28       506.91     3_707.19       0.1571          NaN         1.54
IVF-PQ-nl547-m8-np33 (query)                           3_200.28       605.40     3_805.68       0.1572          NaN         1.54
IVF-PQ-nl547-m8 (self)                                 3_200.28     6_120.81     9_321.09       0.1059          NaN         1.54
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          3_573.40       655.99     4_229.39       0.2711          NaN         2.69
IVF-PQ-nl547-m16-np27 (query)                          3_573.40       786.21     4_359.61       0.2723          NaN         2.69
IVF-PQ-nl547-m16-np33 (query)                          3_573.40       910.07     4_483.47       0.2727          NaN         2.69
IVF-PQ-nl547-m16 (self)                                3_573.40     9_414.41    12_987.82       0.1824          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        16.97     6_902.51     6_919.48       1.0000     0.000000        73.24
Exhaustive (self)                                         16.97    66_240.85    66_257.82       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 724.15       875.77     1_599.93       0.2215          NaN         1.27
Exhaustive-PQ-m8 (self)                                  724.15     8_805.58     9_529.73       0.1616          NaN         1.27
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_050.96     1_915.95     2_966.91       0.3518          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_050.96    20_554.95    21_605.91       0.2621          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_109.49       265.81     1_375.31       0.3766          NaN         1.41
IVF-PQ-nl273-m8-np16 (query)                           1_109.49       316.68     1_426.18       0.3768          NaN         1.41
IVF-PQ-nl273-m8-np23 (query)                           1_109.49       448.07     1_557.56       0.3768          NaN         1.41
IVF-PQ-nl273-m8 (self)                                 1_109.49     4_520.94     5_630.43       0.2757          NaN         1.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_418.22       409.22     1_827.44       0.5492          NaN         2.55
IVF-PQ-nl273-m16-np16 (query)                          1_418.22       512.43     1_930.65       0.5498          NaN         2.55
IVF-PQ-nl273-m16-np23 (query)                          1_418.22       707.25     2_125.47       0.5502          NaN         2.55
IVF-PQ-nl273-m16 (self)                                1_418.22     7_133.45     8_551.66       0.4521          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_570.58       355.22     1_925.80       0.3817          NaN         1.46
IVF-PQ-nl387-m8-np27 (query)                           1_570.58       506.25     2_076.83       0.3817          NaN         1.46
IVF-PQ-nl387-m8 (self)                                 1_570.58     5_021.64     6_592.22       0.2788          NaN         1.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          1_897.67       560.27     2_457.95       0.5528          NaN         2.61
IVF-PQ-nl387-m16-np27 (query)                          1_897.67       779.42     2_677.10       0.5532          NaN         2.61
IVF-PQ-nl387-m16 (self)                                1_897.67     7_925.02     9_822.69       0.4547          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           2_471.72       422.83     2_894.55       0.3862          NaN         1.54
IVF-PQ-nl547-m8-np27 (query)                           2_471.72       508.01     2_979.73       0.3862          NaN         1.54
IVF-PQ-nl547-m8-np33 (query)                           2_471.72       588.90     3_060.62       0.3863          NaN         1.54
IVF-PQ-nl547-m8 (self)                                 2_471.72     5_947.07     8_418.79       0.2832          NaN         1.54
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          2_795.78       650.74     3_446.52       0.5576          NaN         2.69
IVF-PQ-nl547-m16-np27 (query)                          2_795.78       752.35     3_548.14       0.5581          NaN         2.69
IVF-PQ-nl547-m16-np33 (query)                          2_795.78       912.99     3_708.78       0.5583          NaN         2.69
IVF-PQ-nl547-m16 (self)                                2_795.78     9_208.43    12_004.22       0.4599          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.25     6_332.48     6_347.73       1.0000     0.000000        73.24
Exhaustive (self)                                         15.25    62_096.14    62_111.40       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m8 (query)                                 750.82       874.23     1_625.06       0.2587          NaN         1.27
Exhaustive-PQ-m8 (self)                                  750.82     8_805.26     9_556.08       0.1987          NaN         1.27
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_053.54     1_907.89     2_961.42       0.3957          NaN         2.41
Exhaustive-PQ-m16 (self)                               1_053.54    19_170.54    20_224.08       0.2941          NaN         2.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m8-np13 (query)                           1_067.40       258.60     1_326.00       0.5459          NaN         1.41
IVF-PQ-nl273-m8-np16 (query)                           1_067.40       321.53     1_388.93       0.5459          NaN         1.41
IVF-PQ-nl273-m8-np23 (query)                           1_067.40       445.24     1_512.64       0.5459          NaN         1.41
IVF-PQ-nl273-m8 (self)                                 1_067.40     4_462.24     5_529.64       0.4006          NaN         1.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          1_381.99       414.70     1_796.69       0.6915          NaN         2.55
IVF-PQ-nl273-m16-np16 (query)                          1_381.99       498.29     1_880.28       0.6915          NaN         2.55
IVF-PQ-nl273-m16-np23 (query)                          1_381.99       713.32     2_095.31       0.6915          NaN         2.55
IVF-PQ-nl273-m16 (self)                                1_381.99     7_153.50     8_535.49       0.5839          NaN         2.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m8-np19 (query)                           1_406.23       368.00     1_774.23       0.5538          NaN         1.46
IVF-PQ-nl387-m8-np27 (query)                           1_406.23       490.88     1_897.10       0.5538          NaN         1.46
IVF-PQ-nl387-m8 (self)                                 1_406.23     4_951.77     6_358.00       0.4015          NaN         1.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          1_776.23       562.15     2_338.38       0.6976          NaN         2.61
IVF-PQ-nl387-m16-np27 (query)                          1_776.23       784.17     2_560.40       0.6976          NaN         2.61
IVF-PQ-nl387-m16 (self)                                1_776.23     7_900.92     9_677.15       0.5860          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m8-np23 (query)                           2_279.83       423.22     2_703.05       0.5594          NaN         1.54
IVF-PQ-nl547-m8-np27 (query)                           2_279.83       481.68     2_761.51       0.5594          NaN         1.54
IVF-PQ-nl547-m8-np33 (query)                           2_279.83       581.55     2_861.39       0.5594          NaN         1.54
IVF-PQ-nl547-m8 (self)                                 2_279.83     5_876.53     8_156.36       0.4052          NaN         1.54
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          2_590.84       637.89     3_228.73       0.7013          NaN         2.69
IVF-PQ-nl547-m16-np27 (query)                          2_590.84       763.83     3_354.67       0.7013          NaN         2.69
IVF-PQ-nl547-m16-np33 (query)                          2_590.84       905.59     3_496.43       0.7013          NaN         2.69
IVF-PQ-nl547-m16 (self)                                2_590.84     9_128.86    11_719.70       0.5895          NaN         2.69
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 256 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is just bad, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>PQ quantisations - Euclidean (Gaussian - 256 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.60    15_365.02    15_394.62       1.0000     0.000000       146.48
Exhaustive (self)                                         29.60   162_625.04   162_654.64       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_569.09     1_940.41     3_509.50       0.0968          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_569.09    19_218.83    20_787.93       0.0817          NaN         2.54
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m32 (query)                              2_031.40     4_393.24     6_424.64       0.1321          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_031.40    43_918.96    45_950.36       0.0941          NaN         4.83
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m64 (query)                              3_430.68    11_920.82    15_351.51       0.2669          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_430.68   119_428.09   122_858.77       0.1882          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_256.32       522.81     2_779.13       0.1529          NaN         2.81
IVF-PQ-nl273-m16-np16 (query)                          2_256.32       665.14     2_921.46       0.1534          NaN         2.81
IVF-PQ-nl273-m16-np23 (query)                          2_256.32       929.01     3_185.33       0.1536          NaN         2.81
IVF-PQ-nl273-m16 (self)                                2_256.32     9_250.79    11_507.11       0.1033          NaN         2.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_810.69       862.92     3_673.61       0.2653          NaN         5.10
IVF-PQ-nl273-m32-np16 (query)                          2_810.69     1_044.65     3_855.34       0.2673          NaN         5.10
IVF-PQ-nl273-m32-np23 (query)                          2_810.69     1_487.75     4_298.44       0.2678          NaN         5.10
IVF-PQ-nl273-m32 (self)                                2_810.69    14_835.23    17_645.91       0.1821          NaN         5.10
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_208.31     1_688.04     5_896.35       0.5114          NaN         9.68
IVF-PQ-nl273-m64-np16 (query)                          4_208.31     2_043.06     6_251.36       0.5180          NaN         9.68
IVF-PQ-nl273-m64-np23 (query)                          4_208.31     2_902.32     7_110.63       0.5205          NaN         9.68
IVF-PQ-nl273-m64 (self)                                4_208.31    29_047.30    33_255.61       0.4336          NaN         9.68
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          3_133.53       706.43     3_839.97       0.1535          NaN         2.92
IVF-PQ-nl387-m16-np27 (query)                          3_133.53       986.57     4_120.10       0.1537          NaN         2.92
IVF-PQ-nl387-m16 (self)                                3_133.53     9_850.75    12_984.28       0.1036          NaN         2.92
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_931.96     1_151.93     5_083.89       0.2664          NaN         5.21
IVF-PQ-nl387-m32-np27 (query)                          3_931.96     1_601.92     5_533.88       0.2674          NaN         5.21
IVF-PQ-nl387-m32 (self)                                3_931.96    16_870.41    20_802.37       0.1827          NaN         5.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          5_495.04     2_194.43     7_689.47       0.5160          NaN         9.79
IVF-PQ-nl387-m64-np27 (query)                          5_495.04     3_077.21     8_572.26       0.5202          NaN         9.79
IVF-PQ-nl387-m64 (self)                                5_495.04    30_482.59    35_977.64       0.4347          NaN         9.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          4_977.71       842.12     5_819.83       0.1549          NaN         3.08
IVF-PQ-nl547-m16-np27 (query)                          4_977.71       965.32     5_943.03       0.1553          NaN         3.08
IVF-PQ-nl547-m16-np33 (query)                          4_977.71     1_166.06     6_143.77       0.1555          NaN         3.08
IVF-PQ-nl547-m16 (self)                                4_977.71    11_662.45    16_640.16       0.1049          NaN         3.08
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          5_605.83     1_329.51     6_935.35       0.2686          NaN         5.37
IVF-PQ-nl547-m32-np27 (query)                          5_605.83     1_528.03     7_133.87       0.2704          NaN         5.37
IVF-PQ-nl547-m32-np33 (query)                          5_605.83     1_845.61     7_451.44       0.2708          NaN         5.37
IVF-PQ-nl547-m32 (self)                                5_605.83    18_430.01    24_035.84       0.1839          NaN         5.37
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          6_973.67     2_450.29     9_423.96       0.5158          NaN         9.95
IVF-PQ-nl547-m64-np27 (query)                          6_973.67     2_855.22     9_828.89       0.5221          NaN         9.95
IVF-PQ-nl547-m64-np33 (query)                          6_973.67     3_421.73    10_395.40       0.5240          NaN         9.95
IVF-PQ-nl547-m64 (self)                                6_973.67    34_603.71    41_577.38       0.4379          NaN         9.95
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Correlated - 256 dim)</b>:</summary>
<pre><code>
</br>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.60    15_569.10    15_598.70       1.0000     0.000000       146.48
Exhaustive (self)                                         29.60   156_228.66   156_258.26       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_418.90     1_938.44     3_357.34       0.2357          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_418.90    19_484.47    20_903.36       0.1675          NaN         2.54
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m32 (query)                              2_058.35     4_515.59     6_573.94       0.3554          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_058.35    45_466.40    47_524.76       0.2679          NaN         4.83
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m64 (query)                              3_688.41    12_210.53    15_898.94       0.4755          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_688.41   121_536.65   125_225.07       0.3907          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_216.46       523.96     2_740.42       0.3836          NaN         2.81
IVF-PQ-nl273-m16-np16 (query)                          2_216.46       657.22     2_873.68       0.3847          NaN         2.81
IVF-PQ-nl273-m16-np23 (query)                          2_216.46       913.60     3_130.06       0.3852          NaN         2.81
IVF-PQ-nl273-m16 (self)                                2_216.46     9_213.63    11_430.09       0.2876          NaN         2.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_763.68       864.45     3_628.13       0.5548          NaN         5.10
IVF-PQ-nl273-m32-np16 (query)                          2_763.68     1_074.70     3_838.38       0.5569          NaN         5.10
IVF-PQ-nl273-m32-np23 (query)                          2_763.68     1_508.50     4_272.17       0.5578          NaN         5.10
IVF-PQ-nl273-m32 (self)                                2_763.68    14_931.55    17_695.22       0.4667          NaN         5.10
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_111.86     1_694.07     5_805.93       0.7063          NaN         9.68
IVF-PQ-nl273-m64-np16 (query)                          4_111.86     2_068.87     6_180.73       0.7100          NaN         9.68
IVF-PQ-nl273-m64-np23 (query)                          4_111.86     2_898.14     7_010.00       0.7118          NaN         9.68
IVF-PQ-nl273-m64 (self)                                4_111.86    29_048.02    33_159.88       0.6484          NaN         9.68
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          2_988.35       710.61     3_698.96       0.3869          NaN         2.92
IVF-PQ-nl387-m16-np27 (query)                          2_988.35       978.46     3_966.81       0.3875          NaN         2.92
IVF-PQ-nl387-m16 (self)                                2_988.35     9_809.30    12_797.65       0.2905          NaN         2.92
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_591.01     1_131.36     4_722.37       0.5561          NaN         5.21
IVF-PQ-nl387-m32-np27 (query)                          3_591.01     1_589.65     5_180.66       0.5576          NaN         5.21
IVF-PQ-nl387-m32 (self)                                3_591.01    16_017.70    19_608.70       0.4682          NaN         5.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          4_992.35     2_163.54     7_155.89       0.7108          NaN         9.79
IVF-PQ-nl387-m64-np27 (query)                          4_992.35     3_032.63     8_024.99       0.7133          NaN         9.79
IVF-PQ-nl387-m64 (self)                                4_992.35    30_315.06    35_307.41       0.6499          NaN         9.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          4_218.40       831.26     5_049.66       0.3895          NaN         3.08
IVF-PQ-nl547-m16-np27 (query)                          4_218.40       960.87     5_179.27       0.3901          NaN         3.08
IVF-PQ-nl547-m16-np33 (query)                          4_218.40     1_171.02     5_389.42       0.3904          NaN         3.08
IVF-PQ-nl547-m16 (self)                                4_218.40    11_788.93    16_007.34       0.2925          NaN         3.08
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          4_860.86     1_305.60     6_166.47       0.5574          NaN         5.37
IVF-PQ-nl547-m32-np27 (query)                          4_860.86     1_519.32     6_380.19       0.5590          NaN         5.37
IVF-PQ-nl547-m32-np33 (query)                          4_860.86     1_837.84     6_698.71       0.5600          NaN         5.37
IVF-PQ-nl547-m32 (self)                                4_860.86    18_425.71    23_286.57       0.4697          NaN         5.37
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          6_261.06     2_408.62     8_669.68       0.7121          NaN         9.95
IVF-PQ-nl547-m64-np27 (query)                          6_261.06     2_805.60     9_066.66       0.7150          NaN         9.95
IVF-PQ-nl547-m64-np33 (query)                          6_261.06     3_526.68     9_787.74       0.7166          NaN         9.95
IVF-PQ-nl547-m64 (self)                                6_261.06    34_557.58    40_818.64       0.6535          NaN         9.95
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>PQ quantisations - Euclidean (Low Rank - 256 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.49    15_753.31    15_782.80       1.0000     0.000000       146.48
Exhaustive (self)                                         29.49   157_716.47   157_745.96       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m16 (query)                              1_416.67     1_935.25     3_351.92       0.3801          NaN         2.54
Exhaustive-PQ-m16 (self)                               1_416.67    19_475.72    20_892.39       0.2823          NaN         2.54
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m32 (query)                              2_073.09     4_445.95     6_519.04       0.5114          NaN         4.83
Exhaustive-PQ-m32 (self)                               2_073.09    44_643.69    46_716.78       0.4053          NaN         4.83
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-PQ-m64 (query)                              3_614.21    12_120.90    15_735.11       0.6355          NaN         9.41
Exhaustive-PQ-m64 (self)                               3_614.21   121_165.02   124_779.23       0.5576          NaN         9.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m16-np13 (query)                          2_095.55       535.11     2_630.66       0.6299          NaN         2.81
IVF-PQ-nl273-m16-np16 (query)                          2_095.55       650.39     2_745.94       0.6300          NaN         2.81
IVF-PQ-nl273-m16-np23 (query)                          2_095.55       917.67     3_013.22       0.6300          NaN         2.81
IVF-PQ-nl273-m16 (self)                                2_095.55     9_317.99    11_413.54       0.4800          NaN         2.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m32-np13 (query)                          2_687.55       864.19     3_551.74       0.7571          NaN         5.10
IVF-PQ-nl273-m32-np16 (query)                          2_687.55     1_085.17     3_772.72       0.7572          NaN         5.10
IVF-PQ-nl273-m32-np23 (query)                          2_687.55     1_511.83     4_199.37       0.7572          NaN         5.10
IVF-PQ-nl273-m32 (self)                                2_687.55    15_002.62    17_690.17       0.6617          NaN         5.10
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl273-m64-np13 (query)                          4_163.83     1_703.15     5_866.98       0.8790          NaN         9.68
IVF-PQ-nl273-m64-np16 (query)                          4_163.83     2_063.88     6_227.71       0.8792          NaN         9.68
IVF-PQ-nl273-m64-np23 (query)                          4_163.83     2_927.67     7_091.50       0.8792          NaN         9.68
IVF-PQ-nl273-m64 (self)                                4_163.83    29_325.04    33_488.88       0.8360          NaN         9.68
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m16-np19 (query)                          2_664.88       722.37     3_387.24       0.6291          NaN         2.92
IVF-PQ-nl387-m16-np27 (query)                          2_664.88       991.70     3_656.57       0.6291          NaN         2.92
IVF-PQ-nl387-m16 (self)                                2_664.88    10_068.11    12_732.99       0.4710          NaN         2.92
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m32-np19 (query)                          3_354.96     1_143.37     4_498.33       0.7564          NaN         5.21
IVF-PQ-nl387-m32-np27 (query)                          3_354.96     1_620.58     4_975.54       0.7564          NaN         5.21
IVF-PQ-nl387-m32 (self)                                3_354.96    16_115.99    19_470.95       0.6572          NaN         5.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl387-m64-np19 (query)                          4_832.84     2_180.43     7_013.27       0.8820          NaN         9.79
IVF-PQ-nl387-m64-np27 (query)                          4_832.84     3_075.51     7_908.35       0.8820          NaN         9.79
IVF-PQ-nl387-m64 (self)                                4_832.84    30_735.20    35_568.04       0.8375          NaN         9.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m16-np23 (query)                          3_878.02       836.33     4_714.34       0.6317          NaN         3.08
IVF-PQ-nl547-m16-np27 (query)                          3_878.02       985.95     4_863.97       0.6317          NaN         3.08
IVF-PQ-nl547-m16-np33 (query)                          3_878.02     1_182.92     5_060.94       0.6317          NaN         3.08
IVF-PQ-nl547-m16 (self)                                3_878.02    12_010.57    15_888.58       0.4646          NaN         3.08
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m32-np23 (query)                          4_500.69     1_314.45     5_815.14       0.7595          NaN         5.37
IVF-PQ-nl547-m32-np27 (query)                          4_500.69     1_550.14     6_050.84       0.7595          NaN         5.37
IVF-PQ-nl547-m32-np33 (query)                          4_500.69     1_906.15     6_406.85       0.7595          NaN         5.37
IVF-PQ-nl547-m32 (self)                                4_500.69    18_824.38    23_325.07       0.6549          NaN         5.37
--------------------------------------------------------------------------------------------------------------------------------
IVF-PQ-nl547-m64-np23 (query)                          5_946.52     2_438.15     8_384.67       0.8845          NaN         9.95
IVF-PQ-nl547-m64-np27 (query)                          5_946.52     2_848.03     8_794.54       0.8845          NaN         9.95
IVF-PQ-nl547-m64-np33 (query)                          5_946.52     3_460.99     9_407.51       0.8845          NaN         9.95
IVF-PQ-nl547-m64 (self)                                5_946.52    34_604.17    40_550.69       0.8389          NaN         9.95
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
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.26     6_321.52     6_336.78       1.0000     0.000000        73.24
Exhaustive (self)                                         15.26    63_670.11    63_685.37       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              4_682.31       884.49     5_566.80       0.0958          NaN         1.33
Exhaustive-OPQ-m8 (self)                               4_682.31     9_106.27    13_788.58       0.0833          NaN         1.33
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_694.56     1_932.67     5_627.23       0.1326          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_694.56    19_563.55    23_258.11       0.0961          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          4_304.28       288.03     4_592.30       0.1519          NaN         1.47
IVF-OPQ-nl273-m8-np16 (query)                          4_304.28       313.26     4_617.53       0.1520          NaN         1.47
IVF-OPQ-nl273-m8-np23 (query)                          4_304.28       447.40     4_751.68       0.1521          NaN         1.47
IVF-OPQ-nl273-m8 (self)                                4_304.28     4_690.27     8_994.55       0.1034          NaN         1.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         4_173.82       419.87     4_593.69       0.2671          NaN         2.61
IVF-OPQ-nl273-m16-np16 (query)                         4_173.82       512.55     4_686.37       0.2686          NaN         2.61
IVF-OPQ-nl273-m16-np23 (query)                         4_173.82       710.22     4_884.04       0.2690          NaN         2.61
IVF-OPQ-nl273-m16 (self)                               4_173.82     7_406.19    11_580.01       0.1785          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          5_028.67       358.69     5_387.37       0.1547          NaN         1.52
IVF-OPQ-nl387-m8-np27 (query)                          5_028.67       501.26     5_529.94       0.1549          NaN         1.52
IVF-OPQ-nl387-m8 (self)                                5_028.67     5_318.51    10_347.19       0.1045          NaN         1.52
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         4_970.52       584.02     5_554.53       0.2683          NaN         2.67
IVF-OPQ-nl387-m16-np27 (query)                         4_970.52       822.82     5_793.34       0.2690          NaN         2.67
IVF-OPQ-nl387-m16 (self)                               4_970.52     8_321.08    13_291.60       0.1801          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          6_327.38       421.86     6_749.24       0.1559          NaN         1.60
IVF-OPQ-nl547-m8-np27 (query)                          6_327.38       515.77     6_843.15       0.1562          NaN         1.60
IVF-OPQ-nl547-m8-np33 (query)                          6_327.38       599.90     6_927.28       0.1562          NaN         1.60
IVF-OPQ-nl547-m8 (self)                                6_327.38     6_338.93    12_666.31       0.1053          NaN         1.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         6_281.58       673.70     6_955.29       0.2693          NaN         2.75
IVF-OPQ-nl547-m16-np27 (query)                         6_281.58       795.50     7_077.09       0.2705          NaN         2.75
IVF-OPQ-nl547-m16-np33 (query)                         6_281.58       971.61     7_253.20       0.2710          NaN         2.75
IVF-OPQ-nl547-m16 (self)                               6_281.58     9_820.50    16_102.09       0.1820          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>OPQ quantisations - Euclidean (Correlated - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.14     6_530.12     6_545.26       1.0000     0.000000        73.24
Exhaustive (self)                                         15.14    64_423.93    64_439.07       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              4_688.06       901.55     5_589.61       0.2221          NaN         1.33
Exhaustive-OPQ-m8 (self)                               4_688.06     9_120.66    13_808.72       0.1616          NaN         1.33
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_571.27     1_925.62     5_496.89       0.3510          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_571.27    19_514.04    23_085.31       0.2620          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          4_207.12       277.64     4_484.76       0.3759          NaN         1.47
IVF-OPQ-nl273-m8-np16 (query)                          4_207.12       322.43     4_529.55       0.3760          NaN         1.47
IVF-OPQ-nl273-m8-np23 (query)                          4_207.12       447.67     4_654.78       0.3761          NaN         1.47
IVF-OPQ-nl273-m8 (self)                                4_207.12     4_694.45     8_901.57       0.2759          NaN         1.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         4_080.73       432.75     4_513.48       0.5484          NaN         2.61
IVF-OPQ-nl273-m16-np16 (query)                         4_080.73       506.09     4_586.82       0.5491          NaN         2.61
IVF-OPQ-nl273-m16-np23 (query)                         4_080.73       732.90     4_813.63       0.5493          NaN         2.61
IVF-OPQ-nl273-m16 (self)                               4_080.73     7_527.81    11_608.54       0.4515          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          4_441.40       362.44     4_803.84       0.3820          NaN         1.52
IVF-OPQ-nl387-m8-np27 (query)                          4_441.40       542.81     4_984.21       0.3821          NaN         1.52
IVF-OPQ-nl387-m8 (self)                                4_441.40     5_357.11     9_798.51       0.2792          NaN         1.52
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         4_406.12       575.50     4_981.62       0.5522          NaN         2.67
IVF-OPQ-nl387-m16-np27 (query)                         4_406.12       819.14     5_225.26       0.5525          NaN         2.67
IVF-OPQ-nl387-m16 (self)                               4_406.12     8_391.66    12_797.78       0.4549          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          5_673.12       478.40     6_151.51       0.3861          NaN         1.60
IVF-OPQ-nl547-m8-np27 (query)                          5_673.12       573.41     6_246.53       0.3862          NaN         1.60
IVF-OPQ-nl547-m8-np33 (query)                          5_673.12       651.86     6_324.98       0.3862          NaN         1.60
IVF-OPQ-nl547-m8 (self)                                5_673.12     6_708.64    12_381.76       0.2836          NaN         1.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         5_623.42       662.81     6_286.24       0.5567          NaN         2.75
IVF-OPQ-nl547-m16-np27 (query)                         5_623.42       776.69     6_400.12       0.5572          NaN         2.75
IVF-OPQ-nl547-m16-np33 (query)                         5_623.42       947.04     6_570.47       0.5574          NaN         2.75
IVF-OPQ-nl547-m16 (self)                               5_623.42     9_783.92    15_407.35       0.4593          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>OPQ quantisations - Euclidean (Low Rank - 128 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.43     6_309.77     6_324.19       1.0000     0.000000        73.24
Exhaustive (self)                                         14.43    67_582.69    67_597.12       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m8 (query)                              3_392.04       883.13     4_275.18       0.2585          NaN         1.33
Exhaustive-OPQ-m8 (self)                               3_392.04     9_114.65    12_506.69       0.1648          NaN         1.33
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             3_979.35     1_939.07     5_918.41       0.3956          NaN         2.48
Exhaustive-OPQ-m16 (self)                              3_979.35    19_630.09    23_609.44       0.2451          NaN         2.48
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m8-np13 (query)                          4_700.01       270.00     4_970.01       0.6165          NaN         1.47
IVF-OPQ-nl273-m8-np16 (query)                          4_700.01       318.44     5_018.45       0.6165          NaN         1.47
IVF-OPQ-nl273-m8-np23 (query)                          4_700.01       448.49     5_148.50       0.6165          NaN         1.47
IVF-OPQ-nl273-m8 (self)                                4_700.01     4_730.52     9_430.53       0.5159          NaN         1.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         3_925.94       421.17     4_347.12       0.7432          NaN         2.61
IVF-OPQ-nl273-m16-np16 (query)                         3_925.94       513.67     4_439.61       0.7432          NaN         2.61
IVF-OPQ-nl273-m16-np23 (query)                         3_925.94       729.85     4_655.79       0.7432          NaN         2.61
IVF-OPQ-nl273-m16 (self)                               3_925.94     7_515.53    11_441.48       0.6640          NaN         2.61
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m8-np19 (query)                          4_464.50       367.85     4_832.35       0.6235          NaN         1.52
IVF-OPQ-nl387-m8-np27 (query)                          4_464.50       505.86     4_970.36       0.6235          NaN         1.52
IVF-OPQ-nl387-m8 (self)                                4_464.50     5_370.47     9_834.97       0.5234          NaN         1.52
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         4_334.11       591.57     4_925.68       0.7467          NaN         2.67
IVF-OPQ-nl387-m16-np27 (query)                         4_334.11       804.85     5_138.96       0.7467          NaN         2.67
IVF-OPQ-nl387-m16 (self)                               4_334.11     8_395.65    12_729.76       0.6683          NaN         2.67
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m8-np23 (query)                          5_510.57       434.30     5_944.88       0.6278          NaN         1.60
IVF-OPQ-nl547-m8-np27 (query)                          5_510.57       521.43     6_032.00       0.6278          NaN         1.60
IVF-OPQ-nl547-m8-np33 (query)                          5_510.57       603.69     6_114.26       0.6278          NaN         1.60
IVF-OPQ-nl547-m8 (self)                                5_510.57     6_366.47    11_877.04       0.5284          NaN         1.60
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         5_137.96       687.20     5_825.17       0.7479          NaN         2.75
IVF-OPQ-nl547-m16-np27 (query)                         5_137.96       780.11     5_918.08       0.7479          NaN         2.75
IVF-OPQ-nl547-m16-np33 (query)                         5_137.96       969.08     6_107.04       0.7479          NaN         2.75
IVF-OPQ-nl547-m16 (self)                               5_137.96     9_789.17    14_927.14       0.6713          NaN         2.75
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 256 dimensions

This is the area in which these indices start making more sense. With "Gaussian
blob" data, the performance is not too great, but the moment there is more
intrinsic data in the structure, the performance increases.

<details>
<summary><b>OPQ quantisations - Euclidean (Gaussian - 256 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.87    15_720.84    15_750.72       1.0000     0.000000       146.48
Exhaustive (self)                                         29.87   156_392.02   156_421.89       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                            10_199.98     1_932.45    12_132.43       0.0953          NaN         2.79
Exhaustive-OPQ-m16 (self)                             10_199.98    20_298.06    30_498.04       0.0813          NaN         2.79
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m32 (query)                             7_510.43     4_468.12    11_978.55       0.1296          NaN         5.08
Exhaustive-OPQ-m32 (self)                              7_510.43    45_757.34    53_267.77       0.0937          NaN         5.08
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m64 (query)                            11_438.92    12_295.20    23_734.13       0.2634          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_438.92   122_375.90   133_814.83       0.1887          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         9_320.79       529.59     9_850.38       0.1507          NaN         3.06
IVF-OPQ-nl273-m16-np16 (query)                         9_320.79       673.29     9_994.08       0.1514          NaN         3.06
IVF-OPQ-nl273-m16-np23 (query)                         9_320.79       914.31    10_235.10       0.1515          NaN         3.06
IVF-OPQ-nl273-m16 (self)                               9_320.79    10_402.34    19_723.12       0.1025          NaN         3.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         8_245.80       880.68     9_126.47       0.2648          NaN         5.35
IVF-OPQ-nl273-m32-np16 (query)                         8_245.80     1_067.19     9_312.99       0.2670          NaN         5.35
IVF-OPQ-nl273-m32-np23 (query)                         8_245.80     1_542.91     9_788.70       0.2675          NaN         5.35
IVF-OPQ-nl273-m32 (self)                               8_245.80    16_355.75    24_601.55       0.1822          NaN         5.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        12_177.18     1_725.64    13_902.82       0.5108          NaN         9.93
IVF-OPQ-nl273-m64-np16 (query)                        12_177.18     2_073.58    14_250.76       0.5174          NaN         9.93
IVF-OPQ-nl273-m64-np23 (query)                        12_177.18     2_924.94    15_102.12       0.5199          NaN         9.93
IVF-OPQ-nl273-m64 (self)                              12_177.18    30_234.93    42_412.11       0.4333          NaN         9.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         9_812.53       740.64    10_553.17       0.1523          NaN         3.17
IVF-OPQ-nl387-m16-np27 (query)                         9_812.53       999.81    10_812.34       0.1525          NaN         3.17
IVF-OPQ-nl387-m16 (self)                               9_812.53    11_015.52    20_828.05       0.1040          NaN         3.17
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_195.51     1_172.38    10_367.89       0.2645          NaN         5.46
IVF-OPQ-nl387-m32-np27 (query)                         9_195.51     1_615.83    10_811.34       0.2654          NaN         5.46
IVF-OPQ-nl387-m32 (self)                               9_195.51    17_097.97    26_293.47       0.1812          NaN         5.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        13_078.01     2_233.85    15_311.86       0.5155          NaN        10.04
IVF-OPQ-nl387-m64-np27 (query)                        13_078.01     3_088.21    16_166.22       0.5201          NaN        10.04
IVF-OPQ-nl387-m64 (self)                              13_078.01    31_644.49    44_722.50       0.4343          NaN        10.04
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                        11_741.05       857.34    12_598.39       0.1537          NaN         3.33
IVF-OPQ-nl547-m16-np27 (query)                        11_741.05       964.35    12_705.40       0.1540          NaN         3.33
IVF-OPQ-nl547-m16-np33 (query)                        11_741.05     1_181.32    12_922.37       0.1541          NaN         3.33
IVF-OPQ-nl547-m16 (self)                              11_741.05    12_768.41    24_509.46       0.1044          NaN         3.33
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                        11_357.34     1_325.64    12_682.97       0.2664          NaN         5.62
IVF-OPQ-nl547-m32-np27 (query)                        11_357.34     1_535.39    12_892.72       0.2683          NaN         5.62
IVF-OPQ-nl547-m32-np33 (query)                        11_357.34     1_849.41    13_206.74       0.2689          NaN         5.62
IVF-OPQ-nl547-m32 (self)                              11_357.34    19_717.28    31_074.61       0.1840          NaN         5.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        15_914.75     2_480.17    18_394.92       0.5154          NaN        10.20
IVF-OPQ-nl547-m64-np27 (query)                        15_914.75     2_866.45    18_781.19       0.5220          NaN        10.20
IVF-OPQ-nl547-m64-np33 (query)                        15_914.75     3_455.15    19_369.90       0.5244          NaN        10.20
IVF-OPQ-nl547-m64 (self)                              15_914.75    35_563.28    51_478.02       0.4373          NaN        10.20
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>OPQ quantisations - Euclidean (Correlated - 256 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        29.77    16_329.47    16_359.24       1.0000     0.000000       146.48
Exhaustive (self)                                         29.77   159_404.45   159_434.22       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             7_036.83     1_929.28     8_966.10       0.2364          NaN         2.79
Exhaustive-OPQ-m16 (self)                              7_036.83    20_603.03    27_639.85       0.1672          NaN         2.79
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m32 (query)                             7_380.05     4_406.21    11_786.26       0.3537          NaN         5.08
Exhaustive-OPQ-m32 (self)                              7_380.05    45_408.65    52_788.70       0.2677          NaN         5.08
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m64 (query)                            11_271.23    12_077.73    23_348.96       0.4830          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_271.23   121_072.26   132_343.49       0.4018          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         8_567.30       587.23     9_154.53       0.3841          NaN         3.06
IVF-OPQ-nl273-m16-np16 (query)                         8_567.30       703.97     9_271.27       0.3851          NaN         3.06
IVF-OPQ-nl273-m16-np23 (query)                         8_567.30       994.28     9_561.57       0.3855          NaN         3.06
IVF-OPQ-nl273-m16 (self)                               8_567.30    10_851.40    19_418.70       0.2879          NaN         3.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         8_331.13       880.52     9_211.65       0.5535          NaN         5.35
IVF-OPQ-nl273-m32-np16 (query)                         8_331.13     1_077.16     9_408.29       0.5559          NaN         5.35
IVF-OPQ-nl273-m32-np23 (query)                         8_331.13     1_488.90     9_820.03       0.5570          NaN         5.35
IVF-OPQ-nl273-m32 (self)                               8_331.13    16_111.82    24_442.95       0.4668          NaN         5.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        12_785.10     1_781.29    14_566.39       0.7100          NaN         9.93
IVF-OPQ-nl273-m64-np16 (query)                        12_785.10     2_136.31    14_921.41       0.7139          NaN         9.93
IVF-OPQ-nl273-m64-np23 (query)                        12_785.10     2_878.91    15_664.01       0.7156          NaN         9.93
IVF-OPQ-nl273-m64 (self)                              12_785.10    30_544.82    43_329.92       0.6537          NaN         9.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         8_299.67       705.60     9_005.27       0.3882          NaN         3.17
IVF-OPQ-nl387-m16-np27 (query)                         8_299.67       982.16     9_281.83       0.3887          NaN         3.17
IVF-OPQ-nl387-m16 (self)                               8_299.67    10_775.61    19_075.28       0.2903          NaN         3.17
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_608.43     1_191.72    10_800.15       0.5569          NaN         5.46
IVF-OPQ-nl387-m32-np27 (query)                         9_608.43     1_605.54    11_213.98       0.5584          NaN         5.46
IVF-OPQ-nl387-m32 (self)                               9_608.43    17_813.02    27_421.45       0.4675          NaN         5.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        13_598.17     2_365.74    15_963.91       0.7131          NaN        10.04
IVF-OPQ-nl387-m64-np27 (query)                        13_598.17     3_072.19    16_670.36       0.7156          NaN        10.04
IVF-OPQ-nl387-m64 (self)                              13_598.17    31_140.75    44_738.92       0.6546          NaN        10.04
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         9_766.62       819.50    10_586.12       0.3895          NaN         3.33
IVF-OPQ-nl547-m16-np27 (query)                         9_766.62       976.77    10_743.40       0.3902          NaN         3.33
IVF-OPQ-nl547-m16-np33 (query)                         9_766.62     1_179.51    10_946.13       0.3905          NaN         3.33
IVF-OPQ-nl547-m16 (self)                               9_766.62    12_748.99    22_515.61       0.2926          NaN         3.33
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                        10_904.57     1_304.92    12_209.49       0.5567          NaN         5.62
IVF-OPQ-nl547-m32-np27 (query)                        10_904.57     1_519.39    12_423.96       0.5584          NaN         5.62
IVF-OPQ-nl547-m32-np33 (query)                        10_904.57     1_822.40    12_726.98       0.5593          NaN         5.62
IVF-OPQ-nl547-m32 (self)                              10_904.57    19_506.16    30_410.73       0.4697          NaN         5.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        15_243.36     2_394.05    17_637.41       0.7134          NaN        10.20
IVF-OPQ-nl547-m64-np27 (query)                        15_243.36     3_001.22    18_244.58       0.7161          NaN        10.20
IVF-OPQ-nl547-m64-np33 (query)                        15_243.36     3_481.10    18_724.46       0.7176          NaN        10.20
IVF-OPQ-nl547-m64 (self)                              15_243.36    36_124.69    51_368.05       0.6576          NaN        10.20
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>OPQ quantisations - Euclidean (Low Rank - 256 dim)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 256D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.66    16_232.13    16_262.79       1.0000     0.000000       146.48
Exhaustive (self)                                         30.66   169_450.56   169_481.21       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m16 (query)                             7_078.50     1_919.08     8_997.59       0.3792          NaN         2.79
Exhaustive-OPQ-m16 (self)                              7_078.50    20_400.18    27_478.69       0.2243          NaN         2.79
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m32 (query)                             7_349.16     4_307.37    11_656.52       0.5109          NaN         5.08
Exhaustive-OPQ-m32 (self)                              7_349.16    45_630.55    52_979.71       0.3554          NaN         5.08
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive-OPQ-m64 (query)                            11_048.54    12_097.89    23_146.42       0.6362          NaN         9.66
Exhaustive-OPQ-m64 (self)                             11_048.54   122_820.08   133_868.62       0.5351          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m16-np13 (query)                         9_087.28       526.91     9_614.19       0.6880          NaN         3.06
IVF-OPQ-nl273-m16-np16 (query)                         9_087.28       639.09     9_726.37       0.6880          NaN         3.06
IVF-OPQ-nl273-m16-np23 (query)                         9_087.28       930.61    10_017.89       0.6880          NaN         3.06
IVF-OPQ-nl273-m16 (self)                               9_087.28    10_595.84    19_683.12       0.6019          NaN         3.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m32-np13 (query)                         7_904.79       867.03     8_771.83       0.7965          NaN         5.35
IVF-OPQ-nl273-m32-np16 (query)                         7_904.79     1_038.98     8_943.77       0.7966          NaN         5.35
IVF-OPQ-nl273-m32-np23 (query)                         7_904.79     1_483.15     9_387.94       0.7966          NaN         5.35
IVF-OPQ-nl273-m32 (self)                               7_904.79    15_784.47    23_689.27       0.7263          NaN         5.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl273-m64-np13 (query)                        12_648.60     1_656.04    14_304.64       0.8951          NaN         9.93
IVF-OPQ-nl273-m64-np16 (query)                        12_648.60     2_105.64    14_754.24       0.8952          NaN         9.93
IVF-OPQ-nl273-m64-np23 (query)                        12_648.60     3_098.02    15_746.62       0.8952          NaN         9.93
IVF-OPQ-nl273-m64 (self)                              12_648.60    30_994.86    43_643.46       0.8498          NaN         9.93
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m16-np19 (query)                         8_990.03       874.83     9_864.86       0.6892          NaN         3.17
IVF-OPQ-nl387-m16-np27 (query)                         8_990.03     1_224.23    10_214.26       0.6892          NaN         3.17
IVF-OPQ-nl387-m16 (self)                               8_990.03    11_949.59    20_939.62       0.6035          NaN         3.17
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m32-np19 (query)                         9_296.95     1_197.48    10_494.43       0.7955          NaN         5.46
IVF-OPQ-nl387-m32-np27 (query)                         9_296.95     1_594.74    10_891.69       0.7955          NaN         5.46
IVF-OPQ-nl387-m32 (self)                               9_296.95    17_389.35    26_686.30       0.7274          NaN         5.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl387-m64-np19 (query)                        12_279.64     2_121.61    14_401.25       0.8963          NaN        10.04
IVF-OPQ-nl387-m64-np27 (query)                        12_279.64     3_023.99    15_303.63       0.8963          NaN        10.04
IVF-OPQ-nl387-m64 (self)                              12_279.64    32_125.60    44_405.24       0.8510          NaN        10.04
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m16-np23 (query)                         9_470.72       836.19    10_306.90       0.6894          NaN         3.33
IVF-OPQ-nl547-m16-np27 (query)                         9_470.72       953.69    10_424.41       0.6894          NaN         3.33
IVF-OPQ-nl547-m16-np33 (query)                         9_470.72     1_186.50    10_657.22       0.6894          NaN         3.33
IVF-OPQ-nl547-m16 (self)                               9_470.72    12_720.31    22_191.02       0.6073          NaN         3.33
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m32-np23 (query)                         9_773.76     1_306.76    11_080.51       0.7953          NaN         5.62
IVF-OPQ-nl547-m32-np27 (query)                         9_773.76     1_509.83    11_283.59       0.7953          NaN         5.62
IVF-OPQ-nl547-m32-np33 (query)                         9_773.76     1_844.53    11_618.29       0.7953          NaN         5.62
IVF-OPQ-nl547-m32 (self)                               9_773.76    19_806.43    29_580.19       0.7286          NaN         5.62
--------------------------------------------------------------------------------------------------------------------------------
IVF-OPQ-nl547-m64-np23 (query)                        13_731.66     2_390.25    16_121.91       0.8962          NaN        10.20
IVF-OPQ-nl547-m64-np27 (query)                        13_731.66     2_792.16    16_523.82       0.8962          NaN        10.20
IVF-OPQ-nl547-m64-np33 (query)                        13_731.66     3_409.66    17_141.32       0.8962          NaN        10.20
IVF-OPQ-nl547-m64 (self)                              13_731.66    35_360.32    49_091.98       0.8529          NaN        10.20
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
