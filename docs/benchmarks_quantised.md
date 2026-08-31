## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the vectors the index stores, trading recall for a
smaller memory footprint. In several cases the query gets faster too, because
integer kernels beat float ones. Below is how to run each example.

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

**HNSW on SQ8 codes:**

```bash
cargo run --example gridsearch_hnsw_quantised --release --features quantised -- --dim 128 --data cell
```

**SOAR-PQ and SOAR-OPQ:**

```bash
cargo run --example gridsearch_soar_pq  --release --features quantised -- --dim 512 --data embedding --n-samples 50000
cargo run --example gridsearch_soar_opq --release --features quantised -- --dim 512 --data embedding --n-samples 50000
```

As with the other benchmarks: index build, query against a 10% subsample with
noise added, and full self-kNN generation, plus the in-memory index size. The
PQ-family runs use `"correlated"`, `"lowrank"` and `"embedding"` at higher
dimensionality with fewer samples, since that is the regime these methods are
for.

**On the distance-ratio column.** A quantised index reports the codec's
*estimate* of a distance, not the distance. Feeding that straight into the ratio
conflates two errors and can push it below 1.0, which reads as "better than
optimal" and is nothing of the sort. Every ratio here is recomputed in `f32`
from the original vectors against the neighbours the index returned, so it
measures retrieval quality alone and is directly comparable to an unquantised
index's.

## Table of Contents

- [BF16 quantisation](#bf16-ivf-and-exhaustive)
- [SQ8 quantisation](#sq8-ivf-and-exhaustive)
- [HNSW on SQ8 codes](#hnsw-on-sq8-codes)
- [Product quantisation](#product-quantisation-exhaustive-and-ivf)
- [Optimised product quantisation](#optimised-product-quantisation-exhaustive-and-ivf)
- [SOAR-PQ and SOAR-OPQ](#soar-pq-and-soar-opq)

### BF16 (IVF and exhaustive)

Storage drops to `bf16`, which keeps the exponent range of `f32` and throws
away mantissa bits from roughly the third digit on. Distances are still computed
in `f32`, so the only loss is the stored value. Memory nearly halves for `f32`
input. Cosine loses more precision than Euclidean.

**Tunable parameters:**

- *Number of lists (nl)*: Number of k-means clusters. `sqrt(n)` is the usual
  heuristic when the structure is unknown.
- *Number of probes (np)*: Clusters probed at query time, typically
  `sqrt(nlist)` or up to 5% of `nlist`.

<details>
<summary><b>BF16 quantisations - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.41       623.53       634.94       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.41     6_078.75     6_090.16       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.56     1_166.54     1_179.10       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    12.56    11_708.82    11_721.38       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              289.40       117.31       406.70       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              289.40       134.95       424.35       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              289.40       183.01       472.41       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    289.40     1_347.67     1_637.07       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              537.15       123.28       660.44       0.9820          1.0001            1.0000         9.21
IVF-BF16-nl387-np27 (query)                              537.15       161.89       699.04       0.9828          1.0001            1.0000         9.21
IVF-BF16-nl387 (self)                                    537.15     1_213.30     1_750.45       0.9798          1.0001            1.0000         9.21
IVF-BF16-nl547-np23 (query)                            1_023.70       116.12     1_139.82       0.9773          1.0005            1.0000         9.23
IVF-BF16-nl547-np27 (query)                            1_023.70       131.17     1_154.87       0.9816          1.0002            1.0000         9.23
IVF-BF16-nl547-np33 (query)                            1_023.70       151.98     1_175.68       0.9828          1.0001            1.0000         9.23
IVF-BF16-nl547 (self)                                  1_023.70     1_147.19     2_170.89       0.9798          1.0001            1.0000         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.78       698.16       709.94       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.78     6_710.41     6_722.19       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   12.78     1_180.13     1_192.92       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    12.78    11_986.74    11_999.53       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              283.55        95.12       378.67       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              283.55       109.19       392.74       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              283.55       148.14       431.69       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    283.55     1_493.58     1_777.13       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              527.37       100.40       627.78       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              527.37       139.91       667.28       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    527.37     1_315.98     1_843.36       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                            1_008.04        93.80     1_101.84       0.8848          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                            1_008.04       105.88     1_113.93       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                            1_008.04       121.78     1_129.82       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                  1_008.04     1_221.07     2_229.12       0.8852          1.0073            1.0020         9.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.26       626.23       637.49       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.26     6_059.65     6_070.91       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.75     1_162.96     1_176.71       0.9345          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    13.75    12_261.01    12_274.76       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              410.34       130.14       540.48       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              410.34       128.99       539.33       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              410.34       168.87       579.22       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    410.34     1_358.82     1_769.16       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              621.83       120.74       742.57       0.9345          1.0018            1.0011         9.21
IVF-BF16-nl387-np27 (query)                              621.83       149.60       771.43       0.9345          1.0018            1.0011         9.21
IVF-BF16-nl387 (self)                                    621.83     1_100.28     1_722.11       0.9184          1.0030            1.0021         9.21
IVF-BF16-nl547-np23 (query)                            1_037.05       111.98     1_149.03       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_037.05       124.52     1_161.57       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_037.05       138.84     1_175.89       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547 (self)                                  1_037.05     1_049.79     2_086.83       0.9184          1.0030            1.0021         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.26       635.79       647.06       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.26     6_075.77     6_087.04       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.02     1_168.08     1_182.09       0.9515          1.0010            1.0004         9.16
Exhaustive-BF16 (self)                                    14.02    11_680.50    11_694.51       0.9405          1.0018            1.0010         9.16
IVF-BF16-nl273-np13 (query)                              288.51       110.03       398.54       0.9515          1.0010            1.0004         9.19
IVF-BF16-nl273-np16 (query)                              288.51       119.46       407.97       0.9515          1.0010            1.0004         9.19
IVF-BF16-nl273-np23 (query)                              288.51       164.10       452.61       0.9515          1.0010            1.0004         9.19
IVF-BF16-nl273 (self)                                    288.51     1_230.57     1_519.08       0.9405          1.0018            1.0010         9.19
IVF-BF16-nl387-np19 (query)                              539.90       112.15       652.05       0.9515          1.0010            1.0004         9.21
IVF-BF16-nl387-np27 (query)                              539.90       148.33       688.22       0.9515          1.0010            1.0004         9.21
IVF-BF16-nl387 (self)                                    539.90     1_073.59     1_613.49       0.9405          1.0018            1.0010         9.21
IVF-BF16-nl547-np23 (query)                            1_030.95       111.45     1_142.41       0.9515          1.0010            1.0004         9.23
IVF-BF16-nl547-np27 (query)                            1_030.95       118.68     1_149.63       0.9515          1.0010            1.0004         9.23
IVF-BF16-nl547-np33 (query)                            1_030.95       135.15     1_166.10       0.9515          1.0010            1.0004         9.23
IVF-BF16-nl547 (self)                                  1_030.95     1_034.18     2_065.13       0.9405          1.0018            1.0010         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.62     1_209.69     1_258.30       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.62    11_731.27    11_779.88       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   55.54     5_138.52     5_194.06       0.9716          1.0003            1.0000        36.62
Exhaustive-BF16 (self)                                    55.54    53_923.83    53_979.37       0.9674          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              591.08       300.93       892.00       0.9716          1.0003            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              591.08       338.89       929.96       0.9716          1.0003            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              591.08       472.99     1_064.06       0.9716          1.0003            1.0000        36.76
IVF-BF16-nl273 (self)                                    591.08     4_629.49     5_220.56       0.9674          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                            1_129.64       309.97     1_439.61       0.9716          1.0003            1.0000        36.81
IVF-BF16-nl387-np27 (query)                            1_129.64       406.98     1_536.61       0.9716          1.0003            1.0000        36.81
IVF-BF16-nl387 (self)                                  1_129.64     3_952.16     5_081.80       0.9674          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            2_409.10       316.53     2_725.63       0.9716          1.0003            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            2_409.10       327.64     2_736.75       0.9716          1.0003            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            2_409.10       373.88     2_782.98       0.9716          1.0003            1.0000        36.89
IVF-BF16-nl547 (self)                                  2_409.10     3_353.53     5_762.63       0.9674          1.0005            1.0000        36.89
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### SQ8 (IVF and exhaustive)

Uniform scalar quantisation to 8-bit codes: a per-dimension offset plus a
**single scale shared across every dimension**. That shared scale is the
load-bearing part. With `x_j = s * c_j + b_j`, a difference is
`x_j - y_j = s * (c_j - d_j)`, so the offsets cancel and the scale factors out.
The integer distance between two codes therefore preserves the exact ordering of
the float distance, which is what lets one kernel serve both index construction
and query. Per-dimension *scales* would break that; the offsets are free.

At 96 dimensions a vector goes from *96 x 32 bits = 384 bytes* to
*96 x 8 bits = 96 bytes*, a **4x reduction** plus the codebook. The integer
kernels also make the scan faster than the `f32` one rather than slower.

Ranking is exact whilst the code-space squared distance stays inside the float's
integer range: for `f32` that means `255^2 * dim <= 2^24`, so up to 258
dimensions. Past that, distances differing by one least-significant unit out of
millions can tie. `f64` covers any realistic dimensionality, and PCA or latent
spaces sit well inside the `f32` bound anyway.

**Tunable parameters:**

- *Drop ratio*: Fraction trimmed from **each** tail of every dimension before
  the range is fixed; values outside clamp to the end codes. With one shared
  scale the widest dimension sets the resolution for all of them, so a single
  heavy-tailed dimension would otherwise starve the rest. Exposed via
  `UniformQuantParams`.
- *Calibration sample rows*: Rows sampled to estimate the tails. Auto-picks,
  capped at the dataset size.
- *Number of lists (nl)*: IVF only. Number of k-means clusters, `sqrt(n)` as a
  default.
- *Number of probes (np)*: IVF only. Typically `sqrt(nlist)` or up to 5% of
  `nlist`.

#### With 32 dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        12.10       649.33       661.42       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.10     6_392.09     6_404.19       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.66       997.23     1_014.90       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     17.66    10_364.20    10_381.86       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               310.40        67.94       378.34       0.9244          1.0020            1.0009         6.33
IVF-SQ8-nl273-np16 (query)                               310.40        76.94       387.34       0.9258          1.0018            1.0009         6.33
IVF-SQ8-nl273-np23 (query)                               310.40        97.54       407.94       0.9260          1.0018            1.0009         6.33
IVF-SQ8-nl273 (self)                                     310.40       949.46     1_259.86       0.9253          1.0018            1.0009         6.33
IVF-SQ8-nl387-np19 (query)                               555.03        68.46       623.49       0.9243          1.0019            1.0009         6.35
IVF-SQ8-nl387-np27 (query)                               555.03        86.67       641.70       0.9248          1.0018            1.0009         6.35
IVF-SQ8-nl387 (self)                                     555.03       845.17     1_400.20       0.9252          1.0018            1.0009         6.35
IVF-SQ8-nl547-np23 (query)                             1_066.01        66.23     1_132.24       0.9215          1.0022            1.0010         6.37
IVF-SQ8-nl547-np27 (query)                             1_066.01        74.61     1_140.62       0.9244          1.0019            1.0009         6.37
IVF-SQ8-nl547-np33 (query)                             1_066.01        83.81     1_149.82       0.9251          1.0018            1.0009         6.37
IVF-SQ8-nl547 (self)                                   1_066.01       828.72     1_894.74       0.9252          1.0018            1.0009         6.37
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.93       718.11       730.04       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.93     7_012.09     7_024.02       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    21.99       974.24       996.23       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     21.99    11_184.49    11_206.48       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               318.19        71.47       389.66       0.7391          1.0362            1.0153         6.33
IVF-SQ8-nl273-np16 (query)                               318.19        78.75       396.93       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273-np23 (query)                               318.19       102.94       421.13       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273 (self)                                     318.19     1_035.59     1_353.77       0.7378          1.0362            1.0152         6.33
IVF-SQ8-nl387-np19 (query)                               591.22        75.41       666.63       0.7378          1.0360            1.0155         6.35
IVF-SQ8-nl387-np27 (query)                               591.22        93.07       684.29       0.7379          1.0360            1.0155         6.35
IVF-SQ8-nl387 (self)                                     591.22       956.67     1_547.89       0.7375          1.0362            1.0157         6.35
IVF-SQ8-nl547-np23 (query)                             1_121.07        72.18     1_193.26       0.7365          1.0365            1.0165         6.37
IVF-SQ8-nl547-np27 (query)                             1_121.07        75.62     1_196.69       0.7370          1.0364            1.0163         6.37
IVF-SQ8-nl547-np33 (query)                             1_121.07        91.23     1_212.30       0.7369          1.0364            1.0162         6.37
IVF-SQ8-nl547 (self)                                   1_121.07       925.11     2_046.18       0.7360          1.0365            1.0159         6.37
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        12.66       709.24       721.90       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.66     6_535.91     6_548.57       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.40       970.96       990.36       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     19.40     9_733.15     9_752.55       0.8120          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               304.67        63.46       368.13       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np16 (query)                               304.67        68.47       373.15       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np23 (query)                               304.67        85.00       389.67       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273 (self)                                     304.67       842.75     1_147.42       0.8121          1.0174            1.0155         6.33
IVF-SQ8-nl387-np19 (query)                               542.66        65.25       607.91       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387-np27 (query)                               542.66        78.21       620.87       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387 (self)                                     542.66       760.99     1_303.66       0.8119          1.0175            1.0155         6.35
IVF-SQ8-nl547-np23 (query)                             1_037.65        63.54     1_101.19       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np27 (query)                             1_037.65        68.72     1_106.37       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np33 (query)                             1_037.65        75.82     1_113.47       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547 (self)                                   1_037.65       764.09     1_801.74       0.8118          1.0175            1.0155         6.37
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.98       622.76       633.74       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         10.98     6_076.82     6_087.79       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.73       974.82       991.55       0.7864          1.0270            1.0249         5.15
Exhaustive-SQ8 (self)                                     16.73     9_804.51     9_821.23       0.7862          1.0286            1.0263         5.15
IVF-SQ8-nl273-np13 (query)                               300.15        60.45       360.61       0.7866          1.0270            1.0249         6.33
IVF-SQ8-nl273-np16 (query)                               300.15        66.74       366.89       0.7866          1.0270            1.0249         6.33
IVF-SQ8-nl273-np23 (query)                               300.15        85.22       385.37       0.7866          1.0270            1.0249         6.33
IVF-SQ8-nl273 (self)                                     300.15       864.34     1_164.49       0.7861          1.0286            1.0262         6.33
IVF-SQ8-nl387-np19 (query)                               547.49        62.68       610.18       0.7867          1.0269            1.0247         6.35
IVF-SQ8-nl387-np27 (query)                               547.49        76.90       624.39       0.7867          1.0269            1.0247         6.35
IVF-SQ8-nl387 (self)                                     547.49       758.15     1_305.64       0.7864          1.0286            1.0263         6.35
IVF-SQ8-nl547-np23 (query)                             1_035.47        63.44     1_098.91       0.7856          1.0271            1.0250         6.37
IVF-SQ8-nl547-np27 (query)                             1_035.47        67.52     1_102.99       0.7856          1.0271            1.0250         6.37
IVF-SQ8-nl547-np33 (query)                             1_035.47        77.80     1_113.27       0.7856          1.0271            1.0250         6.37
IVF-SQ8-nl547 (self)                                   1_035.47       735.56     1_771.03       0.7865          1.0286            1.0262         6.37
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### More dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.35     1_199.89     1_249.24       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.35    11_775.01    11_824.36       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    80.65     1_168.77     1_249.42       0.8843          1.0056            1.0047        18.88
Exhaustive-SQ8 (self)                                     80.65    12_023.96    12_104.61       0.8898          1.0067            1.0055        18.88
IVF-SQ8-nl273-np13 (query)                               624.30        80.51       704.81       0.8827          1.0057            1.0048        20.16
IVF-SQ8-nl273-np16 (query)                               624.30        84.04       708.34       0.8827          1.0057            1.0048        20.16
IVF-SQ8-nl273-np23 (query)                               624.30       112.14       736.44       0.8827          1.0057            1.0048        20.16
IVF-SQ8-nl273 (self)                                     624.30       919.20     1_543.50       0.8898          1.0067            1.0055        20.16
IVF-SQ8-nl387-np19 (query)                             1_139.02        87.01     1_226.03       0.8840          1.0056            1.0047        20.22
IVF-SQ8-nl387-np27 (query)                             1_139.02       104.77     1_243.78       0.8840          1.0056            1.0047        20.22
IVF-SQ8-nl387 (self)                                   1_139.02       843.93     1_982.95       0.8898          1.0067            1.0055        20.22
IVF-SQ8-nl547-np23 (query)                             2_354.10        91.31     2_445.41       0.8836          1.0056            1.0048        20.30
IVF-SQ8-nl547-np27 (query)                             2_354.10       102.20     2_456.31       0.8836          1.0056            1.0048        20.30
IVF-SQ8-nl547-np33 (query)                             2_354.10       106.67     2_460.77       0.8836          1.0056            1.0048        20.30
IVF-SQ8-nl547 (self)                                   2_354.10       842.90     3_197.00       0.8898          1.0067            1.0054        20.30
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### HNSW on SQ8 codes

An HNSW built **and** searched entirely on the uniform 8-bit codes described
above, inspired by [pyglass](https://github.com/zilliztech/pyglass). Because the
shared scale makes the integer code distance order-preserving, one kernel serves
graph construction and query alike: the graph never sees a float. Memory drops
4x against a plain HNSW and the build gets faster, since the construction search
is doing integer arithmetic.

The grid runs the full-precision HNSW at matched `(M, ef_construction,
ef_search)` alongside it, plus an exhaustive scan over the same codec. The
exhaustive-SQ8 row is the ceiling the graph rows work against: whatever they
lose up to it is the codec, whatever they lose beyond it is the graph.

**Tunable parameters:**

- *M (m)*: Connections per node per layer.
- *EF construction (ef)*: Candidate budget while wiring the graph.
- *EF search (s)*: Candidate budget at query time.
- *Drop ratio*: Tail trim on the quantiser calibration, swept separately at a
  fixed `(M=16, ef=200)`. `0.0` is the pyglass default; the non-zero settings
  are what a shared scale wants when a handful of points sit far out in one
  dimension. Sweep runs `0.0`, `1e-3` and `1e-2`.

Self is queried with `s=100`.

<details>
<summary><b>HNSW-SQ8U - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.15       661.55       672.70       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.15     6_371.90     6_383.05       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.88       982.11       999.99       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               780.55        56.05       836.60       0.9296          1.0156            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              780.55       101.18       881.73       0.9640          1.0087            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              780.55       190.46       971.01       0.9829          1.0051            1.0000        38.52
HNSW-M16-ef100 (self)                                    780.55       997.84     1_778.39       0.9643          1.0081            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_507.59        58.86     1_566.45       0.9586          1.0078            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_507.59       108.87     1_616.46       0.9830          1.0037            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_507.59       204.67     1_712.26       0.9921          1.0021            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_507.59     1_026.02     2_533.61       0.9835          1.0047            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_635.31        63.37     1_698.68       0.9695          1.0123            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_635.31       114.50     1_749.81       0.9881          1.0074            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_635.31       210.82     1_846.13       0.9954          1.0013            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_635.31     1_154.72     2_790.03       0.9881          1.0060            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_720.06        69.13     1_789.19       0.9740          1.0040            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_720.06       119.49     1_839.54       0.9900          1.0021            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_720.06       218.81     1_938.86       0.9963          1.0003            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_720.06     1_179.95     2_900.01       0.9904          1.0011            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          679.78        35.30       715.07       0.8767          1.0175            1.0033        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         679.78        64.61       744.39       0.9019          1.0085            1.0020        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         679.78       118.81       798.59       0.9148          1.0055            1.0014        26.89
HNSW-SQ8U-M16-ef100 (self)                               679.78       624.46     1_304.24       0.9017          1.0103            1.0020        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_385.66        50.59     1_436.26       0.8955          1.0455            1.0021        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_385.66        68.49     1_454.16       0.9128          1.0242            1.0014        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_385.66       127.68     1_513.35       0.9207          1.0044            1.0011        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_385.66       652.98     2_038.65       0.9122          1.0279            1.0014        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_445.02        40.38     1_485.40       0.9069          1.0097            1.0017        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_445.02        75.62     1_520.65       0.9186          1.0067            1.0012        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_445.02       136.03     1_581.05       0.9230          1.0034            1.0010        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_445.02       714.15     2_159.17       0.9179          1.0058            1.0012        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_527.05        42.26     1_569.30       0.9076          1.0460            1.0016        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_527.05        78.63     1_605.68       0.9188          1.0260            1.0012        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_527.05       142.77     1_669.81       0.9235          1.0039            1.0010        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_527.05       742.45     2_269.50       0.9183          1.0291            1.0012        45.20
HNSW-SQ8U-drop0 (query)                                1_340.56        66.10     1_406.67       0.8936          1.0184            1.0024        26.89
HNSW-SQ8U-drop0.001 (query)                            1_337.08        66.29     1_403.37       0.9144          1.0083            1.0014        26.89
HNSW-SQ8U-drop0.01 (query)                             1_345.20        65.39     1_410.59       0.8980          1.0117            1.0018        26.89
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.87       693.59       705.46       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.87     6_755.36     6_767.23       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    19.81       895.28       915.09       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               840.42        57.91       898.33       0.9350          1.0149            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              840.42       107.69       948.11       0.9687          1.0075            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              840.42       202.20     1_042.62       0.9872          1.0035            1.0000        39.09
HNSW-M16-ef100 (self)                                    840.42     1_023.94     1_864.36       0.9690          1.0087            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_615.17        60.27     1_675.45       0.9637          1.0119            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_615.17       114.49     1_729.67       0.9871          1.0042            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_615.17       214.06     1_829.23       0.9947          1.0014            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_615.17     1_093.16     2_708.34       0.9868          1.0047            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_759.34        69.28     1_828.62       0.9729          1.0181            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_759.34       132.88     1_892.22       0.9907          1.0140            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_759.34       226.65     1_985.99       0.9969          1.0003            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_759.34     1_246.53     3_005.87       0.9907          1.0057            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_806.93        70.03     1_876.96       0.9759          1.0116            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_806.93       138.86     1_945.80       0.9916          1.0049            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_806.93       240.49     2_047.43       0.9973          1.0005            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_806.93     1_289.08     3_096.02       0.9917          1.0032            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          713.70        35.22       748.92       0.6847          1.0579            1.0289        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         713.70        66.13       779.83       0.7085          1.0504            1.0238        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         713.70       121.87       835.57       0.7227          1.0447            1.0208        26.89
HNSW-SQ8U-M16-ef100 (self)                               713.70       614.82     1_328.52       0.7086          1.0492            1.0238        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_372.90        37.34     1_410.23       0.7057          1.0865            1.0235        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_372.90        71.18     1_444.08       0.7231          1.0521            1.0200        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_372.90       129.11     1_502.01       0.7313          1.0417            1.0183        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_372.90       697.55     2_070.45       0.7230          1.0518            1.0199        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_476.85        41.83     1_518.68       0.7178          1.0473            1.0209        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_476.85        75.65     1_552.50       0.7304          1.0386            1.0184        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_476.85       136.63     1_613.48       0.7354          1.0372            1.0173        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_476.85       730.05     2_206.90       0.7296          1.0403            1.0182        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_532.64        53.47     1_586.11       0.7217          1.0570            1.0196        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_532.64        79.00     1_611.64       0.7324          1.0392            1.0176        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_532.64       146.62     1_679.26       0.7365          1.0364            1.0169        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_532.64       783.70     2_316.35       0.7318          1.0395            1.0174        45.20
HNSW-SQ8U-drop0 (query)                                1_380.71        67.01     1_447.72       0.6642          1.0741            1.0306        26.89
HNSW-SQ8U-drop0.001 (query)                            1_414.13        66.91     1_481.04       0.7235          1.0453            1.0200        26.89
HNSW-SQ8U-drop0.01 (query)                             1_359.80        66.01     1_425.82       0.6894          1.0535            1.0260        26.89
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.30       658.64       669.94       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.30     6_568.93     6_580.24       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.45       976.35       995.80       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               830.90        58.14       889.04       0.9475         35.7864            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              830.90       103.25       934.15       0.9650         16.3987            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              830.90       192.92     1_023.81       0.9971          1.0561            1.0000        38.52
HNSW-M16-ef100 (self)                                    830.90     1_003.21     1_834.10       0.9661         17.1377            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_417.70        59.52     1_477.22       0.9616         70.5770            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_417.70       104.17     1_521.87       0.9649         66.7538            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_417.70       195.49     1_613.19       0.9985          1.4798            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_417.70     1_013.27     2_430.97       0.9633         74.5803            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_487.59        62.62     1_550.21       0.9950          1.4525            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_487.59       106.69     1_594.28       0.9957          1.0022            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_487.59       189.05     1_676.64       0.9958          1.0022            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_487.59     1_025.65     2_513.24       0.9955          1.0025            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_492.82        61.77     1_554.59       0.9986          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_492.82       105.83     1_598.65       0.9990          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_492.82       187.63     1_680.44       0.9991          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_492.82     1_019.41     2_512.23       0.9990          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          703.55        36.90       740.45       0.8134          1.4375            1.0149        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         703.55        66.22       769.76       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         703.55       118.16       821.71       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef100 (self)                               703.55       606.42     1_309.97       0.8117          1.0787            1.0156        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_216.59        36.29     1_252.88       0.8141          1.0166            1.0148        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_216.59        66.23     1_282.82       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_216.59       117.86     1_334.45       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_216.59       615.61     1_832.19       0.8119          1.0175            1.0155        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_310.59        38.85     1_349.44       0.8143          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_310.59        74.72     1_385.31       0.8145          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_310.59       122.01     1_432.60       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_310.59       652.06     1_962.65       0.8119          1.0175            1.0155        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_406.67        39.55     1_446.22       0.8144          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_406.67        70.26     1_476.93       0.8145          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_406.67       124.33     1_530.99       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_406.67       664.41     2_071.08       0.8119          1.0175            1.0155        45.20
HNSW-SQ8U-drop0 (query)                                1_213.58        76.55     1_290.13       0.8062          2.3687            1.0158        26.89
HNSW-SQ8U-drop0.001 (query)                            1_213.45        61.61     1_275.05       0.8143          1.3210            1.0148        26.89
HNSW-SQ8U-drop0.01 (query)                             1_219.44        63.42     1_282.85       0.8049          1.0195            1.0163        26.89
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.03       658.57       669.61       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.03     6_626.60     6_637.63       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.61       988.22     1_006.83       0.7864          1.0270            1.0249         5.15
HNSW-M16-ef100-s50 (query)                               877.06        62.26       939.32       0.9975          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              877.06       111.42       988.48       0.9993          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              877.06       202.76     1_079.82       0.9995          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    877.06     1_088.98     1_966.04       0.9993          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_542.03        62.53     1_604.56       0.9978          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_542.03       115.59     1_657.62       0.9994          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_542.03       206.22     1_748.25       0.9995          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_542.03     1_111.68     2_653.70       0.9994          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_652.16        70.59     1_722.75       0.9986          1.0136            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_652.16       123.63     1_775.79       0.9993          1.0135            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_652.16       222.43     1_874.59       0.9993          1.0135            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_652.16     1_214.79     2_866.95       0.9993          1.0192            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_704.74        73.90     1_778.64       0.9990          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_704.74       132.46     1_837.20       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_704.74       230.52     1_935.26       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_704.74     1_269.26     2_974.00       0.9994          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          771.68        37.32       809.00       0.7858          1.1252            1.0250        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         771.68        71.34       843.01       0.7864          1.0270            1.0249        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         771.68       136.41       908.08       0.7864          1.0270            1.0249        26.89
HNSW-SQ8U-M16-ef100 (self)                               771.68       671.93     1_443.61       0.7862          1.0287            1.0263        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_411.12        37.92     1_449.04       0.7861          1.0271            1.0250        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_411.12        74.83     1_485.95       0.7864          1.0270            1.0249        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_411.12       132.23     1_543.34       0.7864          1.0270            1.0249        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_411.12       689.31     2_100.42       0.7862          1.0286            1.0263        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_459.07        42.42     1_501.49       0.7864          1.0270            1.0249        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_459.07        79.36     1_538.43       0.7864          1.0270            1.0249        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_459.07       142.91     1_601.98       0.7864          1.0270            1.0249        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_459.07       762.03     2_221.10       0.7862          1.0286            1.0263        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_538.64        45.42     1_584.07       0.7863          1.0270            1.0249        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_538.64        80.99     1_619.64       0.7864          1.0270            1.0249        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_538.64       146.56     1_685.21       0.7864          1.0270            1.0249        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_538.64       768.40     2_307.04       0.7862          1.0286            1.0263        45.20
HNSW-SQ8U-drop0 (query)                                1_401.43        70.74     1_472.18       0.7775          1.0292            1.0270        26.89
HNSW-SQ8U-drop0.001 (query)                            1_430.91        70.73     1_501.64       0.7864          1.0270            1.0249        26.89
HNSW-SQ8U-drop0.01 (query)                             1_436.44        78.77     1_515.21       0.7791          1.0298            1.0267        26.89
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        53.85     1_400.09     1_453.94       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         53.85    13_331.99    13_385.84       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    78.17     1_219.24     1_297.41       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_509.63       107.02     1_616.65       0.9925          1.0433            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_509.63       191.71     1_701.34       0.9953          1.0217            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_509.63       322.74     1_832.37       0.9973          1.0111            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_509.63     1_620.37     3_130.00       0.9960          1.0167            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_852.87       105.48     2_958.34       0.9968          1.0220            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_852.87       187.47     3_040.34       0.9981          1.0130            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_852.87       300.52     3_153.39       0.9993          1.0043            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_852.87     1_821.86     4_674.73       0.9979          1.0125            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_805.53       103.29     2_908.82       0.9979          1.0109            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_805.53       177.26     2_982.80       0.9988          1.0065            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_805.53       310.33     3_115.86       0.9995          1.0022            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_805.53     1_846.52     4_652.05       0.9990          1.0050            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             3_203.52       115.20     3_318.72       0.9987          1.0072            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            3_203.52       195.74     3_399.26       0.9992          1.0036            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            3_203.52       338.96     3_542.48       0.9997          1.0009            1.0000       111.73
HNSW-M32-ef200 (self)                                  3_203.52     1_891.49     5_095.01       0.9993          1.0035            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          861.69        36.58       898.27       0.9278          1.0420            1.0038        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         861.69        68.32       930.01       0.9306          1.0240            1.0038        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         861.69       123.22       984.91       0.9323          1.0133            1.0037        40.63
HNSW-SQ8U-M16-ef100 (self)                               861.69       657.57     1_519.26       0.9302          1.0250            1.0038        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_456.41        39.53     1_495.94       0.9314          1.0262            1.0037        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_456.41        72.83     1_529.23       0.9330          1.0151            1.0036        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_456.41       128.32     1_584.73       0.9334          1.0116            1.0036        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_456.41       740.63     2_197.04       0.9319          1.0215            1.0037        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_822.85        57.70     1_880.55       0.9323          1.0193            1.0036        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_822.85        80.69     1_903.54       0.9330          1.0145            1.0036        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_822.85       145.67     1_968.52       0.9335          1.0111            1.0036        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_822.85       739.85     2_562.70       0.9329          1.0132            1.0036        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_637.56        44.75     1_682.31       0.9329          1.0148            1.0036        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_637.56        76.39     1_713.95       0.9334          1.0121            1.0036        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_637.56       129.44     1_766.99       0.9338          1.0088            1.0036        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_637.56       819.13     2_456.68       0.9332          1.0118            1.0036        58.94
HNSW-SQ8U-drop0 (query)                                1_533.55        71.71     1_605.26       0.8643          1.0473            1.0220        40.63
HNSW-SQ8U-drop0.001 (query)                            1_466.10        69.71     1_535.81       0.9324          1.0174            1.0037        40.63
HNSW-SQ8U-drop0.01 (query)                             1_451.92        70.38     1_522.30       0.9323          1.0385            1.0020        40.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        54.11     1_447.07     1_501.18       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         54.11    13_576.45    13_630.56       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    93.03     1_256.43     1_349.46       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_383.59        99.35     1_482.94       0.9939          1.1262            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_383.59       259.13     1_642.72       0.9973          1.0441            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_383.59       303.66     1_687.25       0.9986          1.0095            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_383.59     1_440.07     2_823.66       0.9968          1.0425            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_494.02        86.77     2_580.80       0.9931          1.1592            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_494.02       146.37     2_640.39       0.9959          1.0891            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_494.02       264.30     2_758.33       0.9984          1.0232            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_494.02     1_467.93     3_961.95       0.9963          1.0745            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_660.71        89.01     2_749.72       0.9980          1.0335            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_660.71       162.87     2_823.58       0.9989          1.0130            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_660.71       277.25     2_937.96       0.9996          1.0040            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_660.71     1_545.68     4_206.39       0.9987          1.0190            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_702.55        92.33     2_794.88       0.9975          1.0658            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_702.55       153.48     2_856.03       0.9985          1.0345            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_702.55       274.92     2_977.47       0.9994          1.0133            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_702.55     1_522.89     4_225.44       0.9984          1.0350            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          827.14        37.49       864.62       0.6644          1.4141            1.1635        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         827.14        64.10       891.24       0.6658          1.3843            1.1624        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         827.14       118.07       945.20       0.6664          1.3684            1.1621        40.63
HNSW-SQ8U-M16-ef100 (self)                               827.14       619.66     1_446.79       0.6651          1.3845            1.1632        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_568.56        39.95     1_608.51       0.6642          1.4160            1.1633        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_568.56        68.41     1_636.96       0.6658          1.3864            1.1622        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_568.56       121.71     1_690.26       0.6668          1.3661            1.1616        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_568.56       662.08     2_230.64       0.6656          1.3817            1.1626        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_592.74        42.58     1_635.32       0.6653          1.4171            1.1624        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_592.74        70.77     1_663.51       0.6665          1.3767            1.1619        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_592.74       130.78     1_723.52       0.6672          1.3559            1.1614        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_592.74       740.69     2_333.43       0.6658          1.3945            1.1625        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_638.42        64.19     1_702.61       0.6658          1.3867            1.1622        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_638.42        73.47     1_711.89       0.6671          1.3558            1.1614        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_638.42       132.23     1_770.66       0.6675          1.3485            1.1613        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_638.42       747.89     2_386.31       0.6666          1.3642            1.1618        58.94
HNSW-SQ8U-drop0 (query)                                1_590.66        82.39     1_673.05       0.6191          1.5409            1.2372        40.63
HNSW-SQ8U-drop0.001 (query)                            1_687.15        71.98     1_759.12       0.6652          1.3950            1.1629        40.63
HNSW-SQ8U-drop0.01 (query)                             1_791.71        77.45     1_869.17       0.6795          1.3286            1.1501        40.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Product quantisations

PQ and OPQ compress far harder than BF16 or SQ8: each vector is split into
subvectors and every subvector is replaced by a codebook index. These runs use
256, 512 and 768 dimensions at 50k samples, the regime these methods exist for.
Three synthetic types of increasing difficulty:

- `"correlated"`: subspace-clustered activation patterns.
- `"lowrank"`: embedded from a lower-dimensional manifold.
- `"embedding"`: foundation-model cell embeddings, which combine a shared
  anisotropy cone, a few rogue high-variance axes and per-cell-type oriented
  subspaces. Between them those break sign binarisation, axis-aligned subvector
  splits and any single global rotation.

#### Product quantisation (Exhaustive and IVF)

Harsh compression. At 192 dimensions with `m = 32` a vector goes from
*192 x f32 = 768 bytes* to *32 x u8 = 32 bytes*, a **24x reduction** plus the
codebook. Worth it when good enough is good enough and memory is the binding
constraint.

**Tunable parameters:**

- *Number of subvectors (m)*: How many subvectors to split each vector into.
  The dimensionality must be divisible by `m`. Each subvector becomes one `u8`,
  so `m` sets the compressed size directly.
- *Number of lists (nl)*: IVF only. Number of k-means clusters, `sqrt(n)` as a
  default.
- *Number of probes (np)*: IVF only. Typically `sqrt(nlist)` or up to 5% of
  `nlist`. The self queries default to `sqrt(nlist)`.

The self queries run against the compressed vectors held in the index. If you
want a high-quality kNN graph out of one of these, re-supply the uncompressed
data, at the obvious memory cost.

#### Why the IVF variant beats the exhaustive one

PQ's error is driven by the **variance** of whatever it is asked to encode:
lower variance lets 256 centroids per subspace tile the space more densely.
IVF-PQ clusters first and encodes **residuals** against the cell centroid, which
are small and tightly distributed. Exhaustive-PQ encodes raw vectors, so the
whole dataset's diversity competes for the same 256 centroids per subspace.

Clustering creates locality, and locality is what PQ needs. Mean-centring or a
rotation (OPQ) does not: it moves the data without reducing its intrinsic
spread. The clustering step is not optional for high-recall PQ search.

##### Correlated data

Let's start with correlated data.

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.15       778.60       812.75       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.15     2_513.44     2_547.59       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                676.22       674.74     1_350.96       0.2580          1.1826            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 676.22     2_279.96     2_956.18       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_160.21     1_545.30     2_705.51       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_160.21     5_183.92     6_344.13       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              1_810.10     3_708.86     5_518.96       0.3610          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               1_810.10    12_266.10    14_076.20       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_472.59       206.46     1_679.05       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_472.59       332.44     1_805.03       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_472.59       446.29     1_918.89       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16 (self)                                1_472.59     1_419.89     2_892.48       0.3041          1.1282            1.1332         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_906.53       366.38     2_272.91       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_906.53       565.89     2_472.42       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_906.53       766.45     2_672.98       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32 (self)                                1_906.53     2_558.52     4_465.05       0.4068          1.0804            1.0800         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_392.17       637.17     3_029.33       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_392.17     1_011.93     3_404.10       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_392.17     1_358.75     3_750.91       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64 (self)                                2_392.17     4_486.12     6_878.28       0.6338          1.0271            1.0243         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_128.16       301.06     1_429.23       0.3870          1.0887            1.0895         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_128.16       364.20     1_492.36       0.3869          1.0887            1.0895         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_128.16       529.79     1_657.96       0.3869          1.0887            1.0895         1.23
IVF-PQ-nl223-m16 (self)                                1_128.16     1_747.94     2_876.11       0.3098          1.1230            1.1272         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_556.35       529.36     2_085.71       0.4975          1.0564            1.0516         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_556.35       641.94     2_198.28       0.4975          1.0564            1.0517         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_556.35       930.28     2_486.63       0.4975          1.0564            1.0517         2.00
IVF-PQ-nl223-m32 (self)                                1_556.35     3_043.79     4_600.13       0.4146          1.0780            1.0755         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_097.28       948.91     3_046.19       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_097.28     1_116.33     3_213.61       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_097.28     1_609.67     3_706.95       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64 (self)                                2_097.28     5_356.37     7_453.65       0.6386          1.0273            1.0235         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_366.36       386.80     1_753.15       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_366.36       420.03     1_786.39       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_366.36       619.83     1_986.19       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16 (self)                                1_366.36     2_122.45     3_488.80       0.3156          1.1188            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_945.18       734.18     2_679.35       0.5114          1.0520            1.0487         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_945.18       814.75     2_759.92       0.5114          1.0519            1.0487         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_945.18     1_147.71     3_092.89       0.5114          1.0519            1.0487         2.09
IVF-PQ-nl316-m32 (self)                                1_945.18     3_711.01     5_656.19       0.4236          1.0742            1.0728         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_520.82     1_205.49     3_726.31       0.7073          1.0175            1.0146         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_520.82     1_329.61     3_850.43       0.7073          1.0174            1.0146         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_520.82     2_028.74     4_549.56       0.7073          1.0174            1.0146         3.61
IVF-PQ-nl316-m64 (self)                                2_520.82     6_387.62     8_908.43       0.6490          1.0248            1.0221         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.93     1_261.42     1_329.34       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.93     4_637.75     4_705.68       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_100.12       726.85     1_826.97       0.2444          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                               1_100.12     2_309.91     3_410.02       0.2278          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_253.01     1_523.23     2_776.24       0.2648          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_253.01     5_086.88     6_339.90       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_115.59     3_725.81     5_841.41       0.2955          1.0990            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_115.59    12_599.09    14_714.68       0.2627          1.1103            1.1142         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_600.23       293.73     2_893.96       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_600.23       462.22     3_062.45       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_600.23       637.22     3_237.45       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16 (self)                                2_600.23     2_066.16     4_666.39       0.2624          1.1080            1.1146         1.57
IVF-PQ-nl158-m32-np7 (query)                           3_050.73       419.29     3_470.02       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np12 (query)                          3_050.73       662.97     3_713.71       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np17 (query)                          3_050.73       902.45     3_953.19       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32 (self)                                3_050.73     3_073.74     6_124.48       0.2913          1.0913            1.0953         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_980.11       758.09     4_738.20       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_980.11     1_165.17     5_145.28       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_980.11     1_660.85     5_640.96       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64 (self)                                3_980.11     5_824.03     9_804.14       0.3902          1.0583            1.0572         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_821.61       420.18     2_241.79       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_821.61       538.55     2_360.16       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_821.61       783.67     2_605.27       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16 (self)                                1_821.61     2_548.82     4_370.43       0.2659          1.1044            1.1096         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_076.65       595.66     2_672.31       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_076.65       716.49     2_793.14       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_076.65     1_034.12     3_110.77       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                2_076.65     3_466.71     5_543.36       0.2945          1.0892            1.0917         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_199.03     1_125.74     4_324.77       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_199.03     1_363.12     4_562.15       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_199.03     1_963.77     5_162.80       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64 (self)                                3_199.03     6_526.67     9_725.70       0.3958          1.0576            1.0552         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_132.15       575.08     2_707.24       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_132.15       633.83     2_765.99       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_132.15       886.29     3_018.44       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16 (self)                                2_132.15     2_943.63     5_075.78       0.2710          1.0999            1.1061         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_471.04       793.68     3_264.72       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_471.04       836.22     3_307.26       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_471.04     1_194.10     3_665.14       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32 (self)                                2_471.04     3_970.96     6_442.00       0.2993          1.0859            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_335.09     1_351.06     4_686.15       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_335.09     1_518.42     4_853.51       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_335.09     2_153.90     5_488.99       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64 (self)                                3_335.09     7_126.54    10_461.63       0.4046          1.0543            1.0533         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.99     1_803.66     1_906.66       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.99     5_963.15     6_066.14       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_190.26       696.31     1_886.57       0.2346          1.1094            1.0999         1.51
Exhaustive-PQ-m16 (self)                               1_190.26     2_245.28     3_435.54       0.2205          1.1179            1.1047         1.51
Exhaustive-PQ-m32 (query)                              1_585.39     1_564.39     3_149.78       0.2566          1.0942            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_585.39     5_091.04     6_676.43       0.2390          1.1012            1.1020         2.28
Exhaustive-PQ-m64 (query)                              2_492.30     3_604.96     6_097.26       0.2774          1.0854            1.0909         3.80
Exhaustive-PQ-m64 (self)                               2_492.30    11_939.14    14_431.44       0.2516          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_340.83     7_938.14    12_278.97       0.3160          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_340.83    26_172.65    30_513.48       0.2754          1.0816            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_254.84       362.48     3_617.32       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_254.84       545.87     3_800.71       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_254.84       757.78     4_012.62       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16 (self)                                3_254.84     2_508.96     5_763.81       0.2511          1.0937            1.1010         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_885.18       536.15     4_421.33       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_885.18       850.90     4_736.08       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_885.18     1_143.37     5_028.55       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32 (self)                                3_885.18     3_800.83     7_686.01       0.2628          1.0854            1.0912         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_820.07       853.10     5_673.17       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_820.07     1_314.78     6_134.85       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_820.07     1_783.06     6_603.13       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64 (self)                                4_820.07     5_862.88    10_682.95       0.3104          1.0661            1.0674         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_434.41     1_578.88     8_013.29       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_434.41     2_524.36     8_958.77       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_434.41     3_413.73     9_848.15       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128 (self)                               6_434.41    11_345.55    17_779.96       0.4636          1.0342            1.0319         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_383.83       505.12     2_888.94       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_383.83       612.26     2_996.09       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_383.83       883.10     3_266.93       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16 (self)                                2_383.83     2_913.95     5_297.77       0.2553          1.0892            1.0954         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_808.81       742.77     3_551.58       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_808.81       904.39     3_713.20       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_808.81     1_313.07     4_121.88       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32 (self)                                2_808.81     4_335.66     7_144.47       0.2675          1.0815            1.0861         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_801.74     1_166.64     4_968.38       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_801.74     1_447.22     5_248.96       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_801.74     2_081.12     5_882.86       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                3_801.74     6_882.98    10_684.71       0.3132          1.0650            1.0653         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_671.49     2_305.86     7_977.35       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_671.49     2_856.45     8_527.95       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_671.49     4_162.96     9_834.46       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128 (self)                               5_671.49    13_889.34    19_560.83       0.4701          1.0335            1.0307         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_777.59       664.81     3_442.40       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_777.59       729.44     3_507.03       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_777.59     1_039.48     3_817.07       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16 (self)                                2_777.59     3_452.66     6_230.25       0.2597          1.0853            1.0916         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_238.49       995.03     4_233.52       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_238.49     1_112.59     4_351.08       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_238.49     1_598.84     4_837.33       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32 (self)                                3_238.49     5_232.56     8_471.04       0.2693          1.0793            1.0842         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_201.65     1_552.30     5_753.95       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_201.65     1_738.34     5_939.99       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_201.65     2_493.46     6_695.11       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64 (self)                                4_201.65     8_251.45    12_453.10       0.3194          1.0624            1.0635         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_881.12     3_002.36     8_883.49       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_881.12     3_334.84     9_215.96       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_881.12     4_806.29    10_687.42       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128 (self)                               5_881.12    15_915.45    21_796.57       0.4776          1.0316            1.0299         7.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Data where the structure resides on a lower-dimensional manifold.

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.13       695.65       728.77       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.13     2_356.81     2_389.94       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                649.78       710.86     1_360.64       0.2963          1.2496            1.2427         1.01
Exhaustive-PQ-m16 (self)                                 649.78     2_234.20     2_883.98       0.2324          1.3821            1.3758         1.01
Exhaustive-PQ-m32 (query)                              1_059.12     1_598.13     2_657.24       0.4045          1.1613            1.1558         1.78
Exhaustive-PQ-m32 (self)                               1_059.12     5_439.83     6_498.95       0.3204          1.2654            1.2587         1.78
Exhaustive-PQ-m64 (query)                              1_868.04     3_810.27     5_678.31       0.5368          1.0876            1.0837         3.30
Exhaustive-PQ-m64 (self)                               1_868.04    12_319.31    14_187.35       0.4607          1.1458            1.1406         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_477.90       197.05     1_674.95       0.5290          1.0891            1.0864         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_477.90       304.83     1_782.74       0.5290          1.0891            1.0864         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_477.90       427.10     1_905.01       0.5290          1.0891            1.0864         1.17
IVF-PQ-nl158-m16 (self)                                1_477.90     1_456.48     2_934.38       0.4271          1.1645            1.1601         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_854.83       345.96     2_200.78       0.6697          1.0403            1.0381         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_854.83       535.75     2_390.58       0.6697          1.0403            1.0381         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_854.83       735.59     2_590.42       0.6697          1.0403            1.0381         1.93
IVF-PQ-nl158-m32 (self)                                1_854.83     2_457.29     4_312.11       0.6070          1.0684            1.0636         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_391.84       616.55     3_008.39       0.8318          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_391.84       984.27     3_376.11       0.8318          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_391.84     1_351.85     3_743.69       0.8318          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                2_391.84     4_526.02     6_917.86       0.7985          1.0163            1.0140         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_298.49       315.17     1_613.66       0.5325          1.0877            1.0847         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_298.49       376.43     1_674.92       0.5326          1.0877            1.0846         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_298.49       539.00     1_837.49       0.5326          1.0877            1.0846         1.23
IVF-PQ-nl223-m16 (self)                                1_298.49     1_803.98     3_102.47       0.4210          1.1695            1.1658         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_663.59       502.92     2_166.51       0.6722          1.0394            1.0372         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_663.59       610.85     2_274.44       0.6724          1.0394            1.0371         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_663.59       906.25     2_569.84       0.6724          1.0394            1.0371         2.00
IVF-PQ-nl223-m32 (self)                                1_663.59     3_003.27     4_666.86       0.6051          1.0691            1.0645         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_180.05       928.17     3_108.22       0.8348          1.0091            1.0078         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_180.05     1_182.55     3_362.60       0.8352          1.0090            1.0078         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_180.05     1_681.95     3_862.00       0.8352          1.0090            1.0078         3.52
IVF-PQ-nl223-m64 (self)                                2_180.05     5_469.76     7_649.81       0.8012          1.0158            1.0137         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_453.37       375.67     1_829.05       0.5293          1.0886            1.0858         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_453.37       421.77     1_875.15       0.5293          1.0886            1.0858         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_453.37       593.99     2_047.37       0.5293          1.0886            1.0858         1.32
IVF-PQ-nl316-m16 (self)                                1_453.37     1_955.77     3_409.14       0.4124          1.1751            1.1711         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_830.19       638.01     2_468.20       0.6746          1.0390            1.0369         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_830.19       711.79     2_541.98       0.6747          1.0390            1.0368         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_830.19     1_032.88     2_863.07       0.6747          1.0390            1.0368         2.09
IVF-PQ-nl316-m32 (self)                                1_830.19     3_409.77     5_239.96       0.6024          1.0702            1.0658         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_479.94     1_148.03     3_627.96       0.8369          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_479.94     1_290.38     3_770.32       0.8370          1.0088            1.0076         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_479.94     1_880.40     4_360.34       0.8371          1.0088            1.0076         3.61
IVF-PQ-nl316-m64 (self)                                2_479.94     6_214.18     8_694.12       0.8027          1.0155            1.0136         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.43     1_277.34     1_346.76       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.43     4_162.26     4_231.69       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                880.22       684.06     1_564.28       0.2165          1.2233            1.2207         1.26
Exhaustive-PQ-m16 (self)                                 880.22     2_203.39     3_083.61       0.1776          1.3091            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_283.23     1_544.08     2_827.31       0.2863          1.1669            1.1636         2.03
Exhaustive-PQ-m32 (self)                               1_283.23     5_018.43     6_301.66       0.2244          1.2482            1.2473         2.03
Exhaustive-PQ-m64 (query)                              2_080.72     3_595.69     5_676.41       0.3812          1.1141            1.1115         3.55
Exhaustive-PQ-m64 (self)                               2_080.72    11_993.13    14_073.85       0.3012          1.1804            1.1788         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_218.29       264.26     2_482.56       0.3741          1.1188            1.1187         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_218.29       407.46     2_625.76       0.3741          1.1188            1.1187         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_218.29       563.28     2_781.57       0.3741          1.1188            1.1187         1.57
IVF-PQ-nl158-m16 (self)                                2_218.29     1_924.53     4_142.82       0.2675          1.2100            1.2132         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_759.64       397.91     3_157.55       0.4850          1.0727            1.0711         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_759.64       658.12     3_417.76       0.4850          1.0727            1.0711         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_759.64       848.70     3_608.34       0.4850          1.0727            1.0711         2.34
IVF-PQ-nl158-m32 (self)                                2_759.64     2_819.78     5_579.42       0.3900          1.1244            1.1232         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_712.30       738.27     4_450.57       0.6267          1.0349            1.0333         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_712.30     1_135.76     4_848.06       0.6267          1.0349            1.0333         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_712.30     1_528.94     5_241.24       0.6267          1.0349            1.0333         3.86
IVF-PQ-nl158-m64 (self)                                3_712.30     5_072.87     8_785.17       0.5759          1.0530            1.0492         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_934.71       411.81     2_346.52       0.3724          1.1189            1.1183         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_934.71       495.32     2_430.03       0.3724          1.1189            1.1183         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_934.71       721.12     2_655.83       0.3724          1.1189            1.1183         1.70
IVF-PQ-nl223-m16 (self)                                1_934.71     2_419.81     4_354.52       0.2586          1.2183            1.2212         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_313.33       586.41     2_899.74       0.4832          1.0732            1.0719         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_313.33       749.66     3_062.99       0.4832          1.0732            1.0719         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_313.33     1_038.70     3_352.03       0.4832          1.0732            1.0719         2.46
IVF-PQ-nl223-m32 (self)                                2_313.33     3_397.91     5_711.24       0.3779          1.1312            1.1297         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_187.14     1_047.93     4_235.06       0.6294          1.0346            1.0332         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_187.14     1_287.44     4_474.58       0.6294          1.0346            1.0332         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_187.14     2_001.27     5_188.41       0.6294          1.0346            1.0332         3.99
IVF-PQ-nl223-m64 (self)                                3_187.14     6_195.92     9_383.06       0.5713          1.0547            1.0507         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_301.08       538.95     2_840.03       0.3695          1.1197            1.1196         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_301.08       604.87     2_905.95       0.3695          1.1197            1.1196         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_301.08       841.35     3_142.43       0.3695          1.1197            1.1196         1.88
IVF-PQ-nl316-m16 (self)                                2_301.08     2_785.51     5_086.59       0.2528          1.2222            1.2257         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_697.54       742.00     3_439.54       0.4846          1.0727            1.0711         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_697.54       817.20     3_514.73       0.4846          1.0727            1.0711         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_697.54     1_166.21     3_863.75       0.4846          1.0727            1.0711         2.65
IVF-PQ-nl316-m32 (self)                                2_697.54     3_854.66     6_552.20       0.3705          1.1349            1.1338         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_576.22     1_341.51     4_917.73       0.6315          1.0340            1.0323         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_576.22     1_498.56     5_074.78       0.6315          1.0340            1.0323         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_576.22     2_128.94     5_705.16       0.6315          1.0340            1.0323         4.17
IVF-PQ-nl316-m64 (self)                                3_576.22     7_046.85    10_623.07       0.5691          1.0551            1.0512         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:pq:euclidean:lowrank:768:50000 -->
</code></pre>
</details>

##### Cell embeddings

Synthetic data that resembles the embeddings generated by single cell models
such as GeneFormer, scGPT, etc.

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.00       705.57       738.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.00     2_315.14     2_348.13       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                626.31       668.25     1_294.55       0.7119          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 626.31     2_187.93     2_814.24       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_070.99     1_499.89     2_570.87       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_070.99     4_986.39     6_057.38       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              1_627.80     3_550.07     5_177.87       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               1_627.80    11_733.89    13_361.68       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_475.66       213.46     1_689.13       0.8272          1.0522            1.0448         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_475.66       351.24     1_826.90       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_475.66       494.87     1_970.53       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16 (self)                                1_475.66     1_601.19     3_076.86       0.7669          1.0989            1.0836         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_908.54       387.03     2_295.57       0.8746          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_908.54       649.98     2_558.53       0.8751          1.0262            1.0217         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_908.54       901.74     2_810.28       0.8751          1.0262            1.0217         1.93
IVF-PQ-nl158-m32 (self)                                1_908.54     2_995.95     4_904.49       0.8288          1.0511            1.0423         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_448.43       714.44     3_162.87       0.9048          1.0151            1.0118         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_448.43     1_205.11     3_653.54       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_448.43     1_683.10     4_131.53       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64 (self)                                2_448.43     5_623.00     8_071.43       0.8704          1.0288            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_092.14       294.49     1_386.63       0.8428          1.0430            1.0365         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_092.14       365.65     1_457.79       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_092.14       567.62     1_659.77       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16 (self)                                1_092.14     1_795.33     2_887.47       0.7841          1.0842            1.0704         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_502.38       540.20     2_042.58       0.8837          1.0224            1.0183         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_502.38       669.03     2_171.42       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_502.38       992.55     2_494.94       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32 (self)                                1_502.38     3_290.72     4_793.10       0.8403          1.0440            1.0356         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_035.29       954.72     2_990.01       0.9100          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_035.29     1_202.47     3_237.76       0.9102          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_035.29     1_829.52     3_864.80       0.9102          1.0133            1.0102         3.52
IVF-PQ-nl223-m64 (self)                                2_035.29     6_063.47     8_098.76       0.8765          1.0259            1.0200         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_377.17       396.85     1_774.02       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_377.17       443.68     1_820.86       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_377.17       643.46     2_020.64       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16 (self)                                1_377.17     2_125.87     3_503.05       0.7922          1.0785            1.0637         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_758.26       687.96     2_446.22       0.8867          1.0214            1.0175         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_758.26       780.09     2_538.35       0.8867          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_758.26     1_124.35     2_882.61       0.8868          1.0214            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_758.26     3_726.12     5_484.38       0.8425          1.0433            1.0344         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_269.00     1_190.21     3_459.21       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_269.00     1_347.72     3_616.72       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_269.00     1_962.60     4_231.60       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64 (self)                                2_269.00     6_589.68     8_858.68       0.8791          1.0247            1.0188         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.76     1_269.50     1_340.26       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.76     4_169.23     4_239.99       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                870.51       678.33     1_548.84       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 870.51     2_214.61     3_085.12       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_257.50     1_529.80     2_787.30       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_257.50     5_057.70     6_315.20       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_314.37     3_697.13     6_011.49       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_314.37    11_933.40    14_247.76       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_647.93       275.86     2_923.79       0.8455          1.0448            1.0357         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_647.93       438.76     3_086.69       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_647.93       620.00     3_267.93       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16 (self)                                2_647.93     2_025.83     4_673.76       0.7844          1.0913            1.0648         1.57
IVF-PQ-nl158-m32-np7 (query)                           3_025.75       462.19     3_487.95       0.8726          1.0297            1.0231         2.34
IVF-PQ-nl158-m32-np12 (query)                          3_025.75       734.78     3_760.53       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32-np17 (query)                          3_025.75     1_046.89     4_072.65       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32 (self)                                3_025.75     3_384.82     6_410.57       0.8208          1.0615            1.0426         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_069.54       849.78     4_919.32       0.8936          1.0202            1.0150         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_069.54     1_409.78     5_479.32       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_069.54     1_955.18     6_024.73       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64 (self)                                4_069.54     6_508.01    10_577.55       0.8494          1.0420            1.0290         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_719.89       441.41     2_161.30       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_719.89       573.14     2_293.03       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_719.89       837.74     2_557.63       0.8544          1.0397            1.0313         1.70
IVF-PQ-nl223-m16 (self)                                1_719.89     2_725.88     4_445.77       0.7965          1.0807            1.0566         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_207.71       639.80     2_847.50       0.8795          1.0266            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_207.71       792.18     2_999.89       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_207.71     1_129.81     3_337.52       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                2_207.71     3_690.84     5_898.55       0.8306          1.0550            1.0377         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_934.64     1_129.07     4_063.71       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_934.64     1_388.78     4_323.42       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_934.64     2_051.89     4_986.53       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64 (self)                                2_934.64     6_739.41     9_674.05       0.8568          1.0378            1.0256         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_912.51       536.59     2_449.11       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_912.51       588.71     2_501.23       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_912.51       843.28     2_755.79       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16 (self)                                1_912.51     2_793.73     4_706.25       0.8149          1.0655            1.0461         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_468.16       788.79     3_256.95       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_468.16       884.29     3_352.45       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_468.16     1_262.65     3_730.81       0.8917          1.0214            1.0159         2.65
IVF-PQ-nl316-m32 (self)                                2_468.16     4_105.59     6_573.76       0.8455          1.0452            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_114.06     1_400.56     4_514.62       0.9064          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_114.06     1_580.08     4_694.14       0.9064          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_114.06     2_276.88     5_390.95       0.9065          1.0155            1.0112         4.17
IVF-PQ-nl316-m64 (self)                                3_114.06     7_518.78    10_632.84       0.8655          1.0331            1.0218         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:pq:euclidean:embedding:768:50000 -->
</code></pre>
</details>

#### Optimised product quantisation (Exhaustive and IVF)

PQ with a learned rotation applied first, so the subvector splits land on axes
that carry independent variance. Same compression ratio as PQ, substantially
longer build. Worth reaching for when the data has a correlation structure a
single global rotation can actually align.

**Tunable parameters:** identical to [PQ](#product-quantisation-exhaustive-and-ivf),
with the rotation learned during the build rather than exposed as a knob.

The [locality argument](#why-the-ivf-variant-beats-the-exhaustive-one) from PQ
applies unchanged. OPQ's rotation improves subspace independence but does not
create locality: it transforms the data without reducing its intrinsic spread,
so the clustering step is still not optional.

##### Correlated data

As for PQ, let's start with correlated data.

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:correlated:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:correlated:768:50000 -->
</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:lowrank:768:50000 -->
</code></pre>
</details>

##### Cell embeddings

Lastly, also here the synthetic data that resembles the embeddings generated by
single cell models.

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:embedding:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:opq:euclidean:embedding:768:50000 -->
</code></pre>
</details>

### SOAR-PQ and SOAR-OPQ

[SOAR](benchmarks_standard.md#soar) spilling on top of IVF-PQ and IVF-OPQ. On
exact full-vector search spilling loses, because probing one more cell there
costs nothing fixed. PQ changes the arithmetic: codes are residuals against a
cell centroid, so every probed cell has to rebuild the ADC lookup table at
`n_pq_centroids * dim` operations before it scores a single candidate. At 512
dimensions that table build is over an order of magnitude more expensive than
scanning the cell it serves, so pulling twice the candidates out of *one* cell
should beat pulling them out of two.

Spilling costs `2 * n * m` code bytes instead of `n * m`. The comparison that
matters is therefore against an IVF-PQ index with **twice the subspaces**, which
is the third column in sweep A. Beating IVF-PQ at the same `m` whilst using
twice the memory would prove nothing.

OPQ adds a learned rotation on top: codes are `PQ(R * r)`, and the lookup table
is built from the *rotated* query residual. `R` is orthogonal, so distances hold
only when both sides are rotated.

**Tunable parameters:**

- *Subvector width*: Fixed at 16 here, so `m = dim / 16` and the equal-memory
  column runs `2 * m`, which stays a divisor of `dim` for any `dim` that is a
  multiple of 16.
- *Number of lists (nl)*: Skewed lower than the exact-search SOAR sweep, since
  per-query cost is dominated by the per-cell table rebuild rather than by the
  candidate scan.
- *Number of probes (np)*: As SOAR, skewed low. Read recall against query time,
  not against `nprobe`.
- *Rule*: The three secondary-assignment rules from
  [SOAR](benchmarks_standard.md#soar), swept in the second table.

Default target is 50k cell embeddings at 512 dimensions, the foundation-model
regime where PQ earns its place.

#### SOAR-PQ

<details>
<summary><b>SOAR-PQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_pq:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_pq:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_pq:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_pq:cosine:embedding:512:50000 -->
</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_opq:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_opq:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_opq:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
<!-- BENCH:soar_opq:cosine:embedding:512:50000 -->
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
