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
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.20       630.77       641.97       1.0000          1.0000        18.31
Exhaustive (self)                                         11.20     6_303.18     6_314.38       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                   12.95     1_223.04     1_235.99       0.9828          1.0001         9.16
Exhaustive-BF16 (self)                                    12.95    12_241.56    12_254.51       0.9798          1.0001         9.16
IVF-BF16-nl273-np13 (query)                              307.70        93.03       400.72       0.9806          1.0003         9.19
IVF-BF16-nl273-np16 (query)                              307.70       102.15       409.85       0.9825          1.0001         9.19
IVF-BF16-nl273-np23 (query)                              307.70       137.63       445.32       0.9828          1.0001         9.19
IVF-BF16-nl273 (self)                                    307.70     1_424.97     1_732.67       0.9798          1.0001         9.19
IVF-BF16-nl387-np19 (query)                              559.66        96.80       656.46       0.9820          1.0001         9.21
IVF-BF16-nl387-np27 (query)                              559.66       124.80       684.46       0.9828          1.0001         9.21
IVF-BF16-nl387 (self)                                    559.66     1_256.21     1_815.87       0.9798          1.0001         9.21
IVF-BF16-nl547-np23 (query)                            1_085.73        91.40     1_177.12       0.9773          1.0005         9.23
IVF-BF16-nl547-np27 (query)                            1_085.73       102.36     1_188.09       0.9816          1.0002         9.23
IVF-BF16-nl547-np33 (query)                            1_085.73       116.88     1_202.61       0.9828          1.0001         9.23
IVF-BF16-nl547 (self)                                  1_085.73     1_198.40     2_284.13       0.9798          1.0001         9.23
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
Exhaustive (query)                                        11.73       709.56       721.29       1.0000          1.0000        18.88
Exhaustive (self)                                         11.73     6_980.83     6_992.56       1.0000          1.0000        18.88
Exhaustive-BF16 (query)                                   15.46     1_243.65     1_259.12       0.8870          1.0071         9.44
Exhaustive-BF16 (self)                                    15.46    12_443.36    12_458.82       0.8852          1.0073         9.44
IVF-BF16-nl273-np13 (query)                              303.77        95.57       399.34       0.8860          1.0073         9.48
IVF-BF16-nl273-np16 (query)                              303.77       109.29       413.06       0.8870          1.0071         9.48
IVF-BF16-nl273-np23 (query)                              303.77       147.74       451.51       0.8870          1.0071         9.48
IVF-BF16-nl273 (self)                                    303.77     1_538.86     1_842.63       0.8852          1.0073         9.48
IVF-BF16-nl387-np19 (query)                              551.93       100.62       652.55       0.8867          1.0072         9.49
IVF-BF16-nl387-np27 (query)                              551.93       129.50       681.43       0.8870          1.0071         9.49
IVF-BF16-nl387 (self)                                    551.93     1_341.78     1_893.71       0.8852          1.0073         9.49
IVF-BF16-nl547-np23 (query)                            1_026.06        94.00     1_120.06       0.8848          1.0075         9.51
IVF-BF16-nl547-np27 (query)                            1_026.06       104.17     1_130.23       0.8866          1.0072         9.51
IVF-BF16-nl547-np33 (query)                            1_026.06       121.54     1_147.60       0.8870          1.0071         9.51
IVF-BF16-nl547 (self)                                  1_026.06     1_248.19     2_274.25       0.8852          1.0073         9.51
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
Exhaustive (query)                                        11.14       625.84       636.98       1.0000          1.0000        18.31
Exhaustive (self)                                         11.14     6_178.69     6_189.83       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                   13.75     1_173.30     1_187.05       0.9223          1.0022         9.16
Exhaustive-BF16 (self)                                    13.75    11_684.29    11_698.04       0.9032          1.0037         9.16
IVF-BF16-nl273-np13 (query)                              289.89        92.89       382.78       0.9223          1.0022         9.19
IVF-BF16-nl273-np16 (query)                              289.89       107.71       397.61       0.9223          1.0022         9.19
IVF-BF16-nl273-np23 (query)                              289.89       145.11       435.01       0.9223          1.0022         9.19
IVF-BF16-nl273 (self)                                    289.89     1_471.55     1_761.44       0.9032          1.0037         9.19
IVF-BF16-nl387-np19 (query)                              540.08        94.78       634.87       0.9223          1.0022         9.21
IVF-BF16-nl387-np27 (query)                              540.08       124.64       664.73       0.9223          1.0022         9.21
IVF-BF16-nl387 (self)                                    540.08     1_233.59     1_773.67       0.9032          1.0037         9.21
IVF-BF16-nl547-np23 (query)                            1_030.28        86.99     1_117.27       0.9223          1.0022         9.23
IVF-BF16-nl547-np27 (query)                            1_030.28        96.88     1_127.16       0.9223          1.0022         9.23
IVF-BF16-nl547-np33 (query)                            1_030.28       112.77     1_143.05       0.9223          1.0022         9.23
IVF-BF16-nl547 (self)                                  1_030.28     1_133.61     2_163.88       0.9032          1.0037         9.23
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
Exhaustive (query)                                        11.33       617.93       629.25       1.0000          1.0000        18.31
Exhaustive (self)                                         11.33     6_098.73     6_110.05       1.0000          1.0000        18.31
Exhaustive-BF16 (query)                                   12.54     1_179.79     1_192.33       0.9515          1.0010         9.16
Exhaustive-BF16 (self)                                    12.54    11_615.49    11_628.03       0.9405          1.0018         9.16
IVF-BF16-nl273-np13 (query)                              294.95        81.62       376.56       0.9515          1.0010         9.19
IVF-BF16-nl273-np16 (query)                              294.95        90.94       385.89       0.9515          1.0010         9.19
IVF-BF16-nl273-np23 (query)                              294.95       120.52       415.47       0.9515          1.0010         9.19
IVF-BF16-nl273 (self)                                    294.95     1_215.04     1_509.98       0.9405          1.0018         9.19
IVF-BF16-nl387-np19 (query)                              543.02        85.88       628.89       0.9515          1.0010         9.21
IVF-BF16-nl387-np27 (query)                              543.02       107.96       650.98       0.9515          1.0010         9.21
IVF-BF16-nl387 (self)                                    543.02     1_074.57     1_617.59       0.9405          1.0018         9.21
IVF-BF16-nl547-np23 (query)                            1_030.74        83.85     1_114.59       0.9515          1.0010         9.23
IVF-BF16-nl547-np27 (query)                            1_030.74        92.54     1_123.28       0.9515          1.0010         9.23
IVF-BF16-nl547-np33 (query)                            1_030.74       102.17     1_132.91       0.9515          1.0010         9.23
IVF-BF16-nl547 (self)                                  1_030.74     1_019.60     2_050.34       0.9405          1.0018         9.23
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
Exhaustive (query)                                        48.68     1_187.90     1_236.58       1.0000          1.0000        73.24
Exhaustive (self)                                         48.68    11_670.62    11_719.31       1.0000          1.0000        73.24
Exhaustive-BF16 (query)                                   55.17     5_071.36     5_126.53       0.9716          1.0003        36.62
Exhaustive-BF16 (self)                                    55.17    52_901.61    52_956.78       0.9674          1.0005        36.62
IVF-BF16-nl273-np13 (query)                              588.65       265.36       854.00       0.9716          1.0003        36.76
IVF-BF16-nl273-np16 (query)                              588.65       299.49       888.14       0.9716          1.0003        36.76
IVF-BF16-nl273-np23 (query)                              588.65       416.69     1_005.34       0.9716          1.0003        36.76
IVF-BF16-nl273 (self)                                    588.65     4_456.66     5_045.31       0.9674          1.0005        36.76
IVF-BF16-nl387-np19 (query)                            1_137.41       275.31     1_412.72       0.9716          1.0003        36.81
IVF-BF16-nl387-np27 (query)                            1_137.41       365.89     1_503.30       0.9716          1.0003        36.81
IVF-BF16-nl387 (self)                                  1_137.41     3_618.79     4_756.20       0.9674          1.0005        36.81
IVF-BF16-nl547-np23 (query)                            2_377.05       265.48     2_642.53       0.9716          1.0003        36.89
IVF-BF16-nl547-np27 (query)                            2_377.05       290.14     2_667.19       0.9716          1.0003        36.89
IVF-BF16-nl547-np33 (query)                            2_377.05       331.40     2_708.45       0.9716          1.0003        36.89
IVF-BF16-nl547 (self)                                  2_377.05     3_316.05     5_693.10       0.9674          1.0005        36.89
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.24       618.40       629.64       1.0000          1.0000        18.31
Exhaustive (self)                                         11.24     6_109.53     6_120.77       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    17.49       980.85       998.35       0.9256          1.0018         5.15
Exhaustive-SQ8 (self)                                     17.49     9_678.98     9_696.47       0.9251          1.0018         5.15
IVF-SQ8-nl273-np13 (query)                               292.34        65.48       357.82       0.9244          1.0020         6.33
IVF-SQ8-nl273-np16 (query)                               292.34        75.11       367.45       0.9258          1.0018         6.33
IVF-SQ8-nl273-np23 (query)                               292.34        94.28       386.62       0.9260          1.0018         6.33
IVF-SQ8-nl273 (self)                                     292.34       937.58     1_229.92       0.9253          1.0018         6.33
IVF-SQ8-nl387-np19 (query)                               543.91        68.07       611.98       0.9243          1.0019         6.35
IVF-SQ8-nl387-np27 (query)                               543.91        85.48       629.38       0.9248          1.0018         6.35
IVF-SQ8-nl387 (self)                                     543.91       845.11     1_389.01       0.9252          1.0018         6.35
IVF-SQ8-nl547-np23 (query)                             1_044.89        65.46     1_110.35       0.9215          1.0022         6.37
IVF-SQ8-nl547-np27 (query)                             1_044.89        71.88     1_116.76       0.9244          1.0019         6.37
IVF-SQ8-nl547-np33 (query)                             1_044.89        85.01     1_129.89       0.9251          1.0018         6.37
IVF-SQ8-nl547 (self)                                   1_044.89       816.32     1_861.20       0.9252          1.0018         6.37
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
Exhaustive (query)                                        11.77       691.20       702.98       1.0000          1.0000        18.88
Exhaustive (self)                                         11.77     6_722.80     6_734.57       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                    19.70       895.08       914.78       0.7397          1.0354         5.15
Exhaustive-SQ8 (self)                                     19.70     9_791.09     9_810.79       0.7390          1.0356         5.15
IVF-SQ8-nl273-np13 (query)                               290.98        63.50       354.48       0.7391          1.0362         6.33
IVF-SQ8-nl273-np16 (query)                               290.98        70.76       361.74       0.7395          1.0361         6.33
IVF-SQ8-nl273-np23 (query)                               290.98        92.90       383.88       0.7395          1.0361         6.33
IVF-SQ8-nl273 (self)                                     290.98       979.89     1_270.87       0.7378          1.0362         6.33
IVF-SQ8-nl387-np19 (query)                               537.80        66.12       603.91       0.7378          1.0360         6.35
IVF-SQ8-nl387-np27 (query)                               537.80        86.94       624.73       0.7379          1.0360         6.35
IVF-SQ8-nl387 (self)                                     537.80       861.97     1_399.77       0.7375          1.0362         6.35
IVF-SQ8-nl547-np23 (query)                             1_016.07        62.97     1_079.03       0.7365          1.0365         6.37
IVF-SQ8-nl547-np27 (query)                             1_016.07        69.44     1_085.51       0.7370          1.0364         6.37
IVF-SQ8-nl547-np33 (query)                             1_016.07        81.54     1_097.61       0.7369          1.0364         6.37
IVF-SQ8-nl547 (self)                                   1_016.07       832.17     1_848.24       0.7360          1.0365         6.37
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
Exhaustive (query)                                        12.06       616.34       628.39       1.0000          1.0000        18.31
Exhaustive (self)                                         12.06     6_064.67     6_076.72       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    17.21       976.39       993.60       0.7805          1.0206         5.15
Exhaustive-SQ8 (self)                                     17.21     9_691.27     9_708.48       0.7786          1.0217         5.15
IVF-SQ8-nl273-np13 (query)                               301.50        67.30       368.80       0.7818          1.0205         6.33
IVF-SQ8-nl273-np16 (query)                               301.50        76.36       377.87       0.7818          1.0205         6.33
IVF-SQ8-nl273-np23 (query)                               301.50       102.90       404.41       0.7818          1.0205         6.33
IVF-SQ8-nl273 (self)                                     301.50     1_012.48     1_313.98       0.7787          1.0217         6.33
IVF-SQ8-nl387-np19 (query)                               557.98        68.65       626.64       0.7823          1.0204         6.35
IVF-SQ8-nl387-np27 (query)                               557.98        87.39       645.37       0.7823          1.0204         6.35
IVF-SQ8-nl387 (self)                                     557.98       860.66     1_418.64       0.7786          1.0217         6.35
IVF-SQ8-nl547-np23 (query)                             1_032.54        67.37     1_099.91       0.7813          1.0206         6.37
IVF-SQ8-nl547-np27 (query)                             1_032.54        71.29     1_103.83       0.7813          1.0206         6.37
IVF-SQ8-nl547-np33 (query)                             1_032.54        84.90     1_117.44       0.7813          1.0206         6.37
IVF-SQ8-nl547 (self)                                   1_032.54       813.96     1_846.50       0.7788          1.0217         6.37
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
Exhaustive (query)                                        11.20       630.79       641.99       1.0000          1.0000        18.31
Exhaustive (self)                                         11.20     6_029.88     6_041.08       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    17.20       965.33       982.53       0.7864          1.0270         5.15
Exhaustive-SQ8 (self)                                     17.20     9_650.90     9_668.10       0.7862          1.0286         5.15
IVF-SQ8-nl273-np13 (query)                               291.53        59.61       351.15       0.7866          1.0270         6.33
IVF-SQ8-nl273-np16 (query)                               291.53        65.44       356.97       0.7866          1.0270         6.33
IVF-SQ8-nl273-np23 (query)                               291.53        86.28       377.81       0.7866          1.0270         6.33
IVF-SQ8-nl273 (self)                                     291.53       858.49     1_150.02       0.7861          1.0286         6.33
IVF-SQ8-nl387-np19 (query)                               558.83        62.24       621.07       0.7867          1.0269         6.35
IVF-SQ8-nl387-np27 (query)                               558.83        76.74       635.56       0.7867          1.0269         6.35
IVF-SQ8-nl387 (self)                                     558.83       751.88     1_310.71       0.7864          1.0286         6.35
IVF-SQ8-nl547-np23 (query)                             1_034.97        62.39     1_097.36       0.7856          1.0271         6.37
IVF-SQ8-nl547-np27 (query)                             1_034.97        67.68     1_102.64       0.7856          1.0271         6.37
IVF-SQ8-nl547-np33 (query)                             1_034.97        74.93     1_109.90       0.7856          1.0271         6.37
IVF-SQ8-nl547 (self)                                   1_034.97       734.13     1_769.10       0.7865          1.0286         6.37
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### More dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.56     1_180.38     1_228.94       1.0000          1.0000        73.24
Exhaustive (self)                                         48.56    11_629.69    11_678.25       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    79.87     1_163.02     1_242.90       0.8843          1.0056        18.88
Exhaustive-SQ8 (self)                                     79.87    11_790.48    11_870.35       0.8898          1.0067        18.88
IVF-SQ8-nl273-np13 (query)                               604.38        79.60       683.98       0.8827          1.0057        20.16
IVF-SQ8-nl273-np16 (query)                               604.38        85.94       690.32       0.8827          1.0057        20.16
IVF-SQ8-nl273-np23 (query)                               604.38       112.87       717.25       0.8827          1.0057        20.16
IVF-SQ8-nl273 (self)                                     604.38       923.43     1_527.81       0.8898          1.0067        20.16
IVF-SQ8-nl387-np19 (query)                             1_140.00        89.30     1_229.29       0.8840          1.0056        20.22
IVF-SQ8-nl387-np27 (query)                             1_140.00       104.91     1_244.91       0.8840          1.0056        20.22
IVF-SQ8-nl387 (self)                                   1_140.00       848.01     1_988.00       0.8898          1.0067        20.22
IVF-SQ8-nl547-np23 (query)                             2_625.96        99.21     2_725.17       0.8836          1.0056        20.30
IVF-SQ8-nl547-np27 (query)                             2_625.96       100.01     2_725.97       0.8836          1.0056        20.30
IVF-SQ8-nl547-np33 (query)                             2_625.96       115.52     2_741.48       0.8836          1.0056        20.30
IVF-SQ8-nl547 (self)                                   2_625.96       981.27     3_607.23       0.8898          1.0067        20.30
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.23       659.01       670.24       1.0000          1.0000        18.31
Exhaustive (self)                                         11.23     6_324.93     6_336.16       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    19.83       980.95     1_000.77       0.9256          1.0018         5.15
HNSW-M16-ef100-s50 (query)                               790.08        54.91       844.99       0.9308          1.0117        38.52
HNSW-M16-ef100-s100 (query)                              790.08       100.49       890.57       0.9654          1.0053        38.52
HNSW-M16-ef100-s200 (query)                              790.08       186.99       977.07       0.9839          1.0026        38.52
HNSW-M16-ef100 (self)                                    790.08       964.73     1_754.81       0.9652          1.0062        38.52
HNSW-M16-ef200-s50 (query)                             1_506.91        57.84     1_564.75       0.9599          1.0089        38.52
HNSW-M16-ef200-s100 (query)                            1_506.91       106.67     1_613.58       0.9830          1.0045        38.52
HNSW-M16-ef200-s200 (query)                            1_506.91       199.67     1_706.58       0.9923          1.0029        38.52
HNSW-M16-ef200 (self)                                  1_506.91     1_038.13     2_545.04       0.9832          1.0037        38.52
HNSW-M24-ef200-s50 (query)                             1_636.28        63.56     1_699.84       0.9698          1.0301        47.66
HNSW-M24-ef200-s100 (query)                            1_636.28       112.86     1_749.14       0.9882          1.0175        47.66
HNSW-M24-ef200-s200 (query)                            1_636.28       215.91     1_852.19       0.9953          1.0059        47.66
HNSW-M24-ef200 (self)                                  1_636.28     1_101.37     2_737.65       0.9883          1.0128        47.66
HNSW-M32-ef200-s50 (query)                             1_691.55        68.47     1_760.02       0.9712          1.0391        56.80
HNSW-M32-ef200-s100 (query)                            1_691.55       124.57     1_816.12       0.9883          1.0330        56.80
HNSW-M32-ef200-s200 (query)                            1_691.55       227.18     1_918.73       0.9959          1.0030        56.80
HNSW-M32-ef200 (self)                                  1_691.55     1_187.97     2_879.52       0.9885          1.0365        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          682.31        34.34       716.65       0.8761          1.0272        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         682.31        64.97       747.28       0.9016          1.0107        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         682.31       117.18       799.49       0.9150          1.0059        26.89
HNSW-SQ8U-M16-ef100 (self)                               682.31       598.66     1_280.96       0.9014          1.0114        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_334.98        48.30     1_383.27       0.8993          1.0081        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_334.98        68.69     1_403.67       0.9148          1.0047        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_334.98       125.65     1_460.63       0.9208          1.0030        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_334.98       641.63     1_976.60       0.9147          1.0046        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_444.96        40.85     1_485.80       0.9065          1.0121        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_444.96        72.87     1_517.82       0.9183          1.0059        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_444.96       142.47     1_587.42       0.9226          1.0038        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_444.96       709.22     2_154.17       0.9180          1.0055        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_512.72        41.77     1_554.49       0.9089          1.0050        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_512.72        77.54     1_590.26       0.9197          1.0038        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_512.72       138.61     1_651.33       0.9235          1.0021        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_512.72       736.00     2_248.71       0.9194          1.0033        45.20
HNSW-SQ8U-drop0 (query)                                1_370.80        64.57     1_435.38       0.8936          1.0172        26.89
HNSW-SQ8U-drop0.001 (query)                            1_337.50        69.36     1_406.86       0.9150          1.0070        26.89
HNSW-SQ8U-drop0.01 (query)                             1_344.72        66.22     1_410.93       0.8983          1.0076        26.89
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.77       693.35       705.12       1.0000          1.0000        18.88
Exhaustive (self)                                         11.77     6_656.80     6_668.57       1.0000          1.0000        18.88
Exhaustive-SQ8 (query)                                    20.88       977.01       997.89       0.7397          1.0354         5.15
HNSW-M16-ef100-s50 (query)                               841.49        55.19       896.68       0.9359          1.0189        39.09
HNSW-M16-ef100-s100 (query)                              841.49       110.29       951.78       0.9699          1.0093        39.09
HNSW-M16-ef100-s200 (query)                              841.49       192.52     1_034.00       0.9880          1.0051        39.09
HNSW-M16-ef100 (self)                                    841.49     1_010.25     1_851.74       0.9695          1.0080        39.09
HNSW-M16-ef200-s50 (query)                             1_638.05        59.35     1_697.40       0.9633          1.0128        39.09
HNSW-M16-ef200-s100 (query)                            1_638.05       108.00     1_746.05       0.9866          1.0051        39.09
HNSW-M16-ef200-s200 (query)                            1_638.05       201.72     1_839.77       0.9951          1.0016        39.09
HNSW-M16-ef200 (self)                                  1_638.05     1_079.09     2_717.14       0.9869          1.0051        39.09
HNSW-M24-ef200-s50 (query)                             1_778.08        67.56     1_845.63       0.9733          1.0087        48.23
HNSW-M24-ef200-s100 (query)                            1_778.08       120.75     1_898.82       0.9909          1.0016        48.23
HNSW-M24-ef200-s200 (query)                            1_778.08       217.86     1_995.93       0.9970          1.0007        48.23
HNSW-M24-ef200 (self)                                  1_778.08     1_199.44     2_977.52       0.9908          1.0057        48.23
HNSW-M32-ef200-s50 (query)                             1_816.01        72.21     1_888.22       0.9753          1.0132        57.37
HNSW-M32-ef200-s100 (query)                            1_816.01       131.10     1_947.10       0.9917          1.0027        57.37
HNSW-M32-ef200-s200 (query)                            1_816.01       229.05     2_045.06       0.9974          1.0002        57.37
HNSW-M32-ef200 (self)                                  1_816.01     1_222.79     3_038.79       0.9915          1.0051        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          722.14        34.85       756.99       0.6843          1.0562        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         722.14        65.36       787.50       0.7085          1.0491        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         722.14       123.24       845.38       0.7225          1.0428        26.89
HNSW-SQ8U-M16-ef100 (self)                               722.14       611.14     1_333.28       0.7079          1.0486        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_379.20        37.31     1_416.50       0.7080          1.0492        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_379.20        69.55     1_448.74       0.7242          1.0445        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_379.20       127.41     1_506.61       0.7317          1.0397        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_379.20       658.79     2_037.99       0.7235          1.0447        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_470.60        41.26     1_511.86       0.7182          1.0480        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_470.60        75.75     1_546.35       0.7303          1.0460        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_470.60       137.11     1_607.70       0.7357          1.0367        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_470.60       721.30     2_191.89       0.7299          1.0405        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_566.15        42.92     1_609.07       0.7215          1.0436        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_566.15        79.72     1_645.87       0.7323          1.0385        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_566.15       139.69     1_705.84       0.7365          1.0368        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_566.15       744.01     2_310.16       0.7317          1.0387        45.20
HNSW-SQ8U-drop0 (query)                                1_363.39        66.94     1_430.33       0.6649          1.0638        26.89
HNSW-SQ8U-drop0.001 (query)                            1_371.70        68.67     1_440.37       0.7233          1.0470        26.89
HNSW-SQ8U-drop0.01 (query)                             1_336.98        67.53     1_404.51       0.6897          1.0527        26.89
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.26       654.32       665.57       1.0000          1.0000        18.31
Exhaustive (self)                                         11.26     6_490.94     6_502.20       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    17.09       963.88       980.97       0.7805          1.0206         5.15
HNSW-M16-ef100-s50 (query)                               826.94        59.12       886.06       0.9967          1.0001        38.52
HNSW-M16-ef100-s100 (query)                              826.94       104.04       930.98       0.9987          1.0000        38.52
HNSW-M16-ef100-s200 (query)                              826.94       187.42     1_014.36       0.9989          1.0000        38.52
HNSW-M16-ef100 (self)                                    826.94     1_030.71     1_857.65       0.9987          1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_464.99        60.80     1_525.79       0.9958          3.0171        38.52
HNSW-M16-ef200-s100 (query)                            1_464.99       106.69     1_571.68       0.9975          3.0170        38.52
HNSW-M16-ef200-s200 (query)                            1_464.99       192.10     1_657.09       0.9989          1.0000        38.52
HNSW-M16-ef200 (self)                                  1_464.99     1_051.07     2_516.06       0.9971          3.6732        38.52
HNSW-M24-ef200-s50 (query)                             1_555.26        67.29     1_622.56       0.9982          1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_555.26       117.62     1_672.88       0.9989          1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_555.26       205.07     1_760.34       0.9989          1.0000        47.66
HNSW-M24-ef200 (self)                                  1_555.26     1_175.70     2_730.96       0.9988          1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_667.37        70.99     1_738.36       0.9699         43.1293        56.80
HNSW-M32-ef200-s100 (query)                            1_667.37       121.38     1_788.75       0.9704         43.0395        56.80
HNSW-M32-ef200-s200 (query)                            1_667.37       218.91     1_886.28       0.9882          1.7987        56.80
HNSW-M32-ef200 (self)                                  1_667.37     1_205.13     2_872.50       0.9707         45.9590        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          748.95        37.20       786.15       0.7793          2.3959        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         748.95        68.58       817.53       0.7803          1.0949        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         748.95       124.20       873.15       0.7804          1.0510        26.89
HNSW-SQ8U-M16-ef100 (self)                               748.95       682.24     1_431.19       0.7785          1.0634        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_325.42        38.52     1_363.95       0.7802          1.1172        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_325.42        72.61     1_398.03       0.7804          1.1172        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_325.42       132.69     1_458.11       0.7804          1.0206        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_325.42       692.73     2_018.16       0.7786          1.1110        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_436.40        41.38     1_477.78       0.7804          1.0206        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_436.40        75.74     1_512.14       0.7805          1.0206        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_436.40       131.95     1_568.35       0.7805          1.0206        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_436.40       714.18     2_150.59       0.7786          1.0217        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_499.55        48.49     1_548.04       0.7805          1.0206        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_499.55        82.91     1_582.45       0.7805          1.0206        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_499.55       144.39     1_643.93       0.7805          1.0206        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_499.55       792.69     2_292.24       0.7786          1.0217        45.20
HNSW-SQ8U-drop0 (query)                                1_333.17        68.08     1_401.25       0.7780          1.0213        26.89
HNSW-SQ8U-drop0.001 (query)                            1_360.12        70.22     1_430.35       0.7804          1.0206        26.89
HNSW-SQ8U-drop0.01 (query)                             1_329.15        67.99     1_397.13       0.7731          1.0234        26.89
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.37       667.86       679.22       1.0000          1.0000        18.31
Exhaustive (self)                                         11.37     6_589.46     6_600.83       1.0000          1.0000        18.31
Exhaustive-SQ8 (query)                                    18.40       970.06       988.46       0.7864          1.0270         5.15
HNSW-M16-ef100-s50 (query)                               870.73        60.41       931.13       0.9974          1.0001        38.52
HNSW-M16-ef100-s100 (query)                              870.73       108.89       979.61       0.9993          1.0000        38.52
HNSW-M16-ef100-s200 (query)                              870.73       198.84     1_069.56       0.9995          1.0000        38.52
HNSW-M16-ef100 (self)                                    870.73     1_079.15     1_949.88       0.9992          1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_534.75        61.62     1_596.37       0.9979          1.0001        38.52
HNSW-M16-ef200-s100 (query)                            1_534.75       111.61     1_646.35       0.9995          1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_534.75       205.73     1_740.47       0.9995          1.0000        38.52
HNSW-M16-ef200 (self)                                  1_534.75     1_097.53     2_632.28       0.9994          1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_673.44        70.02     1_743.46       0.9988          1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_673.44       124.00     1_797.44       0.9995          1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_673.44       221.56     1_895.00       0.9995          1.0000        47.66
HNSW-M24-ef200 (self)                                  1_673.44     1_206.36     2_879.80       0.9994          1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_697.74        72.86     1_770.59       0.9990          1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_697.74       127.28     1_825.02       0.9995          1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_697.74       227.06     1_924.80       0.9995          1.0000        56.80
HNSW-M32-ef200 (self)                                  1_697.74     1_248.87     2_946.61       0.9994          1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          770.15        36.87       807.02       0.7858          1.1248        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         770.15        71.01       841.16       0.7864          1.0270        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         770.15       128.90       899.05       0.7864          1.0270        26.89
HNSW-SQ8U-M16-ef100 (self)                               770.15       686.42     1_456.57       0.7862          1.0322        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_379.69        37.87     1_417.56       0.7862          1.0271        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_379.69        73.67     1_453.36       0.7864          1.0270        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_379.69       133.60     1_513.29       0.7864          1.0270        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_379.69       687.00     2_066.69       0.7862          1.0286        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_457.00        42.85     1_499.85       0.7863          1.0270        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_457.00        80.41     1_537.41       0.7864          1.0270        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_457.00       142.41     1_599.41       0.7864          1.0270        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_457.00       767.09     2_224.08       0.7862          1.0286        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_523.27        47.03     1_570.30       0.7863          1.0270        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_523.27        82.57     1_605.84       0.7864          1.0270        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_523.27       145.55     1_668.82       0.7864          1.0270        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_523.27       790.76     2_314.03       0.7862          1.0286        45.20
HNSW-SQ8U-drop0 (query)                                1_375.76        77.81     1_453.57       0.7775          1.0292        26.89
HNSW-SQ8U-drop0.001 (query)                            1_376.89        69.74     1_446.63       0.7864          1.0270        26.89
HNSW-SQ8U-drop0.01 (query)                             1_369.93        70.60     1_440.53       0.7791          1.0298        26.89
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.26     1_242.78     1_292.04       1.0000          1.0000        73.24
Exhaustive (self)                                         49.26    12_464.30    12_513.55       1.0000          1.0000        73.24
Exhaustive-SQ8 (query)                                    78.01     1_206.43     1_284.44       0.9341          1.0074        18.88
HNSW-M16-ef100-s50 (query)                             1_403.07        94.12     1_497.19       0.9942          1.0276        93.45
HNSW-M16-ef100-s100 (query)                            1_403.07       155.55     1_558.62       0.9963          1.0131        93.45
HNSW-M16-ef100-s200 (query)                            1_403.07       280.50     1_683.57       0.9975          1.0060        93.45
HNSW-M16-ef100 (self)                                  1_403.07     1_517.52     2_920.59       0.9961          1.0132        93.45
HNSW-M16-ef200-s50 (query)                             2_526.67       105.00     2_631.67       0.9957          1.0346        93.45
HNSW-M16-ef200-s100 (query)                            2_526.67       162.86     2_689.53       0.9977          1.0175        93.45
HNSW-M16-ef200-s200 (query)                            2_526.67       290.99     2_817.66       0.9987          1.0084        93.45
HNSW-M16-ef200 (self)                                  2_526.67     1_592.98     4_119.65       0.9973          1.0187        93.45
HNSW-M24-ef200-s50 (query)                             2_675.64       109.14     2_784.77       0.9980          1.0116       102.59
HNSW-M24-ef200-s100 (query)                            2_675.64       173.40     2_849.04       0.9991          1.0049       102.59
HNSW-M24-ef200-s200 (query)                            2_675.64       304.51     2_980.15       0.9995          1.0025       102.59
HNSW-M24-ef200 (self)                                  2_675.64     1_686.53     4_362.16       0.9989          1.0056       102.59
HNSW-M32-ef200-s50 (query)                             2_753.45       104.60     2_858.05       0.9983          1.0125       111.73
HNSW-M32-ef200-s100 (query)                            2_753.45       177.37     2_930.83       0.9990          1.0075       111.73
HNSW-M32-ef200-s200 (query)                            2_753.45       309.80     3_063.25       0.9997          1.0014       111.73
HNSW-M32-ef200 (self)                                  2_753.45     1_724.22     4_477.68       0.9990          1.0059       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          804.07        38.21       842.28       0.9285          1.0363        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         804.07        67.55       871.62       0.9313          1.0188        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         804.07       124.26       928.33       0.9325          1.0123        40.63
HNSW-SQ8U-M16-ef100 (self)                               804.07       625.69     1_429.76       0.9305          1.0223        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_447.88        39.64     1_487.52       0.9301          1.0354        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_447.88        71.06     1_518.94       0.9315          1.0237        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_447.88       130.23     1_578.11       0.9329          1.0138        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_447.88       687.80     2_135.68       0.9318          1.0221        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_659.94        42.76     1_702.70       0.9324          1.0180        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_659.94        74.39     1_734.33       0.9334          1.0121        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_659.94       133.54     1_793.48       0.9337          1.0086        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_659.94       716.98     2_376.91       0.9330          1.0132        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_639.75        44.13     1_683.88       0.9329          1.0150        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_639.75        75.91     1_715.66       0.9335          1.0111        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_639.75       134.40     1_774.16       0.9337          1.0091        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_639.75       736.01     2_375.76       0.9333          1.0107        58.94
HNSW-SQ8U-drop0 (query)                                1_453.24        86.25     1_539.50       0.8645          1.0415        40.63
HNSW-SQ8U-drop0.001 (query)                            1_440.19        69.75     1_509.94       0.9332          1.0120        40.63
HNSW-SQ8U-drop0.01 (query)                             1_438.94        68.55     1_507.49       0.9320          1.0389        40.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        50.96     1_273.91     1_324.87       1.0000          1.0000        73.81
Exhaustive (self)                                         50.96    12_847.01    12_897.97       1.0000          1.0000        73.81
Exhaustive-SQ8 (query)                                    94.84     1_242.34     1_337.19       0.6675          1.3471        18.88
HNSW-M16-ef100-s50 (query)                             1_279.89        78.77     1_358.66       0.9932          1.1190        94.02
HNSW-M16-ef100-s100 (query)                            1_279.89       135.25     1_415.14       0.9966          1.0363        94.02
HNSW-M16-ef100-s200 (query)                            1_279.89       242.41     1_522.30       0.9979          1.0140        94.02
HNSW-M16-ef100 (self)                                  1_279.89     1_320.49     2_600.38       0.9962          1.0412        94.02
HNSW-M16-ef200-s50 (query)                             2_355.80        91.61     2_447.41       0.9932          1.1422        94.02
HNSW-M16-ef200-s100 (query)                            2_355.80       142.68     2_498.48       0.9965          1.0567        94.02
HNSW-M16-ef200-s200 (query)                            2_355.80       254.48     2_610.28       0.9980          1.0188        94.02
HNSW-M16-ef200 (self)                                  2_355.80     1_384.33     3_740.13       0.9959          1.0751        94.02
HNSW-M24-ef200-s50 (query)                             2_475.62       100.67     2_576.30       0.9972          1.0449       103.16
HNSW-M24-ef200-s100 (query)                            2_475.62       155.88     2_631.51       0.9986          1.0220       103.16
HNSW-M24-ef200-s200 (query)                            2_475.62       270.93     2_746.55       0.9993          1.0106       103.16
HNSW-M24-ef200 (self)                                  2_475.62     1_462.25     3_937.88       0.9987          1.0166       103.16
HNSW-M32-ef200-s50 (query)                             2_604.27        91.12     2_695.39       0.9952          1.1329       112.31
HNSW-M32-ef200-s100 (query)                            2_604.27       155.92     2_760.19       0.9978          1.0596       112.31
HNSW-M32-ef200-s200 (query)                            2_604.27       275.71     2_879.99       0.9995          1.0076       112.31
HNSW-M32-ef200 (self)                                  2_604.27     1_585.66     4_189.94       0.9980          1.0512       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          771.21        37.04       808.25       0.6640          1.4072        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         771.21        60.99       832.20       0.6657          1.3674        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         771.21       110.59       881.80       0.6665          1.3561        40.63
HNSW-SQ8U-M16-ef100 (self)                               771.21       582.08     1_353.29       0.6650          1.3764        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_410.77        40.36     1_451.13       0.6637          1.4261        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_410.77        67.31     1_478.08       0.6655          1.3856        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_410.77       122.87     1_533.64       0.6663          1.3667        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_410.77       629.53     2_040.29       0.6646          1.3973        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_539.14        44.75     1_583.90       0.6649          1.4273        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_539.14        66.44     1_605.58       0.6665          1.3760        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_539.14       118.83     1_657.98       0.6674          1.3516        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_539.14       639.68     2_178.82       0.6661          1.3816        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_621.05        43.51     1_664.56       0.6651          1.4228        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_621.05        73.96     1_695.01       0.6662          1.3911        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_621.05       128.37     1_749.42       0.6672          1.3533        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_621.05       694.23     2_315.28       0.6659          1.3912        58.94
HNSW-SQ8U-drop0 (query)                                1_428.84        66.06     1_494.90       0.6192          1.5300        40.63
HNSW-SQ8U-drop0.001 (query)                            1_386.01        65.27     1_451.28       0.6650          1.3902        40.63
HNSW-SQ8U-drop0.01 (query)                             1_410.79        64.72     1_475.52       0.6787          1.3354        40.63
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.27       689.62       721.89       1.0000          1.0000        48.83
Exhaustive (self)                                         32.27     2_225.77     2_258.04       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                                605.94       656.29     1_262.23       0.1811          1.1573         1.01
Exhaustive-PQ-m16 (self)                                 605.94     2_158.46     2_764.41       0.1575          1.1716         1.01
Exhaustive-PQ-m32 (query)                              1_026.90     1_491.36     2_518.26       0.2138          1.1452         1.78
Exhaustive-PQ-m32 (self)                               1_026.90     4_998.15     6_025.05       0.1773          1.1611         1.78
Exhaustive-PQ-m64 (query)                              1_765.57     3_509.66     5_275.23       0.2945          1.1101         3.30
Exhaustive-PQ-m64 (self)                               1_765.57    11_696.92    13_462.49       0.2455          1.1228         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_472.58       196.21     1_668.78       0.2994          1.0923         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_472.58       317.18     1_789.76       0.2995          1.0923         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_472.58       450.89     1_923.47       0.2995          1.0923         1.17
IVF-PQ-nl158-m16 (self)                                1_472.58     1_504.12     2_976.69       0.2099          1.1316         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_889.75       355.85     2_245.60       0.4313          1.0548         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_889.75       589.57     2_479.32       0.4315          1.0548         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_889.75       825.51     2_715.26       0.4315          1.0548         1.93
IVF-PQ-nl158-m32 (self)                                1_889.75     2_734.85     4_624.60       0.3428          1.0771         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_454.80       635.64     3_090.44       0.6703          1.0156         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_454.80     1_105.21     3_560.01       0.6708          1.0156         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_454.80     1_511.56     3_966.36       0.6708          1.0156         3.46
IVF-PQ-nl158-m64 (self)                                2_454.80     5_072.43     7_527.23       0.6120          1.0222         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_085.24       289.52     1_374.76       0.3078          1.0891         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_085.24       357.81     1_443.05       0.3078          1.0891         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_085.24       546.39     1_631.63       0.3078          1.0891         1.23
IVF-PQ-nl223-m16 (self)                                1_085.24     1_782.92     2_868.16       0.2131          1.1293         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_505.22       515.67     2_020.89       0.4356          1.0532         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_505.22       647.17     2_152.39       0.4356          1.0532         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_505.22       964.19     2_469.41       0.4356          1.0532         2.00
IVF-PQ-nl223-m32 (self)                                1_505.22     3_190.92     4_696.14       0.3447          1.0763         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_042.97       909.31     2_952.29       0.6762          1.0150         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_042.97     1_158.08     3_201.05       0.6762          1.0150         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_042.97     1_738.11     3_781.09       0.6762          1.0150         3.52
IVF-PQ-nl223-m64 (self)                                2_042.97     5_805.34     7_848.32       0.6162          1.0216         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_360.93       370.15     1_731.08       0.3105          1.0872         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_360.93       414.88     1_775.81       0.3105          1.0872         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_360.93       602.62     1_963.55       0.3105          1.0872         1.32
IVF-PQ-nl316-m16 (self)                                1_360.93     1_985.57     3_346.49       0.2149          1.1274         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_761.98       675.40     2_437.39       0.4410          1.0517         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_761.98       752.73     2_514.71       0.4410          1.0517         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_761.98     1_097.95     2_859.93       0.4410          1.0517         2.09
IVF-PQ-nl316-m32 (self)                                1_761.98     3_589.33     5_351.31       0.3478          1.0750         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_304.15     1_158.20     3_462.35       0.6787          1.0147         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_304.15     1_309.62     3_613.77       0.6787          1.0147         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_304.15     1_913.19     4_217.35       0.6787          1.0147         3.61
IVF-PQ-nl316-m64 (self)                                2_304.15     6_355.82     8_659.98       0.6182          1.0213         3.61
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
Exhaustive (query)                                        68.31     1_222.77     1_291.08       1.0000          1.0000        97.66
Exhaustive (self)                                         68.31     4_237.81     4_306.12       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                                842.55       685.26     1_527.81       0.1437          1.1190         1.26
Exhaustive-PQ-m16 (self)                                 842.55     2_204.76     3_047.31       0.1309          1.1260         1.26
Exhaustive-PQ-m32 (query)                              1_205.56     1_527.45     2_733.01       0.1563          1.1194         2.03
Exhaustive-PQ-m32 (self)                               1_205.56     4_958.45     6_164.01       0.1377          1.1255         2.03
Exhaustive-PQ-m64 (query)                              2_044.91     3_531.37     5_576.28       0.1896          1.1090         3.55
Exhaustive-PQ-m64 (self)                               2_044.91    11_703.61    13_748.52       0.1564          1.1171         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_502.36       272.12     2_774.48       0.2137          1.0836         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_502.36       433.56     2_935.93       0.2137          1.0836         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_502.36       600.61     3_102.97       0.2137          1.0836         1.57
IVF-PQ-nl158-m16 (self)                                2_502.36     2_010.96     4_513.33       0.1474          1.1129         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_888.33       408.78     3_297.11       0.2788          1.0654         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_888.33       675.46     3_563.79       0.2788          1.0654         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_888.33       919.91     3_808.24       0.2788          1.0654         2.34
IVF-PQ-nl158-m32 (self)                                2_888.33     3_086.43     5_974.76       0.1891          1.0934         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_709.58       748.81     4_458.39       0.4072          1.0389         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_709.58     1_237.29     4_946.87       0.4072          1.0389         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_709.58     1_731.43     5_441.01       0.4072          1.0389         3.86
IVF-PQ-nl158-m64 (self)                                3_709.58     5_679.00     9_388.58       0.3196          1.0548         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_755.98       404.28     2_160.26       0.2181          1.0818         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_755.98       493.37     2_249.35       0.2181          1.0818         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_755.98       720.48     2_476.46       0.2181          1.0818         1.70
IVF-PQ-nl223-m16 (self)                                1_755.98     2_377.05     4_133.03       0.1500          1.1109         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_175.64       586.35     2_761.98       0.2816          1.0638         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_175.64       724.87     2_900.50       0.2816          1.0638         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_175.64     1_067.93     3_243.57       0.2816          1.0638         2.46
IVF-PQ-nl223-m32 (self)                                2_175.64     3_511.85     5_687.49       0.1913          1.0920         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_966.95     1_056.52     4_023.47       0.4129          1.0379         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_966.95     1_346.21     4_313.16       0.4129          1.0379         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_966.95     1_996.44     4_963.39       0.4129          1.0379         3.99
IVF-PQ-nl223-m64 (self)                                2_966.95     6_576.33     9_543.29       0.3233          1.0540         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_081.89       505.26     2_587.16       0.2198          1.0807         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_081.89       562.14     2_644.03       0.2198          1.0807         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_081.89       806.81     2_888.70       0.2198          1.0807         1.88
IVF-PQ-nl316-m16 (self)                                2_081.89     2_664.11     4_746.00       0.1520          1.1094         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_443.90       741.73     3_185.63       0.2847          1.0627         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_443.90       840.39     3_284.29       0.2847          1.0627         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_443.90     1_215.55     3_659.45       0.2847          1.0627         2.65
IVF-PQ-nl316-m32 (self)                                2_443.90     3_954.89     6_398.79       0.1931          1.0909         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_306.98     1_397.51     4_704.49       0.4156          1.0374         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_306.98     1_578.02     4_885.00       0.4156          1.0374         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_306.98     2_292.57     5_599.55       0.4156          1.0374         4.17
IVF-PQ-nl316-m64 (self)                                3_306.98     7_678.08    10_985.06       0.3246          1.0535         4.17
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
Exhaustive (query)                                        99.12     1_733.51     1_832.62       1.0000          1.0000       146.48
Exhaustive (self)                                         99.12     5_802.50     5_901.61       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_139.44       691.27     1_830.71       0.1331          1.0996         1.51
Exhaustive-PQ-m16 (self)                               1_139.44     2_346.39     3_485.83       0.1234          1.1038         1.51
Exhaustive-PQ-m32 (query)                              1_561.64     1_528.36     3_090.00       0.1399          1.1013         2.28
Exhaustive-PQ-m32 (self)                               1_561.64     4_991.82     6_553.46       0.1267          1.1047         2.28
Exhaustive-PQ-m64 (query)                              2_438.12     3_550.37     5_988.50       0.1591          1.0964         3.80
Exhaustive-PQ-m64 (self)                               2_438.12    11_738.90    14_177.02       0.1361          1.1021         3.80
Exhaustive-PQ-m128 (query)                             4_231.84     7_843.73    12_075.57       0.2067          1.0804         6.86
Exhaustive-PQ-m128 (self)                              4_231.84    25_990.65    30_222.49       0.1673          1.0885         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_587.62       356.12     3_943.75       0.1824          1.0758         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_587.62       564.07     4_151.69       0.1824          1.0758         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_587.62       787.76     4_375.38       0.1824          1.0758         1.98
IVF-PQ-nl158-m16 (self)                                3_587.62     2_624.65     6_212.27       0.1306          1.0974         1.98
IVF-PQ-nl158-m32-np7 (query)                           4_076.61       533.17     4_609.78       0.2299          1.0626         2.74
IVF-PQ-nl158-m32-np12 (query)                          4_076.61       899.73     4_976.34       0.2299          1.0626         2.74
IVF-PQ-nl158-m32-np17 (query)                          4_076.61     1_209.93     5_286.53       0.2299          1.0626         2.74
IVF-PQ-nl158-m32 (self)                                4_076.61     3_970.60     8_047.21       0.1525          1.0875         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_894.50       827.91     5_722.41       0.3062          1.0462         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_894.50     1_380.56     6_275.07       0.3062          1.0462         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_894.50     1_932.77     6_827.28       0.3062          1.0462         4.27
IVF-PQ-nl158-m64 (self)                                4_894.50     6_375.71    11_270.21       0.2162          1.0658         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_650.68     1_610.59     8_261.26       0.4918          1.0211         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_650.68     2_689.15     9_339.83       0.4920          1.0211         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_650.68     3_812.99    10_463.67       0.4920          1.0211         7.32
IVF-PQ-nl158-m128 (self)                               6_650.68    12_527.43    19_178.10       0.4136          1.0295         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_450.60       499.77     2_950.37       0.1860          1.0743         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_450.60       613.72     3_064.32       0.1860          1.0743         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_450.60       905.28     3_355.88       0.1860          1.0743         2.17
IVF-PQ-nl223-m16 (self)                                2_450.60     2_954.28     5_404.89       0.1331          1.0955         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_994.67       789.26     3_783.93       0.2302          1.0618         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_994.67       929.80     3_924.48       0.2302          1.0618         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_994.67     1_366.12     4_360.79       0.2302          1.0618         2.93
IVF-PQ-nl223-m32 (self)                                2_994.67     4_501.22     7_495.89       0.1537          1.0861         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_820.51     1_188.48     5_008.99       0.3115          1.0451         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_820.51     1_562.48     5_382.99       0.3115          1.0451         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_820.51     2_221.78     6_042.28       0.3115          1.0451         4.46
IVF-PQ-nl223-m64 (self)                                3_820.51     7_297.55    11_118.06       0.2190          1.0648         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_663.86     2_326.93     7_990.79       0.4968          1.0206         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_663.86     2_915.72     8_579.58       0.4968          1.0206         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_663.86     4_323.94     9_987.80       0.4968          1.0206         7.51
IVF-PQ-nl223-m128 (self)                               5_663.86    14_353.36    20_017.22       0.4171          1.0291         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_087.60       685.68     3_773.28       0.1886          1.0732         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_087.60       745.42     3_833.02       0.1886          1.0732         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_087.60     1_067.36     4_154.96       0.1886          1.0732         2.44
IVF-PQ-nl316-m16 (self)                                3_087.60     3_531.09     6_618.69       0.1342          1.0944         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_592.29       989.78     4_582.07       0.2338          1.0605         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_592.29     1_107.69     4_699.98       0.2338          1.0605         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_592.29     1_590.73     5_183.02       0.2338          1.0605         3.21
IVF-PQ-nl316-m32 (self)                                3_592.29     5_266.87     8_859.15       0.1556          1.0849         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_419.37     1_553.88     5_973.25       0.3158          1.0439         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_419.37     1_736.44     6_155.81       0.3158          1.0439         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_419.37     2_538.75     6_958.12       0.3158          1.0439         4.73
IVF-PQ-nl316-m64 (self)                                4_419.37     8_382.59    12_801.96       0.2213          1.0639         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_237.41     2_999.72     9_237.12       0.5004          1.0202         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_237.41     3_403.80     9_641.20       0.5005          1.0202         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_237.41     4_930.97    11_168.37       0.5005          1.0202         7.78
IVF-PQ-nl316-m128 (self)                               6_237.41    17_396.36    23_633.77       0.4191          1.0287         7.78
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
Exhaustive (query)                                        32.26       677.35       709.61       1.0000          1.0000        48.83
Exhaustive (self)                                         32.26     2_265.80     2_298.06       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                                631.58       654.40     1_285.98       0.2963          1.2496         1.01
Exhaustive-PQ-m16 (self)                                 631.58     2_174.26     2_805.84       0.2324          1.3821         1.01
Exhaustive-PQ-m32 (query)                              1_071.66     1_486.07     2_557.73       0.4045          1.1613         1.78
Exhaustive-PQ-m32 (self)                               1_071.66     4_928.07     5_999.73       0.3204          1.2654         1.78
Exhaustive-PQ-m64 (query)                              1_688.65     3_514.11     5_202.76       0.5368          1.0876         3.30
Exhaustive-PQ-m64 (self)                               1_688.65    11_643.61    13_332.26       0.4607          1.1458         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_439.32       194.39     1_633.71       0.5290          1.0891         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_439.32       300.23     1_739.55       0.5290          1.0891         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_439.32       417.21     1_856.53       0.5290          1.0891         1.17
IVF-PQ-nl158-m16 (self)                                1_439.32     1_367.46     2_806.78       0.4271          1.1645         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_851.57       345.85     2_197.42       0.6697          1.0403         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_851.57       566.81     2_418.37       0.6697          1.0403         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_851.57       750.72     2_602.29       0.6697          1.0403         1.93
IVF-PQ-nl158-m32 (self)                                1_851.57     2_593.41     4_444.98       0.6070          1.0684         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_391.73       616.98     3_008.71       0.8318          1.0095         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_391.73       977.82     3_369.55       0.8318          1.0095         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_391.73     1_354.92     3_746.65       0.8318          1.0095         3.46
IVF-PQ-nl158-m64 (self)                                2_391.73     4_541.47     6_933.19       0.7985          1.0163         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_200.08       284.98     1_485.05       0.5325          1.0877         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_200.08       349.13     1_549.21       0.5326          1.0877         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_200.08       512.07     1_712.15       0.5326          1.0877         1.23
IVF-PQ-nl223-m16 (self)                                1_200.08     1_708.72     2_908.79       0.4210          1.1695         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_624.52       507.49     2_132.01       0.6722          1.0394         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_624.52       625.62     2_250.14       0.6724          1.0394         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_624.52       936.53     2_561.06       0.6724          1.0394         2.00
IVF-PQ-nl223-m32 (self)                                1_624.52     3_054.54     4_679.06       0.6051          1.0691         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_152.06       895.62     3_047.68       0.8348          1.0091         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_152.06     1_110.39     3_262.45       0.8352          1.0090         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_152.06     1_646.81     3_798.87       0.8352          1.0090         3.52
IVF-PQ-nl223-m64 (self)                                2_152.06     5_764.36     7_916.42       0.8012          1.0158         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_412.57       364.61     1_777.17       0.5293          1.0886         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_412.57       407.27     1_819.84       0.5293          1.0886         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_412.57       581.77     1_994.34       0.5293          1.0886         1.32
IVF-PQ-nl316-m16 (self)                                1_412.57     1_929.75     3_342.32       0.4124          1.1751         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_822.92       655.51     2_478.43       0.6746          1.0390         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_822.92       732.09     2_555.01       0.6747          1.0390         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_822.92     1_069.04     2_891.95       0.6747          1.0390         2.09
IVF-PQ-nl316-m32 (self)                                1_822.92     3_507.77     5_330.69       0.6024          1.0702         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_352.54     1_143.04     3_495.58       0.8369          1.0089         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_352.54     1_273.76     3_626.30       0.8370          1.0088         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_352.54     1_839.66     4_192.20       0.8371          1.0088         3.61
IVF-PQ-nl316-m64 (self)                                2_352.54     6_142.96     8_495.50       0.8027          1.0155         3.61
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
Exhaustive (query)                                        68.91     1_276.99     1_345.91       1.0000          1.0000        97.66
Exhaustive (self)                                         68.91     4_090.28     4_159.19       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                                945.70       672.42     1_618.11       0.2165          1.2233         1.26
Exhaustive-PQ-m16 (self)                                 945.70     2_196.25     3_141.95       0.1776          1.3091         1.26
Exhaustive-PQ-m32 (query)                              1_217.50     1_507.32     2_724.83       0.2863          1.1669         2.03
Exhaustive-PQ-m32 (self)                               1_217.50     4_960.05     6_177.55       0.2244          1.2482         2.03
Exhaustive-PQ-m64 (query)                              2_037.49     3_542.95     5_580.44       0.3812          1.1141         3.55
Exhaustive-PQ-m64 (self)                               2_037.49    11_704.19    13_741.67       0.3012          1.1804         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_198.61       264.60     2_463.21       0.3741          1.1188         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_198.61       411.33     2_609.94       0.3741          1.1188         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_198.61       560.80     2_759.41       0.3741          1.1188         1.57
IVF-PQ-nl158-m16 (self)                                2_198.61     1_865.08     4_063.69       0.2675          1.2100         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_665.38       388.07     3_053.45       0.4850          1.0727         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_665.38       603.17     3_268.55       0.4850          1.0727         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_665.38       836.82     3_502.20       0.4850          1.0727         2.34
IVF-PQ-nl158-m32 (self)                                2_665.38     2_682.57     5_347.95       0.3900          1.1244         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_452.67       701.80     4_154.47       0.6267          1.0349         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_452.67     1_095.90     4_548.57       0.6267          1.0349         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_452.67     1_500.34     4_953.02       0.6267          1.0349         3.86
IVF-PQ-nl158-m64 (self)                                3_452.67     4_948.05     8_400.72       0.5759          1.0530         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_896.48       421.99     2_318.47       0.3724          1.1189         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_896.48       485.00     2_381.48       0.3724          1.1189         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_896.48       703.76     2_600.24       0.3724          1.1189         1.70
IVF-PQ-nl223-m16 (self)                                1_896.48     2_328.62     4_225.10       0.2586          1.2183         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_243.45       554.42     2_797.87       0.4832          1.0732         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_243.45       680.16     2_923.62       0.4832          1.0732         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_243.45     1_000.06     3_243.51       0.4832          1.0732         2.46
IVF-PQ-nl223-m32 (self)                                2_243.45     3_258.79     5_502.24       0.3779          1.1312         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_086.90     1_022.17     4_109.07       0.6294          1.0346         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_086.90     1_250.04     4_336.94       0.6294          1.0346         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_086.90     1_827.13     4_914.04       0.6294          1.0346         3.99
IVF-PQ-nl223-m64 (self)                                3_086.90     6_012.05     9_098.96       0.5713          1.0547         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_269.27       510.53     2_779.81       0.3695          1.1197         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_269.27       564.93     2_834.21       0.3695          1.1197         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_269.27       804.41     3_073.69       0.3695          1.1197         1.88
IVF-PQ-nl316-m16 (self)                                2_269.27     2_652.29     4_921.57       0.2528          1.2222         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_654.03       721.71     3_375.74       0.4846          1.0727         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_654.03       801.68     3_455.72       0.4846          1.0727         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_654.03     1_155.85     3_809.88       0.4846          1.0727         2.65
IVF-PQ-nl316-m32 (self)                                2_654.03     3_759.85     6_413.88       0.3705          1.1349         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_470.40     1_334.96     4_805.36       0.6315          1.0340         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_470.40     1_486.23     4_956.63       0.6315          1.0340         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_470.40     2_128.09     5_598.49       0.6315          1.0340         4.17
IVF-PQ-nl316-m64 (self)                                3_470.40     7_052.71    10_523.10       0.5691          1.0551         4.17
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
Exhaustive (query)                                        99.13     1_773.28     1_872.42       1.0000          1.0000       146.48
Exhaustive (self)                                         99.13     5_808.38     5_907.51       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_144.86       687.90     1_832.77       0.2118          1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_144.86     2_217.51     3_362.37       0.1771          1.3071         1.51
Exhaustive-PQ-m32 (query)                              1_562.06     1_529.70     3_091.77       0.2751          1.1626         2.28
Exhaustive-PQ-m32 (self)                               1_562.06     5_045.74     6_607.80       0.2200          1.2508         2.28
Exhaustive-PQ-m64 (query)                              2_457.28     3_560.14     6_017.42       0.3604          1.1161         3.80
Exhaustive-PQ-m64 (self)                               2_457.28    11_844.17    14_301.45       0.2900          1.1870         3.80
Exhaustive-PQ-m128 (query)                             4_246.49     7_798.02    12_044.51       0.4602          1.0761         6.86
Exhaustive-PQ-m128 (self)                              4_246.49    26_056.81    30_303.30       0.3920          1.1232         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_120.40       351.62     3_472.02       0.3589          1.1169         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_120.40       537.68     3_658.08       0.3589          1.1169         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_120.40       737.78     3_858.18       0.3589          1.1169         1.98
IVF-PQ-nl158-m16 (self)                                3_120.40     2_443.24     5_563.64       0.2569          1.2176         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_600.66       510.61     4_111.27       0.4585          1.0766         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_600.66       804.80     4_405.46       0.4585          1.0766         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_600.66     1_108.50     4_709.16       0.4585          1.0766         2.74
IVF-PQ-nl158-m32 (self)                                3_600.66     3_611.46     7_212.12       0.3638          1.1385         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_477.90       798.55     5_276.44       0.5669          1.0455         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_477.90     1_262.15     5_740.04       0.5669          1.0455         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_477.90     1_729.58     6_207.48       0.5669          1.0455         4.27
IVF-PQ-nl158-m64 (self)                                4_477.90     5_736.22    10_214.11       0.5133          1.0722         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_242.70     1_556.65     7_799.35       0.7348          1.0160         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_242.70     2_459.89     8_702.59       0.7348          1.0160         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_242.70     3_625.67     9_868.37       0.7348          1.0160         7.32
IVF-PQ-nl158-m128 (self)                               6_242.70    11_156.60    17_399.29       0.7158          1.0233         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_631.11       489.27     3_120.38       0.3537          1.1186         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_631.11       600.61     3_231.72       0.3537          1.1186         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_631.11       875.05     3_506.16       0.3537          1.1186         2.17
IVF-PQ-nl223-m16 (self)                                2_631.11     2_866.11     5_497.22       0.2432          1.2305         2.17
IVF-PQ-nl223-m32-np11 (query)                          3_091.25       716.73     3_807.99       0.4493          1.0786         2.93
IVF-PQ-nl223-m32-np14 (query)                          3_091.25       890.54     3_981.79       0.4493          1.0786         2.93
IVF-PQ-nl223-m32-np21 (query)                          3_091.25     1_285.87     4_377.13       0.4493          1.0786         2.93
IVF-PQ-nl223-m32 (self)                                3_091.25     4_232.79     7_324.04       0.3359          1.1543         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_953.86     1_153.32     5_107.19       0.5657          1.0456         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_953.86     1_418.24     5_372.10       0.5657          1.0456         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_953.86     2_069.12     6_022.98       0.5657          1.0456         4.46
IVF-PQ-nl223-m64 (self)                                3_953.86     6_889.79    10_843.65       0.4922          1.0792         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_771.19     2_261.92     8_033.11       0.7377          1.0155         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_771.19     2_827.22     8_598.41       0.7377          1.0155         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_771.19     4_186.16     9_957.36       0.7377          1.0155         7.51
IVF-PQ-nl223-m128 (self)                               5_771.19    13_720.12    19_491.31       0.7123          1.0236         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_139.21       652.63     3_791.84       0.3513          1.1198         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_139.21       728.37     3_867.58       0.3513          1.1198         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_139.21     1_030.98     4_170.19       0.3513          1.1198         2.44
IVF-PQ-nl316-m16 (self)                                3_139.21     3_437.56     6_576.77       0.2365          1.2362         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_583.65       949.56     4_533.21       0.4471          1.0799         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_583.65     1_061.38     4_645.03       0.4471          1.0799         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_583.65     1_521.57     5_105.22       0.4471          1.0799         3.21
IVF-PQ-nl316-m32 (self)                                3_583.65     5_053.69     8_637.33       0.3198          1.1636         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_522.14     1_520.00     6_042.14       0.5661          1.0457         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_522.14     1_700.13     6_222.27       0.5661          1.0457         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_522.14     2_439.19     6_961.33       0.5661          1.0457         4.73
IVF-PQ-nl316-m64 (self)                                4_522.14     8_099.92    12_622.06       0.4795          1.0839         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_265.11     2_962.90     9_228.00       0.7393          1.0152         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_265.11     3_317.82     9_582.93       0.7393          1.0152         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_265.11     4_759.11    11_024.22       0.7393          1.0152         7.78
IVF-PQ-nl316-m128 (self)                               6_265.11    15_823.96    22_089.07       0.7117          1.0236         7.78
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
Exhaustive (query)                                        32.23       685.32       717.55       1.0000          1.0000        48.83
Exhaustive (self)                                         32.23     2_276.12     2_308.34       1.0000          1.0000        48.83
Exhaustive-PQ-m16 (query)                                612.18       659.97     1_272.15       0.7119          1.1576         1.01
Exhaustive-PQ-m16 (self)                                 612.18     2_177.97     2_790.15       0.6210          1.2885         1.01
Exhaustive-PQ-m32 (query)                              1_074.58     1_500.88     2_575.46       0.7717          1.0965         1.78
Exhaustive-PQ-m32 (self)                               1_074.58     4_943.27     6_017.85       0.6993          1.1778         1.78
Exhaustive-PQ-m64 (query)                              1_686.27     3_502.83     5_189.10       0.8251          1.0574         3.30
Exhaustive-PQ-m64 (self)                               1_686.27    11_658.67    13_344.94       0.7675          1.1055         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_472.72       209.42     1_682.14       0.8272          1.0522         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_472.72       336.66     1_809.38       0.8277          1.0518         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_472.72       468.70     1_941.42       0.8277          1.0518         1.17
IVF-PQ-nl158-m16 (self)                                1_472.72     1_553.34     3_026.07       0.7669          1.0989         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_887.64       395.60     2_283.25       0.8746          1.0266         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_887.64       648.34     2_535.99       0.8751          1.0262         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_887.64       891.09     2_778.73       0.8751          1.0262         1.93
IVF-PQ-nl158-m32 (self)                                1_887.64     2_947.21     4_834.86       0.8288          1.0511         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_435.87       709.66     3_145.53       0.9048          1.0151         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_435.87     1_203.84     3_639.72       0.9056          1.0147         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_435.87     1_691.35     4_127.23       0.9056          1.0147         3.46
IVF-PQ-nl158-m64 (self)                                2_435.87     5_643.59     8_079.46       0.8704          1.0288         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_078.38       294.67     1_373.05       0.8428          1.0430         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_078.38       374.46     1_452.84       0.8429          1.0429         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_078.38       531.18     1_609.56       0.8429          1.0429         1.23
IVF-PQ-nl223-m16 (self)                                1_078.38     1_798.42     2_876.80       0.7841          1.0842         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_510.71       526.07     2_036.78       0.8837          1.0224         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_510.71       663.01     2_173.72       0.8838          1.0223         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_510.71       980.14     2_490.84       0.8838          1.0223         2.00
IVF-PQ-nl223-m32 (self)                                1_510.71     3_253.21     4_763.92       0.8403          1.0440         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_035.35       953.51     2_988.86       0.9100          1.0134         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_035.35     1_206.32     3_241.67       0.9102          1.0134         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_035.35     1_796.47     3_831.82       0.9102          1.0133         3.52
IVF-PQ-nl223-m64 (self)                                2_035.35     5_983.67     8_019.02       0.8765          1.0259         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_309.56       373.59     1_683.15       0.8502          1.0391         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_309.56       418.26     1_727.82       0.8502          1.0391         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_309.56       601.04     1_910.60       0.8502          1.0391         1.32
IVF-PQ-nl316-m16 (self)                                1_309.56     2_016.91     3_326.47       0.7922          1.0785         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_729.95       675.91     2_405.86       0.8867          1.0214         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_729.95       759.24     2_489.18       0.8867          1.0214         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_729.95     1_103.13     2_833.08       0.8868          1.0214         2.09
IVF-PQ-nl316-m32 (self)                                1_729.95     3_642.33     5_372.27       0.8425          1.0433         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_241.93     1_199.06     3_440.99       0.9127          1.0125         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_241.93     1_356.33     3_598.26       0.9127          1.0125         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_241.93     1_988.70     4_230.63       0.9127          1.0125         3.61
IVF-PQ-nl316-m64 (self)                                2_241.93     6_678.94     8_920.87       0.8791          1.0247         3.61
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
Exhaustive (query)                                        69.08     1_264.96     1_334.04       1.0000          1.0000        97.66
Exhaustive (self)                                         69.08     3_999.67     4_068.75       1.0000          1.0000        97.66
Exhaustive-PQ-m16 (query)                                839.71       673.03     1_512.74       0.6791          1.1977         1.26
Exhaustive-PQ-m16 (self)                                 839.71     2_184.47     3_024.18       0.5853          1.3494         1.26
Exhaustive-PQ-m32 (query)                              1_224.57     1_509.41     2_733.98       0.7374          1.1283         2.03
Exhaustive-PQ-m32 (self)                               1_224.57     4_959.57     6_184.14       0.6552          1.2348         2.03
Exhaustive-PQ-m64 (query)                              2_029.58     3_520.35     5_549.93       0.7805          1.0879         3.55
Exhaustive-PQ-m64 (self)                               2_029.58    11_760.20    13_789.78       0.7136          1.1583         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_501.81       279.90     2_781.71       0.8455          1.0448         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_501.81       447.60     2_949.41       0.8458          1.0447         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_501.81       620.87     3_122.68       0.8458          1.0447         1.57
IVF-PQ-nl158-m16 (self)                                2_501.81     2_047.70     4_549.51       0.7844          1.0913         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_887.09       423.59     3_310.68       0.8726          1.0297         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_887.09       689.59     3_576.69       0.8731          1.0294         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_887.09       963.61     3_850.71       0.8731          1.0294         2.34
IVF-PQ-nl158-m32 (self)                                2_887.09     3_146.40     6_033.50       0.8208          1.0615         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_693.74       788.72     4_482.46       0.8936          1.0202         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_693.74     1_331.08     5_024.82       0.8941          1.0200         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_693.74     1_848.55     5_542.29       0.8941          1.0200         3.86
IVF-PQ-nl158-m64 (self)                                3_693.74     6_434.89    10_128.63       0.8494          1.0420         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_994.06       480.09     2_474.15       0.8543          1.0397         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_994.06       525.97     2_520.03       0.8543          1.0397         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_994.06       742.82     2_736.88       0.8544          1.0397         1.70
IVF-PQ-nl223-m16 (self)                                1_994.06     2_452.84     4_446.90       0.7965          1.0807         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_005.39       602.68     2_608.07       0.8795          1.0266         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_005.39       775.52     2_780.91       0.8795          1.0265         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_005.39     1_104.51     3_109.89       0.8795          1.0265         2.46
IVF-PQ-nl223-m32 (self)                                2_005.39     3_585.51     5_590.90       0.8306          1.0550         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_855.32     1_087.72     3_943.04       0.9003          1.0178         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_855.32     1_377.55     4_232.87       0.9003          1.0178         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_855.32     2_038.05     4_893.36       0.9003          1.0178         3.99
IVF-PQ-nl223-m64 (self)                                2_855.32     6_817.41     9_672.73       0.8568          1.0378         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_883.11       554.43     2_437.54       0.8694          1.0319         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_883.11       584.86     2_467.96       0.8694          1.0319         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_883.11       851.10     2_734.21       0.8694          1.0319         1.88
IVF-PQ-nl316-m16 (self)                                1_883.11     2_759.40     4_642.51       0.8149          1.0655         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_252.19       751.69     3_003.89       0.8917          1.0215         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_252.19       846.53     3_098.72       0.8917          1.0215         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_252.19     1_230.41     3_482.60       0.8917          1.0214         2.65
IVF-PQ-nl316-m32 (self)                                2_252.19     4_018.67     6_270.86       0.8455          1.0452         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_097.77     1_407.08     4_504.85       0.9064          1.0155         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_097.77     1_573.73     4_671.50       0.9064          1.0155         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_097.77     2_329.75     5_427.52       0.9065          1.0155         4.17
IVF-PQ-nl316-m64 (self)                                3_097.77     7_592.92    10_690.69       0.8655          1.0331         4.17
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
Exhaustive (query)                                        99.16     1_748.41     1_847.58       1.0000          1.0000       146.48
Exhaustive (self)                                         99.16     5_777.75     5_876.91       1.0000          1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_140.33       696.79     1_837.12       0.6502          1.2419         1.51
Exhaustive-PQ-m16 (self)                               1_140.33     2_248.48     3_388.81       0.5522          1.4109         1.51
Exhaustive-PQ-m32 (query)                              1_560.91     1_528.57     3_089.48       0.7657          1.0989         2.28
Exhaustive-PQ-m32 (self)                               1_560.91     5_026.10     6_587.01       0.6925          1.1782         2.28
Exhaustive-PQ-m64 (query)                              2_489.28     3_616.11     6_105.39       0.8202          1.0558         3.80
Exhaustive-PQ-m64 (self)                               2_489.28    11_774.38    14_263.65       0.7633          1.1010         3.80
Exhaustive-PQ-m128 (query)                             4_247.46     7_776.80    12_024.25       0.8668          1.0289         6.86
Exhaustive-PQ-m128 (self)                              4_247.46    25_816.35    30_063.81       0.8261          1.0515         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_495.32       372.92     3_868.24       0.8522          1.0420         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_495.32       600.94     4_096.26       0.8524          1.0420         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_495.32       831.23     4_326.54       0.8524          1.0420         1.98
IVF-PQ-nl158-m16 (self)                                3_495.32     2_704.22     6_199.54       0.7910          1.0835         1.98
IVF-PQ-nl158-m32-np7 (query)                           4_168.64       565.96     4_734.60       0.9000          1.0207         2.74
IVF-PQ-nl158-m32-np12 (query)                          4_168.64       898.03     5_066.67       0.9001          1.0206         2.74
IVF-PQ-nl158-m32-np17 (query)                          4_168.64     1_252.63     5_421.26       0.9001          1.0206         2.74
IVF-PQ-nl158-m32 (self)                                4_168.64     4_106.79     8_275.43       0.8546          1.0436         2.74
IVF-PQ-nl158-m64-np7 (query)                           5_365.27       913.18     6_278.45       0.9202          1.0131         4.27
IVF-PQ-nl158-m64-np12 (query)                          5_365.27     1_483.79     6_849.06       0.9204          1.0130         4.27
IVF-PQ-nl158-m64-np17 (query)                          5_365.27     2_067.00     7_432.27       0.9204          1.0130         4.27
IVF-PQ-nl158-m64 (self)                                5_365.27     6_834.23    12_199.50       0.8831          1.0284         4.27
IVF-PQ-nl158-m128-np7 (query)                          7_036.63     1_775.35     8_811.98       0.9393          1.0072         7.32
IVF-PQ-nl158-m128-np12 (query)                         7_036.63     2_977.93    10_014.57       0.9395          1.0071         7.32
IVF-PQ-nl158-m128-np17 (query)                         7_036.63     4_146.41    11_183.04       0.9395          1.0071         7.32
IVF-PQ-nl158-m128 (self)                               7_036.63    13_787.06    20_823.69       0.9071          1.0171         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_317.84       518.27     2_836.10       0.8627          1.0359         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_317.84       640.16     2_958.00       0.8627          1.0359         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_317.84       928.56     3_246.40       0.8627          1.0359         2.17
IVF-PQ-nl223-m16 (self)                                2_317.84     3_075.49     5_393.32       0.8061          1.0703         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_811.48       772.28     3_583.76       0.9087          1.0172         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_811.48       967.54     3_779.02       0.9088          1.0172         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_811.48     1_455.20     4_266.68       0.9088          1.0172         2.93
IVF-PQ-nl223-m32 (self)                                2_811.48     4_698.90     7_510.38       0.8678          1.0351         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_733.53     1_237.70     4_971.23       0.9272          1.0111         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_733.53     1_572.82     5_306.35       0.9272          1.0111         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_733.53     2_343.04     6_076.57       0.9273          1.0111         4.46
IVF-PQ-nl223-m64 (self)                                3_733.53     7_664.62    11_398.14       0.8925          1.0236         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_597.11     2_445.65     8_042.75       0.9439          1.0059         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_597.11     3_075.60     8_672.70       0.9440          1.0059         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_597.11     4_579.75    10_176.85       0.9440          1.0059         7.51
IVF-PQ-nl223-m128 (self)                               5_597.11    15_246.61    20_843.72       0.9133          1.0148         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_629.23       691.96     3_321.20       0.8689          1.0325         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_629.23       757.17     3_386.41       0.8689          1.0325         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_629.23     1_100.75     3_729.99       0.8689          1.0325         2.44
IVF-PQ-nl316-m16 (self)                                2_629.23     3_631.39     6_260.62       0.8141          1.0644         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_094.31     1_004.54     4_098.86       0.9129          1.0154         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_094.31     1_130.74     4_225.06       0.9129          1.0154         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_094.31     1_639.22     4_733.54       0.9129          1.0154         3.21
IVF-PQ-nl316-m32 (self)                                3_094.31     5_388.93     8_483.24       0.8726          1.0331         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_956.96     1_592.58     5_549.55       0.9302          1.0099         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_956.96     1_790.83     5_747.80       0.9302          1.0099         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_956.96     2_603.14     6_560.11       0.9302          1.0099         4.73
IVF-PQ-nl316-m64 (self)                                3_956.96     8_644.97    12_601.93       0.8964          1.0221         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_764.88     3_116.68     8_881.56       0.9458          1.0055         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_764.88     3_898.87     9_663.75       0.9458          1.0055         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_764.88     5_287.82    11_052.70       0.9458          1.0055         7.78
IVF-PQ-nl316-m128 (self)                               5_764.88    17_350.28    23_115.17       0.9164          1.0139         7.78
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 50k samples, 256D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.38       662.23       694.61       1.0000          1.0000        48.83
Exhaustive (self)                                         32.38     2_169.72     2_202.11       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_478.46       729.89     4_208.36       0.2179          1.1278         1.26
Exhaustive-OPQ-m16 (self)                              3_478.46     2_695.55     6_174.01       0.1873          1.1443         1.26
Exhaustive-OPQ-m32 (query)                             5_772.82     1_543.18     7_316.00       0.2519          1.1149         2.03
Exhaustive-OPQ-m32 (self)                              5_772.82     5_428.00    11_200.82       0.2046          1.1356         2.03
Exhaustive-OPQ-m64 (query)                             8_743.03     3_645.04    12_388.07       0.3180          1.0951         3.55
Exhaustive-OPQ-m64 (self)                              8_743.03    12_560.18    21_303.20       0.2611          1.1115         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_935.24       271.25     5_206.49       0.3096          1.0881         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_935.24       385.31     5_320.55       0.3096          1.0881         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_935.24       508.10     5_443.33       0.3096          1.0881         1.67
IVF-OPQ-nl158-m16 (self)                               4_935.24     2_040.52     6_975.75       0.2194          1.1256         1.67
IVF-OPQ-nl158-m32-np7 (query)                          7_017.43       428.73     7_446.16       0.4331          1.0541         2.43
IVF-OPQ-nl158-m32-np12 (query)                         7_017.43       687.72     7_705.16       0.4332          1.0541         2.43
IVF-OPQ-nl158-m32-np17 (query)                         7_017.43       925.17     7_942.60       0.4332          1.0541         2.43
IVF-OPQ-nl158-m32 (self)                               7_017.43     3_507.36    10_524.79       0.3450          1.0763         2.43
IVF-OPQ-nl158-m64-np7 (query)                         10_068.13       728.91    10_797.04       0.6712          1.0155         3.96
IVF-OPQ-nl158-m64-np12 (query)                        10_068.13     1_161.98    11_230.11       0.6717          1.0155         3.96
IVF-OPQ-nl158-m64-np17 (query)                        10_068.13     1_599.66    11_667.79       0.6717          1.0155         3.96
IVF-OPQ-nl158-m64 (self)                              10_068.13     5_899.33    15_967.46       0.6132          1.0221         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_313.55       354.22     4_667.77       0.3150          1.0862         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_313.55       432.86     4_746.41       0.3150          1.0862         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_313.55       609.53     4_923.08       0.3150          1.0862         1.73
IVF-OPQ-nl223-m16 (self)                               4_313.55     2_365.39     6_678.94       0.2210          1.1245         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_592.93       594.87     7_187.80       0.4371          1.0527         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_592.93       737.96     7_330.89       0.4371          1.0527         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_592.93     1_063.68     7_656.62       0.4371          1.0527         2.50
IVF-OPQ-nl223-m32 (self)                               6_592.93     3_857.46    10_450.40       0.3480          1.0754         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_767.11       994.44    10_761.55       0.6778          1.0149         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_767.11     1_246.93    11_014.04       0.6779          1.0149         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_767.11     1_841.91    11_609.02       0.6779          1.0149         4.02
IVF-OPQ-nl223-m64 (self)                               9_767.11     6_375.16    16_142.27       0.6177          1.0215         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_684.48       453.31     5_137.80       0.3166          1.0849         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_684.48       501.22     5_185.70       0.3166          1.0849         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_684.48       709.04     5_393.52       0.3166          1.0849         2.07
IVF-OPQ-nl316-m16 (self)                               4_684.48     2_666.74     7_351.22       0.2231          1.1228         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_823.54       755.90     7_579.45       0.4423          1.0512         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_823.54       829.71     7_653.26       0.4423          1.0512         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_823.54     1_155.70     7_979.25       0.4423          1.0512         2.84
IVF-OPQ-nl316-m32 (self)                               6_823.54     4_217.37    11_040.92       0.3505          1.0742         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_898.85     1_263.89    11_162.74       0.6797          1.0146         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_898.85     1_402.87    11_301.72       0.6798          1.0146         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_898.85     2_028.52    11_927.36       0.6798          1.0146         4.36
IVF-OPQ-nl316-m64 (self)                               9_898.85     7_037.23    16_936.08       0.6191          1.0212         4.36
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
Exhaustive (query)                                        67.69     1_272.19     1_339.89       1.0000          1.0000        97.66
Exhaustive (self)                                         67.69     4_206.12     4_273.82       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_927.48       998.64     6_926.12       0.1762          1.0982         2.26
Exhaustive-OPQ-m16 (self)                              5_927.48     4_706.42    10_633.90       0.1572          1.1065         2.26
Exhaustive-OPQ-m32 (query)                             7_891.58     1_819.75     9_711.33       0.1907          1.0952         3.03
Exhaustive-OPQ-m32 (self)                              7_891.58     7_489.07    15_380.66       0.1641          1.1051         3.03
Exhaustive-OPQ-m64 (query)                            11_869.59     3_933.84    15_803.43       0.2247          1.0856         4.55
Exhaustive-OPQ-m64 (self)                             11_869.59    14_368.84    26_238.43       0.1794          1.0992         4.55
Exhaustive-OPQ-m128 (query)                           17_468.12     8_117.68    25_585.80       0.2900          1.0699         7.61
Exhaustive-OPQ-m128 (self)                            17_468.12    28_606.74    46_074.86       0.2385          1.0799         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_863.25       593.70     8_456.95       0.2228          1.0801         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_863.25       758.96     8_622.21       0.2228          1.0801         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_863.25       951.78     8_815.03       0.2228          1.0801         3.07
IVF-OPQ-nl158-m16 (self)                               7_863.25     4_435.63    12_298.88       0.1562          1.1072         3.07
IVF-OPQ-nl158-m32-np7 (query)                          9_768.52       741.18    10_509.70       0.2820          1.0639         3.84
IVF-OPQ-nl158-m32-np12 (query)                         9_768.52       989.91    10_758.43       0.2820          1.0639         3.84
IVF-OPQ-nl158-m32-np17 (query)                         9_768.52     1_256.61    11_025.13       0.2820          1.0639         3.84
IVF-OPQ-nl158-m32 (self)                               9_768.52     5_587.54    15_356.06       0.1940          1.0908         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_800.95     1_068.03    14_868.98       0.4082          1.0388         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_800.95     1_557.43    15_358.38       0.4083          1.0387         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_800.95     2_079.71    15_880.66       0.4083          1.0387         5.36
IVF-OPQ-nl158-m64 (self)                              13_800.95     8_106.94    21_907.89       0.3212          1.0545         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_358.41     1_616.08    20_974.49       0.6576          1.0110         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_358.41     2_480.22    21_838.63       0.6578          1.0110         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_358.41     3_316.21    22_674.62       0.6578          1.0110         8.42
IVF-OPQ-nl158-m128 (self)                             19_358.41    12_422.36    31_780.77       0.5972          1.0156         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_742.81       716.67     7_459.47       0.2254          1.0789         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_742.81       798.15     7_540.96       0.2254          1.0789         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_742.81     1_012.82     7_755.63       0.2254          1.0789         3.20
IVF-OPQ-nl223-m16 (self)                               6_742.81     4_783.27    11_526.08       0.1582          1.1057         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_914.46       908.01     9_822.46       0.2859          1.0625         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_914.46     1_060.18     9_974.64       0.2859          1.0625         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_914.46     1_393.99    10_308.45       0.2859          1.0625         3.96
IVF-OPQ-nl223-m32 (self)                               8_914.46     6_048.45    14_962.91       0.1959          1.0899         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_924.72     1_400.19    14_324.90       0.4139          1.0376         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_924.72     1_680.21    14_604.93       0.4139          1.0376         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_924.72     2_342.78    15_267.49       0.4139          1.0376         5.49
IVF-OPQ-nl223-m64 (self)                              12_924.72     9_049.05    21_973.76       0.3251          1.0535         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_433.61     2_133.52    20_567.12       0.6608          1.0108         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_433.61     2_610.17    21_043.77       0.6609          1.0108         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_433.61     3_738.81    22_172.41       0.6609          1.0108         8.54
IVF-OPQ-nl223-m128 (self)                             18_433.61    13_835.56    32_269.17       0.6007          1.0154         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_198.89       896.49     8_095.38       0.2259          1.0783         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_198.89     1_027.10     8_225.99       0.2259          1.0783         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_198.89     1_220.20     8_419.09       0.2259          1.0783         3.88
IVF-OPQ-nl316-m16 (self)                               7_198.89     5_623.05    12_821.94       0.1591          1.1048         3.88
IVF-OPQ-nl316-m32-np15 (query)                         9_194.14     1_088.99    10_283.13       0.2891          1.0616         4.65
IVF-OPQ-nl316-m32-np17 (query)                         9_194.14     1_174.61    10_368.75       0.2892          1.0616         4.65
IVF-OPQ-nl316-m32-np25 (query)                         9_194.14     1_551.65    10_745.78       0.2892          1.0616         4.65
IVF-OPQ-nl316-m32 (self)                               9_194.14     6_584.10    15_778.24       0.1970          1.0891         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_174.47     1_693.84    14_868.31       0.4166          1.0371         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_174.47     1_879.52    15_053.99       0.4166          1.0371         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_174.47     2_590.86    15_765.33       0.4166          1.0371         6.17
IVF-OPQ-nl316-m64 (self)                              13_174.47     9_927.35    23_101.82       0.3269          1.0530         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_886.58     2_640.71    21_527.29       0.6628          1.0106         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_886.58     2_928.95    21_815.53       0.6628          1.0106         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_886.58     4_144.34    23_030.92       0.6628          1.0106         9.23
IVF-OPQ-nl316-m128 (self)                             18_886.58    15_336.54    34_223.12       0.6012          1.0153         9.23
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
Exhaustive (query)                                       100.84     1_788.25     1_889.09       1.0000          1.0000       146.48
Exhaustive (self)                                        100.84     6_006.50     6_107.34       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_643.91     1_512.33    11_156.25       0.1637          1.0821         3.76
Exhaustive-OPQ-m16 (self)                              9_643.91     8_132.35    17_776.26       0.1482          1.0878         3.76
Exhaustive-OPQ-m32 (query)                            11_643.18     2_333.51    13_976.69       0.1741          1.0796         4.53
Exhaustive-OPQ-m32 (self)                             11_643.18    10_906.42    22_549.60       0.1529          1.0869         4.53
Exhaustive-OPQ-m64 (query)                            16_097.13     4_484.09    20_581.22       0.1926          1.0758         6.05
Exhaustive-OPQ-m64 (self)                             16_097.13    17_811.95    33_909.08       0.1589          1.0856         6.05
Exhaustive-OPQ-m128 (query)                           24_699.90     8_657.16    33_357.06       0.2310          1.0682         9.11
Exhaustive-OPQ-m128 (self)                            24_699.90    34_515.27    59_215.17       0.1838          1.0786         9.11
IVF-OPQ-nl158-m16-np7 (query)                         12_454.62     1_152.33    13_606.95       0.1922          1.0718         4.98
IVF-OPQ-nl158-m16-np12 (query)                        12_454.62     1_372.60    13_827.21       0.1922          1.0718         4.98
IVF-OPQ-nl158-m16-np17 (query)                        12_454.62     1_543.52    13_998.13       0.1922          1.0718         4.98
IVF-OPQ-nl158-m16 (self)                              12_454.62     8_336.45    20_791.07       0.1408          1.0916         4.98
IVF-OPQ-nl158-m32-np7 (query)                         14_639.43     1_321.41    15_960.85       0.2355          1.0607         5.74
IVF-OPQ-nl158-m32-np12 (query)                        14_639.43     1_644.60    16_284.03       0.2355          1.0607         5.74
IVF-OPQ-nl158-m32-np17 (query)                        14_639.43     1_963.43    16_602.87       0.2355          1.0607         5.74
IVF-OPQ-nl158-m32 (self)                              14_639.43     9_701.72    24_341.15       0.1593          1.0839         5.74
IVF-OPQ-nl158-m64-np7 (query)                         19_318.16     1_640.37    20_958.53       0.3101          1.0455         7.27
IVF-OPQ-nl158-m64-np12 (query)                        19_318.16     2_182.46    21_500.62       0.3101          1.0455         7.27
IVF-OPQ-nl158-m64-np17 (query)                        19_318.16     2_733.34    22_051.50       0.3101          1.0455         7.27
IVF-OPQ-nl158-m64 (self)                              19_318.16    12_197.62    31_515.78       0.2196          1.0647         7.27
IVF-OPQ-nl158-m128-np7 (query)                        27_435.97     2_417.64    29_853.61       0.4928          1.0210        10.32
IVF-OPQ-nl158-m128-np12 (query)                       27_435.97     3_497.72    30_933.69       0.4930          1.0210        10.32
IVF-OPQ-nl158-m128-np17 (query)                       27_435.97     4_581.98    32_017.95       0.4930          1.0210        10.32
IVF-OPQ-nl158-m128 (self)                             27_435.97    18_411.19    45_847.15       0.4153          1.0292        10.32
IVF-OPQ-nl223-m16-np11 (query)                        11_112.82     1_327.16    12_439.98       0.1954          1.0708         5.17
IVF-OPQ-nl223-m16-np14 (query)                        11_112.82     1_412.79    12_525.61       0.1954          1.0708         5.17
IVF-OPQ-nl223-m16-np21 (query)                        11_112.82     1_652.02    12_764.84       0.1954          1.0708         5.17
IVF-OPQ-nl223-m16 (self)                              11_112.82     8_692.24    19_805.06       0.1428          1.0903         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_116.23     1_556.03    14_672.27       0.2369          1.0599         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_116.23     1_740.71    14_856.94       0.2369          1.0599         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_116.23     2_162.47    15_278.70       0.2369          1.0599         5.93
IVF-OPQ-nl223-m32 (self)                              13_116.23    10_469.44    23_585.67       0.1604          1.0831         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_534.55     1_997.37    19_531.92       0.3139          1.0447         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_534.55     2_296.08    19_830.63       0.3139          1.0447         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_534.55     2_992.51    20_527.06       0.3139          1.0447         7.46
IVF-OPQ-nl223-m64 (self)                              17_534.55    13_156.91    30_691.46       0.2225          1.0639         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_159.48     3_137.83    29_297.32       0.4984          1.0205        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_159.48     3_735.52    29_895.00       0.4985          1.0205        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_159.48     5_128.26    31_287.75       0.4985          1.0205        10.51
IVF-OPQ-nl223-m128 (self)                             26_159.48    20_272.42    46_431.90       0.4183          1.0287        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_767.94     1_444.27    13_212.21       0.1967          1.0700         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_767.94     1_600.49    13_368.43       0.1967          1.0700         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_767.94     1_791.03    13_558.97       0.1967          1.0700         6.19
IVF-OPQ-nl316-m16 (self)                              11_767.94     9_158.06    20_926.00       0.1429          1.0896         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_994.37     1_793.58    15_787.96       0.2379          1.0593         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_994.37     1_919.28    15_913.66       0.2379          1.0593         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_994.37     2_365.93    16_360.30       0.2379          1.0593         6.96
IVF-OPQ-nl316-m32 (self)                              13_994.37    11_058.07    25_052.44       0.1606          1.0827         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_233.82     2_349.96    20_583.78       0.3176          1.0437         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_233.82     2_546.30    20_780.12       0.3176          1.0437         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_233.82     3_296.29    21_530.11       0.3176          1.0437         8.48
IVF-OPQ-nl316-m64 (self)                              18_233.82    14_209.53    32_443.34       0.2232          1.0633         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_730.60     3_821.81    30_552.41       0.5024          1.0200        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_730.60     4_185.77    30_916.36       0.5024          1.0200        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_730.60     5_701.28    32_431.88       0.5024          1.0200        11.54
IVF-OPQ-nl316-m128 (self)                             26_730.60    22_215.74    48_946.34       0.4221          1.0284        11.54
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
Exhaustive (query)                                        32.61       714.20       746.81       1.0000          1.0000        48.83
Exhaustive (self)                                         32.61     2_387.98     2_420.59       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_848.38       719.69     4_568.07       0.3053          1.2404         1.26
Exhaustive-OPQ-m16 (self)                              3_848.38     2_737.93     6_586.30       0.2397          1.3724         1.26
Exhaustive-OPQ-m32 (query)                             6_221.26     1_571.64     7_792.90       0.4232          1.1478         2.03
Exhaustive-OPQ-m32 (self)                              6_221.26     5_534.14    11_755.40       0.3399          1.2445         2.03
Exhaustive-OPQ-m64 (query)                             9_195.97     3_675.33    12_871.30       0.5626          1.0761         3.55
Exhaustive-OPQ-m64 (self)                              9_195.97    12_570.87    21_766.84       0.4889          1.1272         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_685.22       264.33     4_949.55       0.7053          1.0305         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_685.22       383.66     5_068.88       0.7053          1.0305         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_685.22       484.26     5_169.48       0.7053          1.0305         1.67
IVF-OPQ-nl158-m16 (self)                               4_685.22     1_969.87     6_655.08       0.6291          1.0588         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_514.60       413.01     6_927.61       0.7969          1.0139         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_514.60       618.95     7_133.55       0.7969          1.0139         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_514.60       828.10     7_342.70       0.7969          1.0139         2.43
IVF-OPQ-nl158-m32 (self)                               6_514.60     3_102.32     9_616.92       0.7477          1.0257         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_172.62       685.59     9_858.21       0.8582          1.0066         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_172.62     1_033.94    10_206.56       0.8582          1.0066         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_172.62     1_484.03    10_656.65       0.8582          1.0066         3.96
IVF-OPQ-nl158-m64 (self)                               9_172.62     4_999.20    14_171.81       0.8296          1.0113         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_119.85       350.88     4_470.74       0.7119          1.0291         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_119.85       416.36     4_536.21       0.7121          1.0290         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_119.85       579.97     4_699.82       0.7121          1.0290         1.73
IVF-OPQ-nl223-m16 (self)                               4_119.85     2_280.59     6_400.44       0.6400          1.0551         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_191.62       575.60     6_767.23       0.8010          1.0131         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_191.62       701.04     6_892.66       0.8013          1.0130         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_191.62     1_017.74     7_209.36       0.8013          1.0130         2.50
IVF-OPQ-nl223-m32 (self)                               6_191.62     3_704.73     9_896.36       0.7550          1.0241         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_006.31       957.95     9_964.27       0.8605          1.0063         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_006.31     1_166.96    10_173.27       0.8609          1.0063         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_006.31     1_696.65    10_702.97       0.8609          1.0063         4.02
IVF-OPQ-nl223-m64 (self)                               9_006.31     6_002.89    15_009.20       0.8332          1.0108         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_261.32       440.02     4_701.34       0.7177          1.0276         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_261.32       482.69     4_744.02       0.7178          1.0276         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_261.32       662.86     4_924.18       0.7178          1.0276         2.07
IVF-OPQ-nl316-m16 (self)                               4_261.32     2_546.25     6_807.57       0.6468          1.0530         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_340.09       730.62     7_070.70       0.8048          1.0127         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_340.09       806.29     7_146.38       0.8050          1.0127         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_340.09     1_134.88     7_474.97       0.8050          1.0127         2.84
IVF-OPQ-nl316-m32 (self)                               6_340.09     4_112.87    10_452.96       0.7590          1.0233         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_114.15     1_218.09    10_332.23       0.8633          1.0061         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_114.15     1_350.65    10_464.80       0.8635          1.0060         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_114.15     1_962.13    11_076.27       0.8635          1.0060         4.36
IVF-OPQ-nl316-m64 (self)                               9_114.15     6_728.82    15_842.96       0.8359          1.0104         4.36
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
Exhaustive (query)                                        68.78     1_233.07     1_301.85       1.0000          1.0000        97.66
Exhaustive (self)                                         68.78     4_143.38     4_212.16       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_860.76     1_004.34     6_865.10       0.2342          1.2086         2.26
Exhaustive-OPQ-m16 (self)                              5_860.76     4_728.50    10_589.27       0.1875          1.2966         2.26
Exhaustive-OPQ-m32 (query)                             7_710.90     1_830.36     9_541.25       0.3238          1.1452         3.03
Exhaustive-OPQ-m32 (self)                              7_710.90     7_389.42    15_100.32       0.2612          1.2126         3.03
Exhaustive-OPQ-m64 (query)                            11_831.16     3_925.32    15_756.49       0.4399          1.0894         4.55
Exhaustive-OPQ-m64 (self)                             11_831.16    14_321.80    26_152.97       0.3681          1.1368         4.55
Exhaustive-OPQ-m128 (query)                           17_445.53     8_091.92    25_537.45       0.5765          1.0464         7.61
Exhaustive-OPQ-m128 (self)                            17_445.53    28_366.78    45_812.32       0.5090          1.0733         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_724.07       594.70     8_318.77       0.5324          1.0572         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_724.07       729.11     8_453.18       0.5324          1.0572         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_724.07       886.35     8_610.43       0.5324          1.0572         3.07
IVF-OPQ-nl158-m16 (self)                               7_724.07     4_393.42    12_117.49       0.4302          1.1033         3.07
IVF-OPQ-nl158-m32-np7 (query)                          9_447.31       724.69    10_172.00       0.6803          1.0245         3.84
IVF-OPQ-nl158-m32-np12 (query)                         9_447.31       928.56    10_375.86       0.6803          1.0245         3.84
IVF-OPQ-nl158-m32-np17 (query)                         9_447.31     1_141.51    10_588.81       0.6803          1.0245         3.84
IVF-OPQ-nl158-m32 (self)                               9_447.31     5_195.86    14_643.17       0.6088          1.0435         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_458.87     1_043.05    14_501.92       0.7796          1.0112         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_458.87     1_459.40    14_918.27       0.7796          1.0112         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_458.87     1_867.59    15_326.46       0.7796          1.0112         5.36
IVF-OPQ-nl158-m64 (self)                              13_458.87     7_517.05    20_975.92       0.7350          1.0191         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_826.30     1_562.03    20_388.34       0.8386          1.0059         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_826.30     2_211.10    21_037.40       0.8386          1.0059         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_826.30     2_890.61    21_716.91       0.8386          1.0059         8.42
IVF-OPQ-nl158-m128 (self)                             18_826.30    11_258.10    30_084.41       0.8128          1.0095         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_981.95       704.41     7_686.35       0.5381          1.0555         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_981.95       789.93     7_771.87       0.5381          1.0555         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_981.95       999.83     7_981.77       0.5381          1.0555         3.20
IVF-OPQ-nl223-m16 (self)                               6_981.95     4_771.71    11_753.66       0.4375          1.1003         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_672.53       895.33     9_567.86       0.6905          1.0228         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_672.53     1_007.43     9_679.96       0.6905          1.0228         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_672.53     1_314.60     9_987.13       0.6905          1.0228         3.96
IVF-OPQ-nl223-m32 (self)                               8_672.53     5_767.98    14_440.51       0.6196          1.0407         3.96
IVF-OPQ-nl223-m64-np11 (query)                        13_030.46     1_365.09    14_395.55       0.7842          1.0107         5.49
IVF-OPQ-nl223-m64-np14 (query)                        13_030.46     1_618.44    14_648.90       0.7842          1.0107         5.49
IVF-OPQ-nl223-m64-np21 (query)                        13_030.46     2_244.34    15_274.80       0.7842          1.0107         5.49
IVF-OPQ-nl223-m64 (self)                              13_030.46     8_658.81    21_689.27       0.7394          1.0183         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_443.58     2_073.31    20_516.90       0.8431          1.0056         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_443.58     2_478.78    20_922.36       0.8431          1.0056         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_443.58     3_466.68    21_910.26       0.8431          1.0056         8.54
IVF-OPQ-nl223-m128 (self)                             18_443.58    13_049.22    31_492.80       0.8166          1.0089         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_281.08       823.02     8_104.11       0.5428          1.0539         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_281.08       886.39     8_167.47       0.5428          1.0539         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_281.08     1_120.81     8_401.89       0.5428          1.0539         3.88
IVF-OPQ-nl316-m16 (self)                               7_281.08     5_080.23    12_361.32       0.4445          1.0972         3.88
IVF-OPQ-nl316-m32-np15 (query)                         9_185.55     1_048.94    10_234.49       0.6934          1.0223         4.65
IVF-OPQ-nl316-m32-np17 (query)                         9_185.55     1_124.11    10_309.67       0.6934          1.0223         4.65
IVF-OPQ-nl316-m32-np25 (query)                         9_185.55     1_469.79    10_655.34       0.6934          1.0223         4.65
IVF-OPQ-nl316-m32 (self)                               9_185.55     6_226.02    15_411.57       0.6224          1.0399         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_250.00     1_684.17    14_934.17       0.7867          1.0103         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_250.00     1_836.53    15_086.54       0.7867          1.0103         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_250.00     2_510.13    15_760.14       0.7867          1.0103         6.17
IVF-OPQ-nl316-m64 (self)                              13_250.00     9_571.56    22_821.56       0.7419          1.0178         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_852.38     2_587.52    21_439.90       0.8446          1.0055         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_852.38     2_838.20    21_690.58       0.8446          1.0055         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_852.38     3_915.80    22_768.18       0.8446          1.0055         9.23
IVF-OPQ-nl316-m128 (self)                             18_852.38    14_548.96    33_401.33       0.8181          1.0087         9.23
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
Exhaustive (query)                                        99.58     1_782.65     1_882.23       1.0000          1.0000       146.48
Exhaustive (self)                                         99.58     5_919.10     6_018.68       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_538.95     1_520.12    11_059.07       0.2321          1.1964         3.76
Exhaustive-OPQ-m16 (self)                              9_538.95     8_204.43    17_743.38       0.1865          1.2954         3.76
Exhaustive-OPQ-m32 (query)                            12_206.41     2_334.88    14_541.29       0.3106          1.1421         4.53
Exhaustive-OPQ-m32 (self)                             12_206.41    10_948.14    23_154.55       0.2539          1.2184         4.53
Exhaustive-OPQ-m64 (query)                            16_203.54     4_417.57    20_621.11       0.4134          1.0932         6.05
Exhaustive-OPQ-m64 (self)                             16_203.54    17_854.48    34_058.03       0.3530          1.1451         6.05
Exhaustive-OPQ-m128 (query)                           24_924.17     8_627.29    33_551.47       0.5300          1.0549         9.11
Exhaustive-OPQ-m128 (self)                            24_924.17    32_015.64    56_939.82       0.4714          1.0873         9.11
IVF-OPQ-nl158-m16-np7 (query)                         12_039.75     1_136.42    13_176.17       0.5304          1.0537         4.98
IVF-OPQ-nl158-m16-np12 (query)                        12_039.75     1_317.25    13_357.00       0.5304          1.0537         4.98
IVF-OPQ-nl158-m16-np17 (query)                        12_039.75     1_501.08    13_540.83       0.5304          1.0537         4.98
IVF-OPQ-nl158-m16 (self)                              12_039.75     8_196.24    20_235.99       0.4299          1.1033         4.98
IVF-OPQ-nl158-m32-np7 (query)                         14_263.62     1_315.19    15_578.81       0.6726          1.0241         5.74
IVF-OPQ-nl158-m32-np12 (query)                        14_263.62     1_588.10    15_851.72       0.6726          1.0241         5.74
IVF-OPQ-nl158-m32-np17 (query)                        14_263.62     2_000.44    16_264.06       0.6726          1.0241         5.74
IVF-OPQ-nl158-m32 (self)                              14_263.62    10_027.20    24_290.82       0.5980          1.0457         5.74
IVF-OPQ-nl158-m64-np7 (query)                         19_083.65     1_657.29    20_740.94       0.7739          1.0111         7.27
IVF-OPQ-nl158-m64-np12 (query)                        19_083.65     2_067.75    21_151.40       0.7739          1.0111         7.27
IVF-OPQ-nl158-m64-np17 (query)                        19_083.65     2_521.80    21_605.44       0.7739          1.0111         7.27
IVF-OPQ-nl158-m64 (self)                              19_083.65    11_593.81    30_677.46       0.7282          1.0200         7.27
IVF-OPQ-nl158-m128-np7 (query)                        27_245.82     2_378.20    29_624.02       0.8297          1.0062        10.32
IVF-OPQ-nl158-m128-np12 (query)                       27_245.82     3_260.73    30_506.55       0.8297          1.0062        10.32
IVF-OPQ-nl158-m128-np17 (query)                       27_245.82     4_172.17    31_417.98       0.8297          1.0062        10.32
IVF-OPQ-nl158-m128 (self)                             27_245.82    17_091.91    44_337.73       0.8036          1.0105        10.32
IVF-OPQ-nl223-m16-np11 (query)                        11_285.92     1_310.49    12_596.42       0.5388          1.0512         5.17
IVF-OPQ-nl223-m16-np14 (query)                        11_285.92     1_402.13    12_688.05       0.5388          1.0512         5.17
IVF-OPQ-nl223-m16-np21 (query)                        11_285.92     1_675.04    12_960.96       0.5388          1.0512         5.17
IVF-OPQ-nl223-m16 (self)                              11_285.92     8_918.92    20_204.85       0.4377          1.0996         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_312.61     1_548.18    14_860.79       0.6819          1.0226         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_312.61     1_711.17    15_023.79       0.6819          1.0226         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_312.61     2_115.22    15_427.83       0.6819          1.0226         5.93
IVF-OPQ-nl223-m32 (self)                              13_312.61    10_266.10    23_578.71       0.6078          1.0434         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_977.02     1_976.21    19_953.23       0.7818          1.0102         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_977.02     2_288.28    20_265.30       0.7818          1.0102         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_977.02     2_920.16    20_897.17       0.7818          1.0102         7.46
IVF-OPQ-nl223-m64 (self)                              17_977.02    12_893.85    30_870.86       0.7354          1.0187         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_379.86     3_081.79    29_461.65       0.8333          1.0059        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_379.86     3_617.49    29_997.35       0.8333          1.0059        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_379.86     4_934.87    31_314.73       0.8333          1.0059        10.51
IVF-OPQ-nl223-m128 (self)                             26_379.86    19_725.31    46_105.17       0.8073          1.0099        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_777.03     1_548.72    13_325.75       0.5448          1.0496         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_777.03     1_522.40    13_299.43       0.5448          1.0496         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_777.03     1_822.17    13_599.20       0.5448          1.0496         6.19
IVF-OPQ-nl316-m16 (self)                              11_777.03     9_319.02    21_096.05       0.4471          1.0952         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_982.26     1_768.13    15_750.39       0.6839          1.0222         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_982.26     1_909.66    15_891.92       0.6839          1.0222         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_982.26     2_340.36    16_322.62       0.6839          1.0222         6.96
IVF-OPQ-nl316-m32 (self)                              13_982.26    10_973.73    24_955.99       0.6117          1.0422         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_531.09     2_339.35    20_870.45       0.7829          1.0101         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_531.09     2_524.24    21_055.33       0.7829          1.0101         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_531.09     3_253.40    21_784.49       0.7829          1.0101         8.48
IVF-OPQ-nl316-m64 (self)                              18_531.09    14_251.13    32_782.23       0.7378          1.0183         8.48
IVF-OPQ-nl316-m128-np15 (query)                       27_952.83     3_818.23    31_771.06       0.8347          1.0058        11.54
IVF-OPQ-nl316-m128-np17 (query)                       27_952.83     4_184.19    32_137.02       0.8347          1.0058        11.54
IVF-OPQ-nl316-m128-np25 (query)                       27_952.83     5_632.98    33_585.81       0.8347          1.0058        11.54
IVF-OPQ-nl316-m128 (self)                             27_952.83    21_979.47    49_932.31       0.8089          1.0097        11.54
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
Exhaustive (query)                                        32.57       685.25       717.82       1.0000          1.0000        48.83
Exhaustive (self)                                         32.57     2_324.36     2_356.93       1.0000          1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_491.67       716.03     4_207.70       0.7911          1.0819         1.26
Exhaustive-OPQ-m16 (self)                              3_491.67     2_694.49     6_186.16       0.7231          1.1502         1.26
Exhaustive-OPQ-m32 (query)                             5_553.77     1_544.69     7_098.47       0.8303          1.0536         2.03
Exhaustive-OPQ-m32 (self)                              5_553.77     5_436.47    10_990.24       0.7763          1.0975         2.03
Exhaustive-OPQ-m64 (query)                             8_357.30     3_609.47    11_966.77       0.8562          1.0398         3.55
Exhaustive-OPQ-m64 (self)                              8_357.30    12_373.20    20_730.50       0.8092          1.0723         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_443.57       274.85     4_718.42       0.8895          1.0208         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_443.57       418.79     4_862.37       0.8903          1.0203         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_443.57       540.72     4_984.30       0.8904          1.0203         1.67
IVF-OPQ-nl158-m16 (self)                               4_443.57     2_148.87     6_592.45       0.8475          1.0405         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_443.05       453.95     6_897.01       0.9110          1.0133         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_443.05       712.03     7_155.08       0.9118          1.0129         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_443.05       969.08     7_412.13       0.9118          1.0128         2.43
IVF-OPQ-nl158-m32 (self)                               6_443.05     3_586.70    10_029.75       0.8774          1.0255         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_144.75       790.42     9_935.18       0.9240          1.0098         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_144.75     1_268.96    10_413.72       0.9248          1.0094         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_144.75     1_763.25    10_908.00       0.9248          1.0094         3.96
IVF-OPQ-nl158-m64 (self)                               9_144.75     6_074.60    15_219.35       0.8961          1.0185         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_941.63       377.63     4_319.26       0.8979          1.0178         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_941.63       455.79     4_397.42       0.8980          1.0177         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_941.63       647.82     4_589.46       0.8980          1.0177         1.73
IVF-OPQ-nl223-m16 (self)                               3_941.63     2_492.81     6_434.45       0.8581          1.0352         1.73
IVF-OPQ-nl223-m32-np11 (query)                         5_986.14       600.35     6_586.49       0.9152          1.0118         2.50
IVF-OPQ-nl223-m32-np14 (query)                         5_986.14       740.09     6_726.23       0.9153          1.0117         2.50
IVF-OPQ-nl223-m32-np21 (query)                         5_986.14     1_060.57     7_046.71       0.9153          1.0117         2.50
IVF-OPQ-nl223-m32 (self)                               5_986.14     3_885.11     9_871.26       0.8827          1.0234         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_773.09     1_015.85     9_788.95       0.9269          1.0092         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_773.09     1_270.77    10_043.86       0.9271          1.0091         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_773.09     1_869.79    10_642.88       0.9271          1.0091         4.02
IVF-OPQ-nl223-m64 (self)                               8_773.09     6_453.78    15_226.87       0.8987          1.0178         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_238.55       443.89     4_682.44       0.9021          1.0164         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_238.55       492.37     4_730.93       0.9022          1.0164         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_238.55       679.38     4_917.93       0.9022          1.0164         2.07
IVF-OPQ-nl316-m16 (self)                               4_238.55     2_631.23     6_869.78       0.8629          1.0334         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_234.46       752.05     6_986.51       0.9172          1.0112         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_234.46       834.42     7_068.88       0.9173          1.0112         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_234.46     1_221.58     7_456.05       0.9174          1.0112         2.84
IVF-OPQ-nl316-m32 (self)                               6_234.46     4_299.60    10_534.06       0.8839          1.0230         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_899.03     1_278.54    10_177.58       0.9281          1.0087         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_899.03     1_417.56    10_316.59       0.9281          1.0087         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_899.03     2_042.61    10_941.64       0.9282          1.0087         4.36
IVF-OPQ-nl316-m64 (self)                               8_899.03     7_110.94    16_009.97       0.9005          1.0171         4.36
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
Exhaustive (query)                                        69.11     1_256.37     1_325.48       1.0000          1.0000        97.66
Exhaustive (self)                                         69.11     4_213.20     4_282.31       1.0000          1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_928.05     1_000.81     6_928.87       0.7546          1.1136         2.26
Exhaustive-OPQ-m16 (self)                              5_928.05     4_740.58    10_668.64       0.6788          1.2037         2.26
Exhaustive-OPQ-m32 (query)                             7_698.20     1_822.13     9_520.34       0.8064          1.0692         3.03
Exhaustive-OPQ-m32 (self)                              7_698.20     7_387.86    15_086.06       0.7455          1.1245         3.03
Exhaustive-OPQ-m64 (query)                            11_784.22     3_926.82    15_711.04       0.8413          1.0455         4.55
Exhaustive-OPQ-m64 (self)                             11_784.22    14_346.58    26_130.80       0.7916          1.0819         4.55
Exhaustive-OPQ-m128 (query)                           17_487.59     8_071.76    25_559.35       0.9198          1.0107         7.61
Exhaustive-OPQ-m128 (self)                            17_487.59    28_181.56    45_669.15       0.8933          1.0192         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_926.36       616.65     8_543.01       0.8909          1.0219         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_926.36       784.44     8_710.80       0.8913          1.0217         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_926.36       959.10     8_885.45       0.8913          1.0217         3.07
IVF-OPQ-nl158-m16 (self)                               7_926.36     4_532.12    12_458.48       0.8464          1.0446         3.07
IVF-OPQ-nl158-m32-np7 (query)                          9_703.93       753.79    10_457.72       0.9020          1.0171         3.84
IVF-OPQ-nl158-m32-np12 (query)                         9_703.93     1_021.00    10_724.93       0.9026          1.0169         3.84
IVF-OPQ-nl158-m32-np17 (query)                         9_703.93     1_291.32    10_995.25       0.9026          1.0169         3.84
IVF-OPQ-nl158-m32 (self)                               9_703.93     5_672.53    15_376.46       0.8629          1.0344         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_706.59     1_124.80    14_831.39       0.9096          1.0145         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_706.59     1_671.37    15_377.96       0.9102          1.0143         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_706.59     2_190.93    15_897.53       0.9102          1.0143         5.36
IVF-OPQ-nl158-m64 (self)                              13_706.59     8_550.26    22_256.85       0.8749          1.0286         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_292.26     1_725.57    21_017.83       0.9631          1.0027         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_292.26     2_659.00    21_951.26       0.9638          1.0025         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_292.26     3_590.24    22_882.50       0.9638          1.0025         8.42
IVF-OPQ-nl158-m128 (self)                             19_292.26    13_333.79    32_626.05       0.9454          1.0056         8.42
IVF-OPQ-nl223-m16-np11 (query)                         7_016.67       720.54     7_737.21       0.8980          1.0193         3.20
IVF-OPQ-nl223-m16-np14 (query)                         7_016.67       816.15     7_832.82       0.8980          1.0193         3.20
IVF-OPQ-nl223-m16-np21 (query)                         7_016.67     1_042.16     8_058.83       0.8980          1.0193         3.20
IVF-OPQ-nl223-m16 (self)                               7_016.67     4_837.86    11_854.53       0.8555          1.0393         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_621.26       931.67     9_552.93       0.9075          1.0154         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_621.26     1_083.13     9_704.39       0.9075          1.0154         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_621.26     1_447.51    10_068.77       0.9076          1.0154         3.96
IVF-OPQ-nl223-m32 (self)                               8_621.26     6_166.44    14_787.70       0.8701          1.0310         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_685.23     1_424.83    14_110.06       0.9168          1.0124         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_685.23     1_723.44    14_408.67       0.9169          1.0123         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_685.23     2_384.11    15_069.35       0.9169          1.0123         5.49
IVF-OPQ-nl223-m64 (self)                              12_685.23     9_236.78    21_922.01       0.8824          1.0249         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_224.60     2_206.58    20_431.19       0.9660          1.0021         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_224.60     2_727.10    20_951.70       0.9661          1.0021         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_224.60     3_865.34    22_089.94       0.9661          1.0021         8.54
IVF-OPQ-nl223-m128 (self)                             18_224.60    14_274.45    32_499.05       0.9491          1.0049         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_914.62       841.83     7_756.44       0.9054          1.0162         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_914.62       895.03     7_809.64       0.9054          1.0162         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_914.62     1_153.18     8_067.80       0.9054          1.0161         3.88
IVF-OPQ-nl316-m16 (self)                               6_914.62     5_187.70    12_102.32       0.8661          1.0330         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_723.19     1_095.40     9_818.59       0.9145          1.0130         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_723.19     1_198.18     9_921.36       0.9145          1.0130         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_723.19     1_550.93    10_274.12       0.9145          1.0130         4.65
IVF-OPQ-nl316-m32 (self)                               8_723.19     6_534.62    15_257.81       0.8786          1.0267         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_531.80     1_784.03    15_315.84       0.9208          1.0110         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_531.80     2_144.62    15_676.43       0.9208          1.0110         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_531.80     2_676.99    16_208.79       0.9208          1.0110         6.17
IVF-OPQ-nl316-m64 (self)                              13_531.80    10_044.65    23_576.45       0.8884          1.0222         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_632.53     2_680.81    21_313.34       0.9691          1.0017         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_632.53     2_994.80    21_627.33       0.9691          1.0017         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_632.53     4_215.49    22_848.02       0.9691          1.0017         9.23
IVF-OPQ-nl316-m128 (self)                             18_632.53    15_452.38    34_084.91       0.9520          1.0044         9.23
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
Exhaustive (query)                                       112.55     1_822.98     1_935.53       1.0000          1.0000       146.48
Exhaustive (self)                                        112.55     6_045.15     6_157.70       1.0000          1.0000       146.48
Exhaustive-OPQ-m16 (query)                            10_308.90     1_549.57    11_858.47       0.7383          1.1306         3.76
Exhaustive-OPQ-m16 (self)                             10_308.90     8_209.96    18_518.87       0.6595          1.2295         3.76
Exhaustive-OPQ-m32 (query)                            12_788.73     2_377.95    15_166.68       0.8493          1.0411         4.53
Exhaustive-OPQ-m32 (self)                             12_788.73    11_034.03    23_822.76       0.8006          1.0714         4.53
Exhaustive-OPQ-m64 (query)                            17_749.48     4_497.73    22_247.21       0.8796          1.0255         6.05
Exhaustive-OPQ-m64 (self)                             17_749.48    18_020.17    35_769.65       0.8413          1.0441         6.05
Exhaustive-OPQ-m128 (query)                           27_300.25     8_683.29    35_983.54       0.9051          1.0148         9.11
Exhaustive-OPQ-m128 (self)                            27_300.25    32_205.16    59_505.41       0.8741          1.0264         9.11
IVF-OPQ-nl158-m16-np7 (query)                         13_202.14     1_167.82    14_369.96       0.8920          1.0223         4.98
IVF-OPQ-nl158-m16-np12 (query)                        13_202.14     1_380.62    14_582.76       0.8922          1.0222         4.98
IVF-OPQ-nl158-m16-np17 (query)                        13_202.14     1_586.65    14_788.79       0.8922          1.0222         4.98
IVF-OPQ-nl158-m16 (self)                              13_202.14     8_450.55    21_652.69       0.8473          1.0438         4.98
IVF-OPQ-nl158-m32-np7 (query)                         15_410.54     1_350.33    16_760.88       0.9369          1.0085         5.74
IVF-OPQ-nl158-m32-np12 (query)                        15_410.54     1_683.45    17_093.99       0.9371          1.0085         5.74
IVF-OPQ-nl158-m32-np17 (query)                        15_410.54     2_017.58    17_428.12       0.9371          1.0085         5.74
IVF-OPQ-nl158-m32 (self)                              15_410.54     9_877.15    25_287.69       0.9074          1.0183         5.74
IVF-OPQ-nl158-m64-np7 (query)                         19_389.43     1_700.05    21_089.48       0.9509          1.0052         7.27
IVF-OPQ-nl158-m64-np12 (query)                        19_389.43     2_275.23    21_664.66       0.9512          1.0051         7.27
IVF-OPQ-nl158-m64-np17 (query)                        19_389.43     2_860.64    22_250.06       0.9512          1.0051         7.27
IVF-OPQ-nl158-m64 (self)                              19_389.43    12_702.86    32_092.29       0.9272          1.0115         7.27
IVF-OPQ-nl158-m128-np7 (query)                        27_271.24     2_547.55    29_818.79       0.9607          1.0034        10.32
IVF-OPQ-nl158-m128-np12 (query)                       27_271.24     3_713.99    30_985.23       0.9610          1.0033        10.32
IVF-OPQ-nl158-m128-np17 (query)                       27_271.24     4_865.39    32_136.63       0.9610          1.0033        10.32
IVF-OPQ-nl158-m128 (self)                             27_271.24    19_331.98    46_603.22       0.9396          1.0077        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_800.72     1_307.63    12_108.35       0.9001          1.0187         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_800.72     1_431.69    12_232.42       0.9001          1.0187         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_800.72     1_712.35    12_513.08       0.9001          1.0187         5.17
IVF-OPQ-nl223-m16 (self)                              10_800.72     8_823.98    19_624.71       0.8592          1.0363         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_847.93     1_626.47    14_474.39       0.9422          1.0071         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_847.93     1_762.03    14_609.96       0.9423          1.0071         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_847.93     2_196.26    15_044.19       0.9423          1.0071         5.93
IVF-OPQ-nl223-m32 (self)                              12_847.93    10_506.94    23_354.87       0.9156          1.0148         5.93
IVF-OPQ-nl223-m64-np11 (query)                        18_989.66     2_080.80    21_070.45       0.9544          1.0045         7.46
IVF-OPQ-nl223-m64-np14 (query)                        18_989.66     2_510.51    21_500.17       0.9544          1.0045         7.46
IVF-OPQ-nl223-m64-np21 (query)                        18_989.66     3_103.16    22_092.82       0.9544          1.0045         7.46
IVF-OPQ-nl223-m64 (self)                              18_989.66    13_533.23    32_522.89       0.9329          1.0097         7.46
IVF-OPQ-nl223-m128-np11 (query)                       27_168.12     3_233.34    30_401.45       0.9640          1.0028        10.51
IVF-OPQ-nl223-m128-np14 (query)                       27_168.12     3_855.95    31_024.07       0.9641          1.0027        10.51
IVF-OPQ-nl223-m128-np21 (query)                       27_168.12     5_321.55    32_489.66       0.9641          1.0027        10.51
IVF-OPQ-nl223-m128 (self)                             27_168.12    20_844.84    48_012.96       0.9433          1.0068        10.51
IVF-OPQ-nl316-m16-np15 (query)                        12_160.08     1_546.69    13_706.77       0.9039          1.0172         6.19
IVF-OPQ-nl316-m16-np17 (query)                        12_160.08     1_511.88    13_671.96       0.9039          1.0172         6.19
IVF-OPQ-nl316-m16-np25 (query)                        12_160.08     1_809.16    13_969.24       0.9039          1.0172         6.19
IVF-OPQ-nl316-m16 (self)                              12_160.08     9_194.12    21_354.21       0.8640          1.0338         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_371.63     1_793.07    15_164.69       0.9447          1.0063         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_371.63     1_925.54    15_297.16       0.9447          1.0063         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_371.63     2_409.52    15_781.15       0.9447          1.0063         6.96
IVF-OPQ-nl316-m32 (self)                              13_371.63    11_168.72    24_540.35       0.9197          1.0135         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_263.68     2_389.63    20_653.31       0.9563          1.0040         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_263.68     2_616.97    20_880.65       0.9563          1.0040         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_263.68     3_374.60    21_638.28       0.9563          1.0040         8.48
IVF-OPQ-nl316-m64 (self)                              18_263.68    14_427.73    32_691.41       0.9353          1.0090         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_356.71     3_887.73    30_244.44       0.9655          1.0024        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_356.71     4_270.24    30_626.96       0.9655          1.0024        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_356.71     5_889.14    32_245.85       0.9655          1.0024        11.54
IVF-OPQ-nl316-m128 (self)                             26_356.71    22_700.37    49_057.09       0.9460          1.0062        11.54
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.04     1_256.99     1_325.03       1.0000          1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_931.58       100.23     2_031.81       0.2495          1.0746         2.24
IVFPQ-m64-nl111-np1                                    2_758.24       156.38     2_914.62       0.3497          1.0494         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_096.07       115.19     2_211.26       0.2408          1.0823         4.72
IVFPQ-m32-nl111-np2                                    1_931.58       157.43     2_089.02       0.2650          1.0704         2.24
IVFPQ-m64-nl111-np2                                    2_758.24       268.81     3_027.06       0.3846          1.0436         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_096.07       183.30     2_279.37       0.2568          1.0742         4.72
IVFPQ-m32-nl111-np4                                    1_931.58       269.49     2_201.07       0.2690          1.0696         2.24
IVFPQ-m64-nl111-np4                                    2_758.24       488.76     3_247.01       0.3961          1.0421         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_096.07       306.05     2_402.12       0.2672          1.0703         4.72
IVFPQ-m32-nl111-np5                                    1_931.58       328.69     2_260.27       0.2691          1.0695         2.24
IVFPQ-m64-nl111-np5                                    2_758.24       599.51     3_357.75       0.3967          1.0420         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_096.07       368.02     2_464.09       0.2685          1.0698         4.72
IVFPQ-m32-nl111-np8                                    1_931.58       495.68     2_427.26       0.2691          1.0695         2.24
IVFPQ-m64-nl111-np8                                    2_758.24       929.65     3_687.89       0.3969          1.0420         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_096.07       549.14     2_645.21       0.2691          1.0695         4.72
IVFPQ-m32-nl111-np10                                   1_931.58       607.69     2_539.27       0.2691          1.0695         2.24
IVFPQ-m64-nl111-np10                                   2_758.24     1_150.55     3_908.80       0.3969          1.0420         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_096.07       685.29     2_781.36       0.2691          1.0695         4.72
IVFPQ-m32-nl158-np1                                    2_877.33        94.00     2_971.33       0.2517          1.0719         2.34
IVFPQ-m64-nl158-np1                                    3_745.35       144.50     3_889.86       0.3436          1.0487         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_090.93       108.26     3_199.19       0.2485          1.0779         4.82
IVFPQ-m32-nl158-np2                                    2_877.33       146.86     3_024.19       0.2716          1.0669         2.34
IVFPQ-m64-nl158-np2                                    3_745.35       251.13     3_996.49       0.3876          1.0417         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_090.93       167.30     3_258.23       0.2632          1.0715         4.82
IVFPQ-m32-nl158-np4                                    2_877.33       254.69     3_132.02       0.2783          1.0655         2.34
IVFPQ-m64-nl158-np4                                    3_745.35       449.89     4_195.24       0.4052          1.0392         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_090.93       281.58     3_372.51       0.2741          1.0673         4.82
IVFPQ-m32-nl158-np7                                    2_877.33       415.51     3_292.84       0.2788          1.0654         2.34
IVFPQ-m64-nl158-np7                                    3_745.35       750.78     4_496.13       0.4072          1.0389         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_090.93       441.26     3_532.19       0.2784          1.0656         4.82
IVFPQ-m32-nl158-np8                                    2_877.33       467.09     3_344.42       0.2788          1.0654         2.34
IVFPQ-m64-nl158-np8                                    3_745.35       844.83     4_590.18       0.4072          1.0389         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_090.93       497.84     3_588.77       0.2787          1.0655         4.82
IVFPQ-m32-nl158-np12                                   2_877.33       681.96     3_559.29       0.2788          1.0654         2.34
IVFPQ-m64-nl158-np12                                   3_745.35     1_246.32     4_991.67       0.4072          1.0389         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_090.93       713.74     3_804.67       0.2788          1.0654         4.82
IVFPQ-m32-nl223-np1                                    2_129.43        99.78     2_229.21       0.2473          1.0718         2.46
IVFPQ-m64-nl223-np1                                    3_052.83       142.49     3_195.31       0.3302          1.0503         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_393.99       106.52     2_500.51       0.2515          1.0744         4.95
IVFPQ-m32-nl223-np2                                    2_129.43       149.53     2_278.95       0.2708          1.0662         2.46
IVFPQ-m64-nl223-np2                                    3_052.83       238.30     3_291.13       0.3809          1.0423         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_393.99       161.08     2_555.07       0.2625          1.0705         4.95
IVFPQ-m32-nl223-np4                                    2_129.43       251.10     2_380.52       0.2801          1.0641         2.46
IVFPQ-m64-nl223-np4                                    3_052.83       424.04     3_476.87       0.4069          1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_393.99       271.57     2_665.56       0.2732          1.0670         4.95
IVFPQ-m32-nl223-np8                                    2_129.43       458.82     2_588.24       0.2816          1.0638         2.46
IVFPQ-m64-nl223-np8                                    3_052.83       808.46     3_861.29       0.4126          1.0379         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_393.99       473.66     2_867.65       0.2807          1.0642         4.95
IVFPQ-m32-nl223-np11                                   2_129.43       616.45     2_745.88       0.2816          1.0638         2.46
IVFPQ-m64-nl223-np11                                   3_052.83     1_109.90     4_162.72       0.4129          1.0379         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_393.99       629.54     3_023.53       0.2815          1.0639         4.95
IVFPQ-m32-nl223-np14                                   2_129.43       773.34     2_902.76       0.2816          1.0638         2.46
IVFPQ-m64-nl223-np14                                   3_052.83     1_389.82     4_442.64       0.4129          1.0379         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_393.99       791.35     3_185.34       0.2816          1.0638         4.95
IVFPQ-m32-nl316-np1                                    2_498.80       107.04     2_605.84       0.2423          1.0727         2.65
IVFPQ-m64-nl316-np1                                    3_342.04       144.75     3_486.79       0.3138          1.0530         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_692.60       108.61     2_801.21       0.2515          1.0730         5.13
IVFPQ-m32-nl316-np2                                    2_498.80       150.72     2_649.51       0.2684          1.0663         2.65
IVFPQ-m64-nl316-np2                                    3_342.04       235.34     3_577.38       0.3692          1.0438         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_692.60       160.35     2_852.96       0.2644          1.0690         5.13
IVFPQ-m32-nl316-np4                                    2_498.80       250.20     2_748.99       0.2812          1.0634         2.65
IVFPQ-m64-nl316-np4                                    3_342.04       414.38     3_756.41       0.4027          1.0391         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_692.60       266.45     2_959.05       0.2723          1.0669         5.13
IVFPQ-m32-nl316-np8                                    2_498.80       450.65     2_949.44       0.2845          1.0627         2.65
IVFPQ-m64-nl316-np8                                    3_342.04       795.71     4_137.75       0.4144          1.0375         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_692.60       467.31     3_159.91       0.2816          1.0640         5.13
IVFPQ-m32-nl316-np15                                   2_498.80       802.59     3_301.38       0.2847          1.0627         2.65
IVFPQ-m64-nl316-np15                                   3_342.04     1_422.71     4_764.75       0.4156          1.0374         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_692.60       817.27     3_509.87       0.2846          1.0627         5.13
IVFPQ-m32-nl316-np17                                   2_498.80       904.84     3_403.63       0.2847          1.0627         2.65
IVFPQ-m64-nl316-np17                                   3_342.04     1_605.09     4_947.13       0.4156          1.0374         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_692.60       916.83     3_609.43       0.2847          1.0627         5.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-PQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-PQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-PQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-PQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.04     1_256.99     1_325.03       1.0000          1.0000        97.66
SOARPQ-near-np1                                        3_182.40       108.99     3_291.40       0.2486          1.0779         4.82
SOARPQ-near-np2                                        3_182.40       185.81     3_368.22       0.2636          1.0714         4.82
SOARPQ-near-np4                                        3_182.40       281.41     3_463.82       0.2746          1.0671         4.82
SOARPQ-near-np7                                        3_182.40       439.85     3_622.25       0.2786          1.0655         4.82
SOARPQ-near-np8                                        3_182.40       502.11     3_684.52       0.2787          1.0654         4.82
SOARPQ-near-np12                                       3_182.40       732.98     3_915.39       0.2788          1.0654         4.82
SOARPQ-shift0.3-np1                                    3_083.38       107.61     3_190.99       0.2487          1.0778         4.82
SOARPQ-shift0.3-np2                                    3_083.38       174.14     3_257.52       0.2636          1.0714         4.82
SOARPQ-shift0.3-np4                                    3_083.38       279.59     3_362.97       0.2743          1.0672         4.82
SOARPQ-shift0.3-np7                                    3_083.38       437.96     3_521.34       0.2785          1.0656         4.82
SOARPQ-shift0.3-np8                                    3_083.38       501.30     3_584.68       0.2787          1.0655         4.82
SOARPQ-shift0.3-np12                                   3_083.38       722.31     3_805.69       0.2788          1.0654         4.82
SOARPQ-shift0.7-np1                                    3_052.33       108.10     3_160.43       0.2480          1.0781         4.82
SOARPQ-shift0.7-np2                                    3_052.33       168.80     3_221.13       0.2627          1.0717         4.82
SOARPQ-shift0.7-np4                                    3_052.33       279.00     3_331.34       0.2738          1.0675         4.82
SOARPQ-shift0.7-np7                                    3_052.33       441.88     3_494.21       0.2784          1.0656         4.82
SOARPQ-shift0.7-np8                                    3_052.33       489.27     3_541.60       0.2786          1.0655         4.82
SOARPQ-shift0.7-np12                                   3_052.33       727.21     3_779.54       0.2788          1.0654         4.82
SOARPQ-orth1-np1                                       3_080.23       108.12     3_188.35       0.2474          1.0783         4.82
SOARPQ-orth1-np2                                       3_080.23       167.77     3_248.00       0.2617          1.0721         4.82
SOARPQ-orth1-np4                                       3_080.23       278.77     3_359.00       0.2733          1.0677         4.82
SOARPQ-orth1-np7                                       3_080.23       436.79     3_517.02       0.2782          1.0657         4.82
SOARPQ-orth1-np8                                       3_080.23       490.51     3_570.73       0.2786          1.0655         4.82
SOARPQ-orth1-np12                                      3_080.23       707.89     3_788.12       0.2788          1.0654         4.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.66     1_241.77     1_309.43       1.0000          1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_745.20       126.19     1_871.39       0.4760          1.0769         2.24
IVFPQ-m64-nl111-np1                                    2_523.39       225.35     2_748.74       0.6153          1.0391         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_874.46       129.18     2_003.64       0.4682          1.0907         4.72
IVFPQ-m32-nl111-np2                                    1_745.20       172.86     1_918.06       0.4805          1.0749         2.24
IVFPQ-m64-nl111-np2                                    2_523.39       307.41     2_830.80       0.6224          1.0364         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_874.46       179.99     2_054.45       0.4801          1.0760         4.72
IVFPQ-m32-nl111-np4                                    1_745.20       264.45     2_009.65       0.4808          1.0748         2.24
IVFPQ-m64-nl111-np4                                    2_523.39       478.61     3_001.99       0.6230          1.0362         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_874.46       268.13     2_142.59       0.4808          1.0748         4.72
IVFPQ-m32-nl111-np5                                    1_745.20       307.56     2_052.76       0.4808          1.0748         2.24
IVFPQ-m64-nl111-np5                                    2_523.39       559.43     3_082.82       0.6230          1.0362         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_874.46       316.90     2_191.36       0.4808          1.0748         4.72
IVFPQ-m32-nl111-np8                                    1_745.20       445.58     2_190.78       0.4808          1.0748         2.24
IVFPQ-m64-nl111-np8                                    2_523.39       821.73     3_345.12       0.6230          1.0362         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_874.46       468.10     2_342.56       0.4808          1.0748         4.72
IVFPQ-m32-nl111-np10                                   1_745.20       540.97     2_286.17       0.4808          1.0748         2.24
IVFPQ-m64-nl111-np10                                   2_523.39       982.10     3_505.49       0.6230          1.0362         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_874.46       584.44     2_458.90       0.4808          1.0748         4.72
IVFPQ-m32-nl158-np1                                    2_603.06       129.32     2_732.38       0.4787          1.0753         2.34
IVFPQ-m64-nl158-np1                                    3_448.12       226.31     3_674.43       0.6161          1.0387         3.86
SOARPQ-shift0.5-m32-nl158-np1                          2_822.17       131.47     2_953.63       0.4799          1.0774         4.82
IVFPQ-m32-nl158-np2                                    2_603.06       175.49     2_778.55       0.4846          1.0729         2.34
IVFPQ-m64-nl158-np2                                    3_448.12       308.09     3_756.21       0.6260          1.0352         3.86
SOARPQ-shift0.5-m32-nl158-np2                          2_822.17       179.54     3_001.71       0.4842          1.0733         4.82
IVFPQ-m32-nl158-np4                                    2_603.06       265.22     2_868.28       0.4850          1.0727         2.34
IVFPQ-m64-nl158-np4                                    3_448.12       470.67     3_918.79       0.6267          1.0349         3.86
SOARPQ-shift0.5-m32-nl158-np4                          2_822.17       270.14     3_092.31       0.4848          1.0728         4.82
IVFPQ-m32-nl158-np7                                    2_603.06       406.38     3_009.45       0.4850          1.0727         2.34
IVFPQ-m64-nl158-np7                                    3_448.12       727.57     4_175.68       0.6267          1.0349         3.86
SOARPQ-shift0.5-m32-nl158-np7                          2_822.17       437.19     3_259.36       0.4850          1.0727         4.82
IVFPQ-m32-nl158-np8                                    2_603.06       453.25     3_056.32       0.4850          1.0727         2.34
IVFPQ-m64-nl158-np8                                    3_448.12       804.67     4_252.78       0.6267          1.0349         3.86
SOARPQ-shift0.5-m32-nl158-np8                          2_822.17       450.88     3_273.05       0.4850          1.0727         4.82
IVFPQ-m32-nl158-np12                                   2_603.06       653.30     3_256.37       0.4850          1.0727         2.34
IVFPQ-m64-nl158-np12                                   3_448.12     1_180.74     4_628.86       0.6267          1.0349         3.86
SOARPQ-shift0.5-m32-nl158-np12                         2_822.17       649.56     3_471.73       0.4850          1.0727         4.82
IVFPQ-m32-nl223-np1                                    2_270.28       103.78     2_374.06       0.3653          1.1129         2.46
IVFPQ-m64-nl223-np1                                    3_253.72       155.88     3_409.60       0.4294          1.0826         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_489.71       115.29     2_605.01       0.4284          1.0921         4.95
IVFPQ-m32-nl223-np2                                    2_270.28       155.53     2_425.81       0.4367          1.0869         2.46
IVFPQ-m64-nl223-np2                                    3_253.72       273.47     3_527.18       0.5441          1.0522         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_489.71       177.77     2_667.48       0.4690          1.0777         4.95
IVFPQ-m32-nl223-np4                                    2_270.28       273.97     2_544.25       0.4768          1.0750         2.46
IVFPQ-m64-nl223-np4                                    3_253.72       462.27     3_715.99       0.6162          1.0373         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_489.71       279.92     2_769.64       0.4822          1.0737         4.95
IVFPQ-m32-nl223-np8                                    2_270.28       465.47     2_735.75       0.4831          1.0732         2.46
IVFPQ-m64-nl223-np8                                    3_253.72       809.91     4_063.62       0.6291          1.0346         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_489.71       465.76     2_955.48       0.4832          1.0732         4.95
IVFPQ-m32-nl223-np11                                   2_270.28       608.45     2_878.73       0.4832          1.0732         2.46
IVFPQ-m64-nl223-np11                                   3_253.72     1_073.64     4_327.35       0.6294          1.0346         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_489.71       610.16     3_099.87       0.4832          1.0732         4.95
IVFPQ-m32-nl223-np14                                   2_270.28       753.20     3_023.48       0.4832          1.0732         2.46
IVFPQ-m64-nl223-np14                                   3_253.72     1_332.77     4_586.48       0.6294          1.0346         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_489.71       750.33     3_240.04       0.4832          1.0732         4.95
IVFPQ-m32-nl316-np1                                    2_628.88       107.41     2_736.29       0.3328          1.1254         2.65
IVFPQ-m64-nl316-np1                                    3_478.12       151.98     3_630.10       0.3752          1.0976         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_877.38       118.88     2_996.26       0.4036          1.0986         5.13
IVFPQ-m32-nl316-np2                                    2_628.88       157.58     2_786.45       0.4088          1.0954         2.65
IVFPQ-m64-nl316-np2                                    3_478.12       243.94     3_722.06       0.4933          1.0630         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_877.38       169.53     3_046.91       0.4528          1.0825         5.13
IVFPQ-m32-nl316-np4                                    2_628.88       258.25     2_887.12       0.4604          1.0795         2.65
IVFPQ-m64-nl316-np4                                    3_478.12       432.26     3_910.38       0.5828          1.0436         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_877.38       279.52     3_156.90       0.4788          1.0746         5.13
IVFPQ-m32-nl316-np8                                    2_628.88       478.37     3_107.25       0.4833          1.0730         2.65
IVFPQ-m64-nl316-np8                                    3_478.12       801.30     4_279.42       0.6285          1.0346         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_877.38       464.76     3_342.14       0.4841          1.0729         5.13
IVFPQ-m32-nl316-np15                                   2_628.88       795.80     3_424.68       0.4846          1.0727         2.65
IVFPQ-m64-nl316-np15                                   3_478.12     1_381.85     4_859.97       0.6315          1.0340         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_877.38       786.80     3_664.18       0.4846          1.0727         5.13
IVFPQ-m32-nl316-np17                                   2_628.88       893.79     3_522.67       0.4846          1.0727         2.65
IVFPQ-m64-nl316-np17                                   3_478.12     1_562.86     5_040.98       0.6315          1.0340         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_877.38       887.96     3_765.34       0.4846          1.0727         5.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-PQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-PQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-PQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-PQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.66     1_241.77     1_309.43       1.0000          1.0000        97.66
SOARPQ-near-np1                                        2_820.02       130.67     2_950.70       0.4804          1.0769         4.82
SOARPQ-near-np2                                        2_820.02       179.68     2_999.70       0.4843          1.0733         4.82
SOARPQ-near-np4                                        2_820.02       265.30     3_085.33       0.4848          1.0728         4.82
SOARPQ-near-np7                                        2_820.02       404.97     3_225.00       0.4850          1.0727         4.82
SOARPQ-near-np8                                        2_820.02       445.47     3_265.50       0.4850          1.0727         4.82
SOARPQ-near-np12                                       2_820.02       642.04     3_462.07       0.4850          1.0727         4.82
SOARPQ-shift0.3-np1                                    2_810.33       130.36     2_940.69       0.4800          1.0773         4.82
SOARPQ-shift0.3-np2                                    2_810.33       178.05     2_988.38       0.4842          1.0733         4.82
SOARPQ-shift0.3-np4                                    2_810.33       263.73     3_074.07       0.4848          1.0728         4.82
SOARPQ-shift0.3-np7                                    2_810.33       397.39     3_207.72       0.4850          1.0727         4.82
SOARPQ-shift0.3-np8                                    2_810.33       442.61     3_252.94       0.4850          1.0727         4.82
SOARPQ-shift0.3-np12                                   2_810.33       631.83     3_442.16       0.4850          1.0727         4.82
SOARPQ-shift0.7-np1                                    2_786.86       131.23     2_918.09       0.4797          1.0775         4.82
SOARPQ-shift0.7-np2                                    2_786.86       179.56     2_966.42       0.4841          1.0734         4.82
SOARPQ-shift0.7-np4                                    2_786.86       268.60     3_055.46       0.4848          1.0728         4.82
SOARPQ-shift0.7-np7                                    2_786.86       406.09     3_192.95       0.4850          1.0727         4.82
SOARPQ-shift0.7-np8                                    2_786.86       441.09     3_227.95       0.4850          1.0727         4.82
SOARPQ-shift0.7-np12                                   2_786.86       631.78     3_418.64       0.4850          1.0727         4.82
SOARPQ-orth1-np1                                       2_797.00       132.89     2_929.89       0.4799          1.0775         4.82
SOARPQ-orth1-np2                                       2_797.00       179.62     2_976.62       0.4841          1.0733         4.82
SOARPQ-orth1-np4                                       2_797.00       267.05     3_064.05       0.4848          1.0728         4.82
SOARPQ-orth1-np7                                       2_797.00       398.03     3_195.03       0.4850          1.0727         4.82
SOARPQ-orth1-np8                                       2_797.00       442.02     3_239.02       0.4850          1.0727         4.82
SOARPQ-orth1-np12                                      2_797.00       633.16     3_430.16       0.4850          1.0727         4.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.57     1_284.15     1_353.72       1.0000          1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_924.33       100.56     2_024.89       0.7055          1.1845         2.24
IVFPQ-m64-nl111-np1                                    2_753.62       163.02     2_916.64       0.7175          1.1739         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_127.88       123.02     2_250.90       0.8142          1.0740         4.72
IVFPQ-m32-nl111-np2                                    1_924.33       162.10     2_086.43       0.8212          1.0638         2.24
IVFPQ-m64-nl111-np2                                    2_753.62       285.13     3_038.75       0.8434          1.0521         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_127.88       204.36     2_332.24       0.8488          1.0446         4.72
IVFPQ-m32-nl111-np4                                    1_924.33       283.51     2_207.83       0.8539          1.0384         2.24
IVFPQ-m64-nl111-np4                                    2_753.62       531.65     3_285.27       0.8802          1.0260         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_127.88       361.63     2_489.51       0.8555          1.0387         4.72
IVFPQ-m32-nl111-np5                                    1_924.33       353.95     2_278.28       0.8556          1.0373         2.24
IVFPQ-m64-nl111-np5                                    2_753.62       663.70     3_417.32       0.8821          1.0249         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_127.88       434.67     2_562.55       0.8559          1.0382         4.72
IVFPQ-m32-nl111-np8                                    1_924.33       530.27     2_454.60       0.8566          1.0367         2.24
IVFPQ-m64-nl111-np8                                    2_753.62     1_035.38     3_789.00       0.8833          1.0243         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_127.88       642.52     2_770.40       0.8565          1.0371         4.72
IVFPQ-m32-nl111-np10                                   1_924.33       667.88     2_592.21       0.8566          1.0367         2.24
IVFPQ-m64-nl111-np10                                   2_753.62     1_262.85     4_016.47       0.8834          1.0243         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_127.88       770.45     2_898.33       0.8566          1.0369         4.72
IVFPQ-m32-nl158-np1                                    2_943.37        96.68     3_040.05       0.7125          1.1767         2.34
IVFPQ-m64-nl158-np1                                    3_780.93       150.01     3_930.94       0.7212          1.1686         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_147.05       112.96     3_260.01       0.8237          1.0695         4.82
IVFPQ-m32-nl158-np2                                    2_943.37       152.28     3_095.65       0.8339          1.0567         2.34
IVFPQ-m64-nl158-np2                                    3_780.93       257.96     4_038.89       0.8510          1.0477         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_147.05       187.04     3_334.09       0.8617          1.0397         4.82
IVFPQ-m32-nl158-np4                                    2_943.37       270.19     3_213.55       0.8687          1.0320         2.34
IVFPQ-m64-nl158-np4                                    3_780.93       486.35     4_267.28       0.8891          1.0225         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_147.05       313.45     3_460.49       0.8703          1.0331         4.82
IVFPQ-m32-nl158-np7                                    2_943.37       436.15     3_379.51       0.8726          1.0297         2.34
IVFPQ-m64-nl158-np7                                    3_780.93       820.83     4_601.76       0.8936          1.0202         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_147.05       508.34     3_655.39       0.8724          1.0305         4.82
IVFPQ-m32-nl158-np8                                    2_943.37       495.04     3_438.41       0.8730          1.0295         2.34
IVFPQ-m64-nl158-np8                                    3_780.93       912.77     4_693.70       0.8939          1.0201         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_147.05       572.20     3_719.24       0.8727          1.0302         4.82
IVFPQ-m32-nl158-np12                                   2_943.37       723.07     3_666.44       0.8731          1.0294         2.34
IVFPQ-m64-nl158-np12                                   3_780.93     1_347.06     5_127.98       0.8941          1.0200         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_147.05       821.58     3_968.63       0.8731          1.0296         4.82
IVFPQ-m32-nl223-np1                                    2_027.17        99.47     2_126.63       0.6875          1.1973         2.46
IVFPQ-m64-nl223-np1                                    2_897.66       143.28     3_040.94       0.6935          1.1897         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_238.76       106.81     2_345.56       0.8134          1.0794         4.95
IVFPQ-m32-nl223-np2                                    2_027.17       146.96     2_174.12       0.8245          1.0642         2.46
IVFPQ-m64-nl223-np2                                    2_897.66       237.80     3_135.46       0.8396          1.0562         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_238.76       166.57     2_405.32       0.8646          1.0397         4.95
IVFPQ-m32-nl223-np4                                    2_027.17       250.65     2_277.82       0.8726          1.0303         2.46
IVFPQ-m64-nl223-np4                                    2_897.66       434.34     3_332.00       0.8924          1.0218         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_238.76       281.68     2_520.44       0.8758          1.0314         4.95
IVFPQ-m32-nl223-np8                                    2_027.17       461.89     2_489.05       0.8792          1.0267         2.46
IVFPQ-m64-nl223-np8                                    2_897.66       827.20     3_724.86       0.8998          1.0180         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_238.76       513.31     2_752.06       0.8786          1.0279         4.95
IVFPQ-m32-nl223-np11                                   2_027.17       629.82     2_656.98       0.8795          1.0266         2.46
IVFPQ-m64-nl223-np11                                   2_897.66     1_138.48     4_036.13       0.9003          1.0178         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_238.76       686.17     2_924.93       0.8792          1.0271         4.95
IVFPQ-m32-nl223-np14                                   2_027.17       795.71     2_822.88       0.8795          1.0265         2.46
IVFPQ-m64-nl223-np14                                   2_897.66     1_421.91     4_319.57       0.9003          1.0178         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_238.76       865.80     3_104.55       0.8794          1.0268         4.95
IVFPQ-m32-nl316-np1                                    2_276.88       113.30     2_390.18       0.6725          1.2102         2.65
IVFPQ-m64-nl316-np1                                    3_096.24       145.96     3_242.21       0.6767          1.2049         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_455.79       108.50     2_564.29       0.8093          1.0840         5.13
IVFPQ-m32-nl316-np2                                    2_276.88       149.66     2_426.54       0.8228          1.0663         2.65
IVFPQ-m64-nl316-np2                                    3_096.24       234.34     3_330.58       0.8330          1.0605         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_455.79       160.90     2_616.70       0.8708          1.0389         5.13
IVFPQ-m32-nl316-np4                                    2_276.88       248.60     2_525.48       0.8817          1.0267         2.65
IVFPQ-m64-nl316-np4                                    3_096.24       415.35     3_511.60       0.8960          1.0207         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_455.79       269.61     2_725.40       0.8858          1.0287         5.13
IVFPQ-m32-nl316-np8                                    2_276.88       451.24     2_728.12       0.8910          1.0218         2.65
IVFPQ-m64-nl316-np8                                    3_096.24       790.61     3_886.85       0.9056          1.0158         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_455.79       486.89     2_942.68       0.8900          1.0240         5.13
IVFPQ-m32-nl316-np15                                   2_276.88       813.58     3_090.46       0.8917          1.0215         2.65
IVFPQ-m64-nl316-np15                                   3_096.24     1_436.01     4_532.26       0.9064          1.0155         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_455.79       851.84     3_307.64       0.8915          1.0218         5.13
IVFPQ-m32-nl316-np17                                   2_276.88       905.06     3_181.94       0.8917          1.0215         2.65
IVFPQ-m64-nl316-np17                                   3_096.24     1_627.49     4_723.74       0.9064          1.0155         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_455.79       962.98     3_418.77       0.8916          1.0216         5.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-PQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-PQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-PQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-PQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.57     1_284.15     1_353.72       1.0000          1.0000        97.66
SOARPQ-near-np1                                        3_105.98       112.67     3_218.65       0.8232          1.0711         4.82
SOARPQ-near-np2                                        3_105.98       183.48     3_289.46       0.8631          1.0379         4.82
SOARPQ-near-np4                                        3_105.98       313.30     3_419.28       0.8714          1.0316         4.82
SOARPQ-near-np7                                        3_105.98       508.39     3_614.37       0.8727          1.0300         4.82
SOARPQ-near-np8                                        3_105.98       567.22     3_673.20       0.8729          1.0298         4.82
SOARPQ-near-np12                                       3_105.98       814.55     3_920.53       0.8731          1.0295         4.82
SOARPQ-shift0.3-np1                                    3_054.12       113.93     3_168.05       0.8257          1.0677         4.82
SOARPQ-shift0.3-np2                                    3_054.12       181.76     3_235.89       0.8627          1.0386         4.82
SOARPQ-shift0.3-np4                                    3_054.12       312.65     3_366.78       0.8707          1.0325         4.82
SOARPQ-shift0.3-np7                                    3_054.12       504.27     3_558.39       0.8725          1.0304         4.82
SOARPQ-shift0.3-np8                                    3_054.12       563.76     3_617.89       0.8727          1.0301         4.82
SOARPQ-shift0.3-np12                                   3_054.12       805.55     3_859.68       0.8731          1.0296         4.82
SOARPQ-shift0.7-np1                                    3_091.19       111.61     3_202.80       0.8207          1.0725         4.82
SOARPQ-shift0.7-np2                                    3_091.19       182.93     3_274.12       0.8608          1.0409         4.82
SOARPQ-shift0.7-np4                                    3_091.19       311.39     3_402.58       0.8699          1.0337         4.82
SOARPQ-shift0.7-np7                                    3_091.19       500.46     3_591.65       0.8723          1.0308         4.82
SOARPQ-shift0.7-np8                                    3_091.19       562.88     3_654.07       0.8726          1.0304         4.82
SOARPQ-shift0.7-np12                                   3_091.19       810.81     3_902.01       0.8730          1.0296         4.82
SOARPQ-orth1-np1                                       3_072.36       113.76     3_186.12       0.8233          1.0708         4.82
SOARPQ-orth1-np2                                       3_072.36       182.20     3_254.57       0.8623          1.0392         4.82
SOARPQ-orth1-np4                                       3_072.36       315.13     3_387.49       0.8707          1.0327         4.82
SOARPQ-orth1-np7                                       3_072.36       502.14     3_574.50       0.8725          1.0303         4.82
SOARPQ-orth1-np8                                       3_072.36       561.85     3_634.21       0.8727          1.0301         4.82
SOARPQ-orth1-np12                                      3_072.36       802.69     3_875.05       0.8731          1.0296         4.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.38     1_269.45     1_342.82       1.0000          1.0000        97.85
IVFPQ-m32-nl111-np1                                    1_819.98        97.35     1_917.33       0.7687          1.1400         2.24
IVFPQ-m64-nl111-np1                                    2_635.75       157.16     2_792.92       0.7773          1.1327         3.77
SOARPQ-orth1-m32-nl111-np1                             1_986.92       116.07     2_102.99       0.8461          1.0725         4.72
IVFPQ-m32-nl111-np2                                    1_819.98       161.31     1_981.29       0.8642          1.0449         2.24
IVFPQ-m64-nl111-np2                                    2_635.75       271.70     2_907.46       0.8774          1.0371         3.77
SOARPQ-orth1-m32-nl111-np2                             1_986.92       194.24     2_181.16       0.8750          1.0430         4.72
IVFPQ-m32-nl111-np4                                    1_819.98       275.36     2_095.34       0.8829          1.0309         2.24
IVFPQ-m64-nl111-np4                                    2_635.75       500.52     3_136.27       0.8971          1.0232         3.77
SOARPQ-orth1-m32-nl111-np4                             1_986.92       332.17     2_319.09       0.8820          1.0345         4.72
IVFPQ-m32-nl111-np5                                    1_819.98       329.81     2_149.79       0.8838          1.0304         2.24
IVFPQ-m64-nl111-np5                                    2_635.75       620.35     3_256.10       0.8981          1.0227         3.77
SOARPQ-orth1-m32-nl111-np5                             1_986.92       404.06     2_390.98       0.8830          1.0332         4.72
IVFPQ-m32-nl111-np8                                    1_819.98       511.52     2_331.50       0.8844          1.0302         2.24
IVFPQ-m64-nl111-np8                                    2_635.75       974.07     3_609.82       0.8987          1.0225         3.77
SOARPQ-orth1-m32-nl111-np8                             1_986.92       592.50     2_579.42       0.8840          1.0311         4.72
IVFPQ-m32-nl111-np10                                   1_819.98       626.60     2_446.58       0.8844          1.0301         2.24
IVFPQ-m64-nl111-np10                                   2_635.75     1_192.48     3_828.24       0.8987          1.0225         3.77
SOARPQ-orth1-m32-nl111-np10                            1_986.92       752.28     2_739.20       0.8843          1.0305         4.72
IVFPQ-m32-nl158-np1                                    2_821.56        94.58     2_916.14       0.7527          1.1567         2.34
IVFPQ-m64-nl158-np1                                    3_674.04       146.52     3_820.56       0.7586          1.1514         3.86
SOARPQ-orth1-m32-nl158-np1                             3_027.22       107.66     3_134.89       0.8448          1.0760         4.82
IVFPQ-m32-nl158-np2                                    2_821.56       147.42     2_968.98       0.8658          1.0456         2.34
IVFPQ-m64-nl158-np2                                    3_674.04       246.09     3_920.14       0.8753          1.0404         3.86
SOARPQ-orth1-m32-nl158-np2                             3_027.22       170.29     3_197.52       0.8833          1.0399         4.82
IVFPQ-m32-nl158-np4                                    2_821.56       255.77     3_077.32       0.8934          1.0250         2.34
IVFPQ-m64-nl158-np4                                    3_674.04       477.20     4_151.24       0.9045          1.0197         3.86
SOARPQ-orth1-m32-nl158-np4                             3_027.22       298.18     3_325.41       0.8929          1.0292         4.82
IVFPQ-m32-nl158-np7                                    2_821.56       419.04     3_240.59       0.8955          1.0238         2.34
IVFPQ-m64-nl158-np7                                    3_674.04       765.82     4_439.87       0.9069          1.0185         3.86
SOARPQ-orth1-m32-nl158-np7                             3_027.22       483.84     3_511.07       0.8951          1.0254         4.82
IVFPQ-m32-nl158-np8                                    2_821.56       475.48     3_297.04       0.8956          1.0238         2.34
IVFPQ-m64-nl158-np8                                    3_674.04       872.68     4_546.73       0.9070          1.0184         3.86
SOARPQ-orth1-m32-nl158-np8                             3_027.22       545.92     3_573.15       0.8952          1.0250         4.82
IVFPQ-m32-nl158-np12                                   2_821.56       711.71     3_533.27       0.8957          1.0238         2.34
IVFPQ-m64-nl158-np12                                   3_674.04     1_295.87     4_969.91       0.9071          1.0184         3.86
SOARPQ-orth1-m32-nl158-np12                            3_027.22       780.17     3_807.39       0.8956          1.0241         4.82
IVFPQ-m32-nl223-np1                                    2_133.24        97.93     2_231.17       0.7283          1.1815         2.46
IVFPQ-m64-nl223-np1                                    2_968.08       140.69     3_108.77       0.7317          1.1772         3.99
SOARPQ-orth1-m32-nl223-np1                             2_358.48       103.86     2_462.34       0.8393          1.0777         4.95
IVFPQ-m32-nl223-np2                                    2_133.24       149.91     2_283.16       0.8613          1.0501         2.46
IVFPQ-m64-nl223-np2                                    2_968.08       234.11     3_202.19       0.8679          1.0460         3.99
SOARPQ-orth1-m32-nl223-np2                             2_358.48       160.06     2_518.53       0.8871          1.0370         4.95
IVFPQ-m32-nl223-np4                                    2_133.24       243.70     2_376.94       0.8979          1.0231         2.46
IVFPQ-m64-nl223-np4                                    2_968.08       418.47     3_386.55       0.9065          1.0191         3.99
SOARPQ-orth1-m32-nl223-np4                             2_358.48       280.12     2_638.60       0.8976          1.0276         4.95
IVFPQ-m32-nl223-np8                                    2_133.24       448.66     2_581.90       0.9010          1.0214         2.46
IVFPQ-m64-nl223-np8                                    2_968.08       804.28     3_772.36       0.9099          1.0173         3.99
SOARPQ-orth1-m32-nl223-np8                             2_358.48       487.90     2_846.38       0.9006          1.0227         4.95
IVFPQ-m32-nl223-np11                                   2_133.24       612.09     2_745.33       0.9011          1.0214         2.46
IVFPQ-m64-nl223-np11                                   2_968.08     1_092.24     4_060.32       0.9101          1.0173         3.99
SOARPQ-orth1-m32-nl223-np11                            2_358.48       653.29     3_011.77       0.9009          1.0218         4.95
IVFPQ-m32-nl223-np14                                   2_133.24       761.95     2_895.20       0.9011          1.0214         2.46
IVFPQ-m64-nl223-np14                                   2_968.08     1_380.93     4_349.01       0.9101          1.0173         3.99
SOARPQ-orth1-m32-nl223-np14                            2_358.48       818.16     3_176.64       0.9011          1.0216         4.95
IVFPQ-m32-nl316-np1                                    2_449.01       102.04     2_551.05       0.7037          1.2091         2.65
IVFPQ-m64-nl316-np1                                    3_291.84       145.76     3_437.60       0.7071          1.2044         4.17
SOARPQ-orth1-m32-nl316-np1                             2_740.51       108.30     2_848.81       0.8250          1.0882         5.13
IVFPQ-m32-nl316-np2                                    2_449.01       150.06     2_599.08       0.8495          1.0587         2.65
IVFPQ-m64-nl316-np2                                    3_291.84       233.40     3_525.23       0.8573          1.0541         4.17
SOARPQ-orth1-m32-nl316-np2                             2_740.51       160.23     2_900.75       0.8855          1.0375         5.13
IVFPQ-m32-nl316-np4                                    2_449.01       245.52     2_694.53       0.8982          1.0231         2.65
IVFPQ-m64-nl316-np4                                    3_291.84       406.76     3_698.59       0.9091          1.0184         4.17
SOARPQ-orth1-m32-nl316-np4                             2_740.51       262.00     3_002.51       0.8994          1.0269         5.13
IVFPQ-m32-nl316-np8                                    2_449.01       437.29     2_886.30       0.9033          1.0203         2.65
IVFPQ-m64-nl316-np8                                    3_291.84       771.48     4_063.32       0.9144          1.0156         4.17
SOARPQ-orth1-m32-nl316-np8                             2_740.51       465.19     3_205.70       0.9028          1.0220         5.13
IVFPQ-m32-nl316-np15                                   2_449.01       785.32     3_234.33       0.9036          1.0202         2.65
IVFPQ-m64-nl316-np15                                   3_291.84     1_423.51     4_715.34       0.9147          1.0155         4.17
SOARPQ-orth1-m32-nl316-np15                            2_740.51       839.45     3_579.97       0.9035          1.0204         5.13
IVFPQ-m32-nl316-np17                                   2_449.01       882.25     3_331.26       0.9036          1.0202         2.65
IVFPQ-m64-nl316-np17                                   3_291.84     1_609.02     4_900.85       0.9147          1.0155         4.17
SOARPQ-orth1-m32-nl316-np17                            2_740.51       933.88     3_674.39       0.9035          1.0203         5.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-PQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-PQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-PQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-PQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.38     1_269.45     1_342.82       1.0000          1.0000        97.85
SOARPQ-near-np1                                        2_997.49       107.28     3_104.76       0.8512          1.0652         4.82
SOARPQ-near-np2                                        2_997.49       175.47     3_172.96       0.8858          1.0343         4.82
SOARPQ-near-np4                                        2_997.49       293.47     3_290.96       0.8938          1.0267         4.82
SOARPQ-near-np7                                        2_997.49       471.34     3_468.83       0.8953          1.0244         4.82
SOARPQ-near-np8                                        2_997.49       540.37     3_537.85       0.8954          1.0242         4.82
SOARPQ-near-np12                                       2_997.49       765.32     3_762.80       0.8957          1.0239         4.82
SOARPQ-shift0.3-np1                                    3_047.76       107.30     3_155.07       0.8500          1.0681         4.82
SOARPQ-shift0.3-np2                                    3_047.76       172.19     3_219.96       0.8840          1.0373         4.82
SOARPQ-shift0.3-np4                                    3_047.76       292.03     3_339.79       0.8927          1.0284         4.82
SOARPQ-shift0.3-np7                                    3_047.76       472.90     3_520.66       0.8950          1.0250         4.82
SOARPQ-shift0.3-np8                                    3_047.76       558.15     3_605.91       0.8952          1.0247         4.82
SOARPQ-shift0.3-np12                                   3_047.76       779.37     3_827.14       0.8956          1.0240         4.82
SOARPQ-shift0.7-np1                                    2_961.15       108.85     3_070.00       0.8446          1.0765         4.82
SOARPQ-shift0.7-np2                                    2_961.15       169.25     3_130.40       0.8805          1.0432         4.82
SOARPQ-shift0.7-np4                                    2_961.15       291.93     3_253.09       0.8912          1.0314         4.82
SOARPQ-shift0.7-np7                                    2_961.15       472.38     3_433.53       0.8945          1.0262         4.82
SOARPQ-shift0.7-np8                                    2_961.15       537.25     3_498.40       0.8949          1.0255         4.82
SOARPQ-shift0.7-np12                                   2_961.15       765.20     3_726.35       0.8955          1.0242         4.82
SOARPQ-orth1-np1                                       2_973.83       107.15     3_080.99       0.8448          1.0760         4.82
SOARPQ-orth1-np2                                       2_973.83       170.01     3_143.84       0.8833          1.0399         4.82
SOARPQ-orth1-np4                                       2_973.83       291.73     3_265.56       0.8929          1.0292         4.82
SOARPQ-orth1-np7                                       2_973.83       477.98     3_451.82       0.8951          1.0254         4.82
SOARPQ-orth1-np8                                       2_973.83       532.04     3_505.88       0.8952          1.0250         4.82
SOARPQ-orth1-np12                                      2_973.83       760.86     3_734.70       0.8956          1.0241         4.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.66     1_186.69     1_254.35       1.0000          1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_894.52       426.72     8_321.24       0.2557          1.0717         3.49
IVFOPQ-m64-nl111-np1                                  13_944.05       479.38    14_423.43       0.3496          1.0492         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_307.21       458.87     9_766.08       0.2509          1.0762         5.98
IVFOPQ-m32-nl111-np2                                   7_894.52       479.78     8_374.30       0.2719          1.0674         3.49
IVFOPQ-m64-nl111-np2                                  13_944.05       593.35    14_537.40       0.3846          1.0433         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_307.21       524.37     9_831.59       0.2665          1.0700         5.98
IVFOPQ-m32-nl111-np4                                   7_894.52       593.89     8_488.41       0.2758          1.0666         3.49
IVFOPQ-m64-nl111-np4                                  13_944.05       813.21    14_757.26       0.3965          1.0416         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_307.21       635.37     9_942.58       0.2743          1.0671         5.98
IVFOPQ-m32-nl111-np5                                   7_894.52       657.24     8_551.76       0.2759          1.0665         3.49
IVFOPQ-m64-nl111-np5                                  13_944.05       923.66    14_867.71       0.3972          1.0415         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_307.21       711.52    10_018.73       0.2753          1.0667         5.98
IVFOPQ-m32-nl111-np8                                   7_894.52       827.53     8_722.05       0.2759          1.0665         3.49
IVFOPQ-m64-nl111-np8                                  13_944.05     1_254.02    15_198.08       0.3974          1.0415         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_307.21       898.38    10_205.60       0.2759          1.0665         5.98
IVFOPQ-m32-nl111-np10                                  7_894.52       925.82     8_820.34       0.2759          1.0665         3.49
IVFOPQ-m64-nl111-np10                                 13_944.05     1_464.74    15_408.79       0.3974          1.0415         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_307.21     1_009.41    10_316.63       0.2759          1.0665         5.98
IVFOPQ-m32-nl158-np1                                   9_769.24       421.41    10_190.65       0.2553          1.0703         3.84
IVFOPQ-m64-nl158-np1                                  15_750.26       481.86    16_232.12       0.3436          1.0486         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_873.25       440.67    11_313.92       0.2532          1.0747         6.32
IVFOPQ-m32-nl158-np2                                   9_769.24       470.34    10_239.58       0.2750          1.0655         3.84
IVFOPQ-m64-nl158-np2                                  15_750.26       570.58    16_320.84       0.3879          1.0416         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_873.25       516.02    11_389.27       0.2675          1.0692         6.32
IVFOPQ-m32-nl158-np4                                   9_769.24       574.39    10_343.64       0.2815          1.0640         3.84
IVFOPQ-m64-nl158-np4                                  15_750.26       769.31    16_519.57       0.4061          1.0390         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_873.25       611.59    11_484.84       0.2775          1.0655         6.32
IVFOPQ-m32-nl158-np7                                   9_769.24       742.80    10_512.05       0.2820          1.0639         3.84
IVFOPQ-m64-nl158-np7                                  15_750.26     1_081.23    16_831.50       0.4082          1.0388         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_873.25       772.84    11_646.09       0.2816          1.0641         6.32
IVFOPQ-m32-nl158-np8                                   9_769.24       783.66    10_552.90       0.2820          1.0639         3.84
IVFOPQ-m64-nl158-np8                                  15_750.26     1_169.18    16_919.44       0.4083          1.0387         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_873.25       837.17    11_710.42       0.2819          1.0640         6.32
IVFOPQ-m32-nl158-np12                                  9_769.24       988.44    10_757.68       0.2820          1.0639         3.84
IVFOPQ-m64-nl158-np12                                 15_750.26     1_560.54    17_310.80       0.4083          1.0387         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_873.25     1_053.82    11_927.06       0.2820          1.0639         6.32
IVFOPQ-m32-nl223-np1                                   8_789.59       425.67     9_215.26       0.2503          1.0707         3.96
IVFOPQ-m64-nl223-np1                                  14_789.61       465.22    15_254.84       0.3307          1.0501         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_185.94       434.69    10_620.63       0.2567          1.0724         6.45
IVFOPQ-m32-nl223-np2                                   8_789.59       469.31     9_258.90       0.2743          1.0650         3.96
IVFOPQ-m64-nl223-np2                                  14_789.61       560.87    15_350.49       0.3813          1.0421         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_185.94       488.62    10_674.56       0.2687          1.0685         6.45
IVFOPQ-m32-nl223-np4                                   8_789.59       575.60     9_365.19       0.2843          1.0629         3.96
IVFOPQ-m64-nl223-np4                                  14_789.61       751.99    15_541.61       0.4076          1.0384         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_185.94       601.33    10_787.27       0.2782          1.0655         6.45
IVFOPQ-m32-nl223-np8                                   8_789.59       767.61     9_557.20       0.2859          1.0625         3.96
IVFOPQ-m64-nl223-np8                                  14_789.61     1_124.29    15_913.90       0.4137          1.0377         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_185.94       815.81    11_001.75       0.2851          1.0629         6.45
IVFOPQ-m32-nl223-np11                                  8_789.59       912.76     9_702.34       0.2859          1.0625         3.96
IVFOPQ-m64-nl223-np11                                 14_789.61     1_400.66    16_190.27       0.4139          1.0376         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_185.94     1_002.56    11_188.50       0.2858          1.0626         6.45
IVFOPQ-m32-nl223-np14                                  8_789.59     1_055.39     9_844.98       0.2859          1.0625         3.96
IVFOPQ-m64-nl223-np14                                 14_789.61     1_686.94    16_476.56       0.4139          1.0376         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_185.94     1_139.87    11_325.81       0.2859          1.0625         6.45
IVFOPQ-m32-nl316-np1                                   9_218.18       430.94     9_649.12       0.2448          1.0719         4.65
IVFOPQ-m64-nl316-np1                                  15_111.27       478.61    15_589.87       0.3144          1.0528         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_382.24       449.75    10_831.99       0.2550          1.0716         7.13
IVFOPQ-m32-nl316-np2                                   9_218.18       479.74     9_697.92       0.2717          1.0654         4.65
IVFOPQ-m64-nl316-np2                                  15_111.27       558.84    15_670.11       0.3706          1.0435         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_382.24       491.15    10_873.39       0.2694          1.0677         7.13
IVFOPQ-m32-nl316-np4                                   9_218.18       569.99     9_788.17       0.2855          1.0624         4.65
IVFOPQ-m64-nl316-np4                                  15_111.27       742.17    15_853.43       0.4041          1.0388         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_382.24       597.65    10_979.90       0.2773          1.0657         7.13
IVFOPQ-m32-nl316-np8                                   9_218.18       759.85     9_978.03       0.2889          1.0617         4.65
IVFOPQ-m64-nl316-np8                                  15_111.27     1_092.19    16_203.46       0.4154          1.0372         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_382.24       794.36    11_176.60       0.2863          1.0628         7.13
IVFOPQ-m32-nl316-np15                                  9_218.18     1_083.36    10_301.54       0.2891          1.0616         4.65
IVFOPQ-m64-nl316-np15                                 15_111.27     1_725.46    16_836.72       0.4166          1.0371         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_382.24     1_135.29    11_517.53       0.2891          1.0616         7.13
IVFOPQ-m32-nl316-np17                                  9_218.18     1_182.95    10_401.13       0.2892          1.0616         4.65
IVFOPQ-m64-nl316-np17                                 15_111.27     1_900.74    17_012.00       0.4166          1.0371         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_382.24     1_241.14    11_623.38       0.2891          1.0616         7.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-OPQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-OPQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-OPQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-OPQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.66     1_186.69     1_254.35       1.0000          1.0000        97.66
SOAROPQ-near-np1                                      11_025.43       439.38    11_464.81       0.2535          1.0746         6.32
SOAROPQ-near-np2                                      11_025.43       498.64    11_524.07       0.2680          1.0690         6.32
SOAROPQ-near-np4                                      11_025.43       613.92    11_639.35       0.2783          1.0653         6.32
SOAROPQ-near-np7                                      11_025.43       789.55    11_814.98       0.2818          1.0640         6.32
SOAROPQ-near-np8                                      11_025.43       832.49    11_857.92       0.2820          1.0640         6.32
SOAROPQ-near-np12                                     11_025.43     1_062.59    12_088.02       0.2820          1.0639         6.32
SOAROPQ-shift0.3-np1                                  10_944.86       436.54    11_381.40       0.2535          1.0745         6.32
SOAROPQ-shift0.3-np2                                  10_944.86       496.90    11_441.75       0.2678          1.0690         6.32
SOAROPQ-shift0.3-np4                                  10_944.86       613.39    11_558.24       0.2777          1.0654         6.32
SOAROPQ-shift0.3-np7                                  10_944.86       784.44    11_729.30       0.2817          1.0641         6.32
SOAROPQ-shift0.3-np8                                  10_944.86       827.98    11_772.84       0.2819          1.0640         6.32
SOAROPQ-shift0.3-np12                                 10_944.86     1_051.97    11_996.82       0.2820          1.0639         6.32
SOAROPQ-shift0.7-np1                                  10_899.59       438.44    11_338.03       0.2529          1.0747         6.32
SOAROPQ-shift0.7-np2                                  10_899.59       513.67    11_413.27       0.2670          1.0693         6.32
SOAROPQ-shift0.7-np4                                  10_899.59       613.82    11_513.41       0.2772          1.0656         6.32
SOAROPQ-shift0.7-np7                                  10_899.59       774.65    11_674.25       0.2816          1.0641         6.32
SOAROPQ-shift0.7-np8                                  10_899.59       829.03    11_728.63       0.2819          1.0640         6.32
SOAROPQ-shift0.7-np12                                 10_899.59     1_061.14    11_960.73       0.2820          1.0639         6.32
SOAROPQ-orth1-np1                                     10_949.11       436.24    11_385.35       0.2527          1.0750         6.32
SOAROPQ-orth1-np2                                     10_949.11       506.73    11_455.84       0.2665          1.0696         6.32
SOAROPQ-orth1-np4                                     10_949.11       614.60    11_563.70       0.2768          1.0658         6.32
SOAROPQ-orth1-np7                                     10_949.11       772.26    11_721.37       0.2815          1.0642         6.32
SOAROPQ-orth1-np8                                     10_949.11       845.51    11_794.62       0.2818          1.0641         6.32
SOAROPQ-orth1-np12                                    10_949.11     1_050.86    11_999.97       0.2820          1.0639         6.32
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.56     1_266.10     1_333.66       1.0000          1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_752.05       455.32     8_207.37       0.6620          1.0298         3.49
IVFOPQ-m64-nl111-np1                                  13_666.15       551.30    14_217.46       0.7619          1.0160         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_943.00       460.09     9_403.08       0.6625          1.0333         5.98
IVFOPQ-m32-nl111-np2                                   7_752.05       510.48     8_262.53       0.6716          1.0264         3.49
IVFOPQ-m64-nl111-np2                                  13_666.15       636.39    14_302.55       0.7737          1.0122         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_943.00       514.19     9_457.18       0.6719          1.0265         5.98
IVFOPQ-m32-nl111-np4                                   7_752.05       586.07     8_338.12       0.6726          1.0261         3.49
IVFOPQ-m64-nl111-np4                                  13_666.15       802.80    14_468.95       0.7749          1.0119         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_943.00       634.36     9_577.35       0.6726          1.0261         5.98
IVFOPQ-m32-nl111-np5                                   7_752.05       640.69     8_392.74       0.6726          1.0261         3.49
IVFOPQ-m64-nl111-np5                                  13_666.15       887.97    14_554.13       0.7749          1.0119         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_943.00       656.08     9_599.08       0.6726          1.0261         5.98
IVFOPQ-m32-nl111-np8                                   7_752.05       763.01     8_515.06       0.6726          1.0261         3.49
IVFOPQ-m64-nl111-np8                                  13_666.15     1_154.61    14_820.76       0.7749          1.0119         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_943.00       815.68     9_758.67       0.6726          1.0261         5.98
IVFOPQ-m32-nl111-np10                                  7_752.05       969.78     8_721.83       0.6726          1.0261         3.49
IVFOPQ-m64-nl111-np10                                 13_666.15     1_469.85    15_136.01       0.7749          1.0119         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_943.00       987.27     9_930.26       0.6726          1.0261         5.98
IVFOPQ-m32-nl158-np1                                   9_924.75       465.61    10_390.35       0.6657          1.0293         3.84
IVFOPQ-m64-nl158-np1                                  15_506.48       549.57    16_056.06       0.7625          1.0163         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_743.26       461.67    11_204.93       0.6749          1.0267         6.32
IVFOPQ-m32-nl158-np2                                   9_924.75       533.07    10_457.81       0.6789          1.0249         3.84
IVFOPQ-m64-nl158-np2                                  15_506.48       633.84    16_140.33       0.7778          1.0116         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_743.26       512.30    11_255.56       0.6796          1.0248         6.32
IVFOPQ-m32-nl158-np4                                   9_924.75       593.14    10_517.88       0.6803          1.0245         3.84
IVFOPQ-m64-nl158-np4                                  15_506.48       841.60    16_348.08       0.7796          1.0112         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_743.26       603.04    11_346.30       0.6802          1.0245         6.32
IVFOPQ-m32-nl158-np7                                   9_924.75       749.01    10_673.76       0.6803          1.0245         3.84
IVFOPQ-m64-nl158-np7                                  15_506.48     1_046.38    16_552.86       0.7796          1.0112         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_743.26       748.22    11_491.48       0.6803          1.0245         6.32
IVFOPQ-m32-nl158-np8                                   9_924.75       762.66    10_687.41       0.6803          1.0245         3.84
IVFOPQ-m64-nl158-np8                                  15_506.48     1_123.84    16_630.32       0.7796          1.0112         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_743.26       786.40    11_529.66       0.6803          1.0245         6.32
IVFOPQ-m32-nl158-np12                                  9_924.75       934.38    10_859.12       0.6803          1.0245         3.84
IVFOPQ-m64-nl158-np12                                 15_506.48     1_457.06    16_963.54       0.7796          1.0112         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_743.26       982.87    11_726.12       0.6803          1.0245         6.32
IVFOPQ-m32-nl223-np1                                   8_910.15       439.62     9_349.77       0.4489          1.0729         3.96
IVFOPQ-m64-nl223-np1                                  14_959.59       481.98    15_441.57       0.4764          1.0623         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_159.35       447.61    10_606.96       0.5754          1.0451         6.45
IVFOPQ-m32-nl223-np2                                   8_910.15       510.49     9_420.63       0.5858          1.0411         3.96
IVFOPQ-m64-nl223-np2                                  14_959.59       582.80    15_542.38       0.6476          1.0297         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_159.35       519.91    10_679.26       0.6587          1.0284         6.45
IVFOPQ-m32-nl223-np4                                   8_910.15       588.28     9_498.43       0.6733          1.0257         3.96
IVFOPQ-m64-nl223-np4                                  14_959.59       864.59    15_824.17       0.7622          1.0137         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_159.35       609.02    10_768.36       0.6873          1.0233         6.45
IVFOPQ-m32-nl223-np8                                   8_910.15       769.62     9_679.76       0.6902          1.0229         3.96
IVFOPQ-m64-nl223-np8                                  14_959.59     1_122.55    16_082.14       0.7839          1.0107         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_159.35       794.37    10_953.72       0.6904          1.0228         6.45
IVFOPQ-m32-nl223-np11                                  8_910.15       907.27     9_817.41       0.6905          1.0228         3.96
IVFOPQ-m64-nl223-np11                                 14_959.59     1_386.53    16_346.11       0.7842          1.0107         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_159.35       945.49    11_104.84       0.6905          1.0228         6.45
IVFOPQ-m32-nl223-np14                                  8_910.15     1_049.23     9_959.38       0.6905          1.0228         3.96
IVFOPQ-m64-nl223-np14                                 14_959.59     1_644.82    16_604.40       0.7842          1.0107         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_159.35     1_076.70    11_236.05       0.6905          1.0228         6.45
IVFOPQ-m32-nl316-np1                                   9_354.46       439.74     9_794.21       0.3855          1.0892         4.65
IVFOPQ-m64-nl316-np1                                  15_666.15       473.49    16_139.64       0.4006          1.0791         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_649.20       440.43    11_089.62       0.5128          1.0563         7.13
IVFOPQ-m32-nl316-np2                                   9_354.46       478.28     9_832.74       0.5231          1.0527         4.65
IVFOPQ-m64-nl316-np2                                  15_666.15       571.18    16_237.33       0.5655          1.0417         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_649.20       499.97    11_149.17       0.6174          1.0354         7.13
IVFOPQ-m32-nl316-np4                                   9_354.46       577.11     9_931.57       0.6320          1.0326         4.65
IVFOPQ-m64-nl316-np4                                  15_666.15       755.80    16_421.95       0.7063          1.0211         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_649.20       607.89    11_257.08       0.6762          1.0252         7.13
IVFOPQ-m32-nl316-np8                                   9_354.46       767.23    10_121.69       0.6892          1.0230         4.65
IVFOPQ-m64-nl316-np8                                  15_666.15     1_109.62    16_775.77       0.7814          1.0111         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_649.20       806.82    11_456.01       0.6927          1.0224         7.13
IVFOPQ-m32-nl316-np15                                  9_354.46     1_082.35    10_436.81       0.6934          1.0223         4.65
IVFOPQ-m64-nl316-np15                                 15_666.15     1_703.20    17_369.35       0.7867          1.0103         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_649.20     1_127.01    11_776.20       0.6934          1.0223         7.13
IVFOPQ-m32-nl316-np17                                  9_354.46     1_176.63    10_531.09       0.6934          1.0223         4.65
IVFOPQ-m64-nl316-np17                                 15_666.15     1_870.34    17_536.49       0.7867          1.0103         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_649.20     1_226.25    11_875.45       0.6934          1.0223         7.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-OPQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-OPQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-OPQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-OPQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.56     1_266.10     1_333.66       1.0000          1.0000        97.66
SOAROPQ-near-np1                                      11_245.03       471.92    11_716.95       0.6749          1.0266         6.32
SOAROPQ-near-np2                                      11_245.03       511.84    11_756.87       0.6796          1.0247         6.32
SOAROPQ-near-np4                                      11_245.03       607.11    11_852.14       0.6802          1.0245         6.32
SOAROPQ-near-np7                                      11_245.03       754.34    11_999.37       0.6803          1.0245         6.32
SOAROPQ-near-np8                                      11_245.03       786.40    12_031.43       0.6803          1.0245         6.32
SOAROPQ-near-np12                                     11_245.03       986.07    12_231.10       0.6803          1.0245         6.32
SOAROPQ-shift0.3-np1                                  10_628.16       471.74    11_099.91       0.6749          1.0267         6.32
SOAROPQ-shift0.3-np2                                  10_628.16       518.68    11_146.84       0.6796          1.0247         6.32
SOAROPQ-shift0.3-np4                                  10_628.16       604.49    11_232.66       0.6802          1.0245         6.32
SOAROPQ-shift0.3-np7                                  10_628.16       736.84    11_365.00       0.6803          1.0245         6.32
SOAROPQ-shift0.3-np8                                  10_628.16       789.65    11_417.81       0.6803          1.0245         6.32
SOAROPQ-shift0.3-np12                                 10_628.16       979.77    11_607.93       0.6803          1.0245         6.32
SOAROPQ-shift0.7-np1                                  10_643.69       473.37    11_117.06       0.6748          1.0267         6.32
SOAROPQ-shift0.7-np2                                  10_643.69       510.49    11_154.18       0.6795          1.0248         6.32
SOAROPQ-shift0.7-np4                                  10_643.69       604.01    11_247.70       0.6802          1.0246         6.32
SOAROPQ-shift0.7-np7                                  10_643.69       737.12    11_380.81       0.6803          1.0245         6.32
SOAROPQ-shift0.7-np8                                  10_643.69       788.83    11_432.52       0.6803          1.0245         6.32
SOAROPQ-shift0.7-np12                                 10_643.69       980.78    11_624.47       0.6803          1.0245         6.32
SOAROPQ-orth1-np1                                     10_648.21       467.13    11_115.34       0.6745          1.0268         6.32
SOAROPQ-orth1-np2                                     10_648.21       512.98    11_161.19       0.6794          1.0248         6.32
SOAROPQ-orth1-np4                                     10_648.21       600.61    11_248.82       0.6801          1.0246         6.32
SOAROPQ-orth1-np7                                     10_648.21       744.28    11_392.49       0.6803          1.0245         6.32
SOAROPQ-orth1-np8                                     10_648.21       803.03    11_451.24       0.6803          1.0245         6.32
SOAROPQ-orth1-np12                                    10_648.21     1_017.08    11_665.29       0.6803          1.0245         6.32
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.53     1_290.92     1_360.46       1.0000          1.0000        97.66
IVFOPQ-m32-nl111-np1                                   8_074.99       432.28     8_507.27       0.7216          1.1712         3.49
IVFOPQ-m64-nl111-np1                                  14_082.39       489.13    14_571.52       0.7255          1.1681         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_288.69       450.70     9_739.39       0.8436          1.0561         5.98
IVFOPQ-m32-nl111-np2                                   8_074.99       496.07     8_571.05       0.8515          1.0485         3.49
IVFOPQ-m64-nl111-np2                                  14_082.39       612.32    14_694.71       0.8597          1.0453         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_288.69       544.60     9_833.29       0.8854          1.0261         5.98
IVFOPQ-m32-nl111-np4                                   8_074.99       611.29     8_686.28       0.8900          1.0220         3.49
IVFOPQ-m64-nl111-np4                                  14_082.39       856.80    14_939.19       0.9006          1.0185         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_288.69       702.95     9_991.64       0.8923          1.0213         5.98
IVFOPQ-m32-nl111-np5                                   8_074.99       668.39     8_743.38       0.8919          1.0209         3.49
IVFOPQ-m64-nl111-np5                                  14_082.39       981.48    15_063.87       0.9028          1.0174         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_288.69       773.60    10_062.29       0.8927          1.0209         5.98
IVFOPQ-m32-nl111-np8                                   8_074.99       862.10     8_937.09       0.8932          1.0202         3.49
IVFOPQ-m64-nl111-np8                                  14_082.39     1_355.02    15_437.41       0.9042          1.0167         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_288.69       984.86    10_273.55       0.8931          1.0205         5.98
IVFOPQ-m32-nl111-np10                                  8_074.99       980.22     9_055.21       0.8933          1.0202         3.49
IVFOPQ-m64-nl111-np10                                 14_082.39     1_591.59    15_673.98       0.9042          1.0167         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_288.69     1_119.13    10_407.82       0.8932          1.0203         5.98
IVFOPQ-m32-nl158-np1                                   9_965.75       425.74    10_391.50       0.7234          1.1670         3.84
IVFOPQ-m64-nl158-np1                                  15_788.38       476.02    16_264.40       0.7265          1.1644         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_999.34       439.94    11_439.28       0.8467          1.0546         6.32
IVFOPQ-m32-nl158-np2                                   9_965.75       476.37    10_442.12       0.8571          1.0452         3.84
IVFOPQ-m64-nl158-np2                                  15_788.38       583.49    16_371.88       0.8632          1.0426         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_999.34       519.31    11_518.64       0.8916          1.0239         6.32
IVFOPQ-m32-nl158-np4                                   9_965.75       586.69    10_552.44       0.8972          1.0195         3.84
IVFOPQ-m64-nl158-np4                                  15_788.38       801.37    16_589.75       0.9048          1.0170         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_999.34       649.94    11_649.28       0.9004          1.0186         6.32
IVFOPQ-m32-nl158-np7                                   9_965.75       754.40    10_720.15       0.9020          1.0171         3.84
IVFOPQ-m64-nl158-np7                                  15_788.38     1_128.48    16_916.86       0.9096          1.0145         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_999.34       847.92    11_847.25       0.9021          1.0173         6.32
IVFOPQ-m32-nl158-np8                                   9_965.75       807.95    10_773.70       0.9024          1.0169         3.84
IVFOPQ-m64-nl158-np8                                  15_788.38     1_237.54    17_025.92       0.9100          1.0144         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_999.34       916.35    11_915.69       0.9023          1.0172         6.32
IVFOPQ-m32-nl158-np12                                  9_965.75     1_029.27    10_995.02       0.9026          1.0169         3.84
IVFOPQ-m64-nl158-np12                                 15_788.38     1_663.58    17_451.97       0.9102          1.0143         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_999.34     1_172.44    12_171.77       0.9025          1.0169         6.32
IVFOPQ-m32-nl223-np1                                   8_633.41       427.01     9_060.42       0.6955          1.1880         3.96
IVFOPQ-m64-nl223-np1                                  14_651.82       472.66    15_124.48       0.6983          1.1856         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_103.37       438.28    10_541.65       0.8331          1.0653         6.45
IVFOPQ-m32-nl223-np2                                   8_633.41       472.39     9_105.79       0.8441          1.0543         3.96
IVFOPQ-m64-nl223-np2                                  14_651.82       574.03    15_225.85       0.8506          1.0516         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_103.37       494.55    10_597.92       0.8917          1.0253         6.45
IVFOPQ-m32-nl223-np4                                   8_633.41       572.97     9_206.38       0.8994          1.0194         3.96
IVFOPQ-m64-nl223-np4                                  14_651.82       760.96    15_412.79       0.9080          1.0165         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_103.37       616.24    10_719.61       0.9047          1.0178         6.45
IVFOPQ-m32-nl223-np8                                   8_633.41       782.89     9_416.29       0.9071          1.0156         3.96
IVFOPQ-m64-nl223-np8                                  14_651.82     1_149.92    15_801.75       0.9164          1.0125         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_103.37       845.49    10_948.86       0.9071          1.0159         6.45
IVFOPQ-m32-nl223-np11                                  8_633.41       926.54     9_559.94       0.9075          1.0154         3.96
IVFOPQ-m64-nl223-np11                                 14_651.82     1_450.69    16_102.51       0.9168          1.0124         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_103.37     1_085.22    11_188.59       0.9075          1.0156         6.45
IVFOPQ-m32-nl223-np14                                  8_633.41     1_075.24     9_708.65       0.9075          1.0154         3.96
IVFOPQ-m64-nl223-np14                                 14_651.82     1_737.99    16_389.81       0.9169          1.0123         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_103.37     1_264.12    11_367.50       0.9075          1.0154         6.45
IVFOPQ-m32-nl316-np1                                   8_889.49       433.75     9_323.24       0.6783          1.2033         4.65
IVFOPQ-m64-nl316-np1                                  14_957.78       470.27    15_428.05       0.6798          1.2013         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_012.77       434.04    10_446.81       0.8246          1.0715         7.13
IVFOPQ-m32-nl316-np2                                   8_889.49       473.39     9_362.88       0.8377          1.0586         4.65
IVFOPQ-m64-nl316-np2                                  14_957.78       560.25    15_518.03       0.8418          1.0567         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_012.77       489.86    10_502.63       0.8939          1.0255         7.13
IVFOPQ-m32-nl316-np4                                   8_889.49       578.85     9_468.33       0.9032          1.0184         4.65
IVFOPQ-m64-nl316-np4                                  14_957.78       738.88    15_696.66       0.9089          1.0165         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_012.77       609.93    10_622.70       0.9102          1.0163         7.13
IVFOPQ-m32-nl316-np8                                   8_889.49       758.32     9_647.81       0.9137          1.0133         4.65
IVFOPQ-m64-nl316-np8                                  14_957.78     1_101.69    16_059.47       0.9199          1.0113         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_012.77       817.71    10_830.48       0.9136          1.0139         7.13
IVFOPQ-m32-nl316-np15                                  8_889.49     1_102.81     9_992.30       0.9145          1.0130         4.65
IVFOPQ-m64-nl316-np15                                 14_957.78     1_762.21    16_720.00       0.9208          1.0110         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_012.77     1_197.11    11_209.88       0.9144          1.0131         7.13
IVFOPQ-m32-nl316-np17                                  8_889.49     1_204.60    10_094.09       0.9145          1.0130         4.65
IVFOPQ-m64-nl316-np17                                 14_957.78     1_950.93    16_908.72       0.9208          1.0110         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_012.77     1_314.71    11_327.48       0.9145          1.0130         7.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-OPQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-OPQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-OPQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-OPQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.53     1_290.92     1_360.46       1.0000          1.0000        97.66
SOAROPQ-near-np1                                      11_038.97       464.27    11_503.25       0.8452          1.0573         6.32
SOAROPQ-near-np2                                      11_038.97       522.04    11_561.02       0.8917          1.0237         6.32
SOAROPQ-near-np4                                      11_038.97       661.69    11_700.66       0.9010          1.0181         6.32
SOAROPQ-near-np7                                      11_038.97       847.42    11_886.39       0.9023          1.0171         6.32
SOAROPQ-near-np8                                      11_038.97       910.82    11_949.79       0.9024          1.0170         6.32
SOAROPQ-near-np12                                     11_038.97     1_158.56    12_197.53       0.9026          1.0169         6.32
SOAROPQ-shift0.3-np1                                  11_148.32       447.06    11_595.37       0.8485          1.0533         6.32
SOAROPQ-shift0.3-np2                                  11_148.32       513.15    11_661.47       0.8922          1.0234         6.32
SOAROPQ-shift0.3-np4                                  11_148.32       655.04    11_803.35       0.9007          1.0183         6.32
SOAROPQ-shift0.3-np7                                  11_148.32       870.69    12_019.01       0.9022          1.0172         6.32
SOAROPQ-shift0.3-np8                                  11_148.32       914.99    12_063.31       0.9024          1.0171         6.32
SOAROPQ-shift0.3-np12                                 11_148.32     1_158.69    12_307.01       0.9025          1.0169         6.32
SOAROPQ-shift0.7-np1                                  10_817.04       442.95    11_259.99       0.8435          1.0574         6.32
SOAROPQ-shift0.7-np2                                  10_817.04       514.27    11_331.32       0.8909          1.0247         6.32
SOAROPQ-shift0.7-np4                                  10_817.04       648.22    11_465.26       0.9001          1.0189         6.32
SOAROPQ-shift0.7-np7                                  10_817.04       853.93    11_670.97       0.9021          1.0174         6.32
SOAROPQ-shift0.7-np8                                  10_817.04       915.31    11_732.35       0.9023          1.0172         6.32
SOAROPQ-shift0.7-np12                                 10_817.04     1_173.81    11_990.85       0.9025          1.0169         6.32
SOAROPQ-orth1-np1                                     11_057.27       444.31    11_501.58       0.8460          1.0560         6.32
SOAROPQ-orth1-np2                                     11_057.27       513.51    11_570.78       0.8917          1.0239         6.32
SOAROPQ-orth1-np4                                     11_057.27       693.79    11_751.06       0.9006          1.0184         6.32
SOAROPQ-orth1-np7                                     11_057.27       884.52    11_941.79       0.9022          1.0172         6.32
SOAROPQ-orth1-np8                                     11_057.27       909.91    11_967.19       0.9024          1.0171         6.32
SOAROPQ-orth1-np12                                    11_057.27     1_155.55    12_212.82       0.9025          1.0169         6.32
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        74.52     1_290.24     1_364.76       1.0000          1.0000        97.85
IVFOPQ-m32-nl111-np1                                   7_885.34       428.43     8_313.77       0.7779          1.1326         3.49
IVFOPQ-m64-nl111-np1                                  13_867.76       479.86    14_347.62       0.7817          1.1296         5.02
SOAROPQ-orth1-m32-nl111-np1                            9_115.56       444.04     9_559.61       0.8616          1.0578         5.98
IVFOPQ-m32-nl111-np2                                   7_885.34       480.10     8_365.44       0.8790          1.0367         3.49
IVFOPQ-m64-nl111-np2                                  13_867.76       601.97    14_469.73       0.8858          1.0334         5.02
SOAROPQ-orth1-m32-nl111-np2                            9_115.56       523.13     9_638.69       0.8926          1.0300         5.98
IVFOPQ-m32-nl111-np4                                   7_885.34       597.68     8_483.02       0.8990          1.0225         3.49
IVFOPQ-m64-nl111-np4                                  13_867.76       824.45    14_692.21       0.9071          1.0192         5.02
SOAROPQ-orth1-m32-nl111-np4                            9_115.56       675.26     9_790.83       0.8988          1.0242         5.98
IVFOPQ-m32-nl111-np5                                   7_885.34       663.35     8_548.68       0.9000          1.0220         3.49
IVFOPQ-m64-nl111-np5                                  13_867.76       946.54    14_814.30       0.9082          1.0187         5.02
SOAROPQ-orth1-m32-nl111-np5                            9_115.56       745.40     9_860.96       0.8995          1.0234         5.98
IVFOPQ-m32-nl111-np8                                   7_885.34       832.97     8_718.31       0.9005          1.0218         3.49
IVFOPQ-m64-nl111-np8                                  13_867.76     1_301.69    15_169.45       0.9088          1.0185         5.02
SOAROPQ-orth1-m32-nl111-np8                            9_115.56       946.23    10_061.79       0.9003          1.0222         5.98
IVFOPQ-m32-nl111-np10                                  7_885.34       946.72     8_832.06       0.9005          1.0217         3.49
IVFOPQ-m64-nl111-np10                                 13_867.76     1_613.95    15_481.71       0.9088          1.0184         5.02
SOAROPQ-orth1-m32-nl111-np10                           9_115.56     1_062.18    10_177.75       0.9005          1.0220         5.98
IVFOPQ-m32-nl158-np1                                   9_599.82       452.59    10_052.41       0.7594          1.1510         3.84
IVFOPQ-m64-nl158-np1                                  15_697.88       469.24    16_167.13       0.7621          1.1486         5.36
SOAROPQ-orth1-m32-nl158-np1                           10_853.71       436.87    11_290.58       0.8568          1.0634         6.32
IVFOPQ-m32-nl158-np2                                   9_599.82       470.19    10_070.01       0.8773          1.0396         3.84
IVFOPQ-m64-nl158-np2                                  15_697.88       573.44    16_271.32       0.8816          1.0375         5.36
SOAROPQ-orth1-m32-nl158-np2                           10_853.71       501.35    11_355.05       0.8979          1.0286         6.32
IVFOPQ-m32-nl158-np4                                   9_599.82       575.91    10_175.73       0.9070          1.0189         3.84
IVFOPQ-m64-nl158-np4                                  15_697.88       774.23    16_472.11       0.9125          1.0166         5.36
SOAROPQ-orth1-m32-nl158-np4                           10_853.71       627.37    11_481.08       0.9073          1.0207         6.32
IVFOPQ-m32-nl158-np7                                   9_599.82       738.65    10_338.47       0.9093          1.0176         3.84
IVFOPQ-m64-nl158-np7                                  15_697.88     1_087.50    16_785.38       0.9151          1.0153         5.36
SOAROPQ-orth1-m32-nl158-np7                           10_853.71       824.05    11_677.76       0.9090          1.0184         6.32
IVFOPQ-m32-nl158-np8                                   9_599.82       792.04    10_391.86       0.9094          1.0176         3.84
IVFOPQ-m64-nl158-np8                                  15_697.88     1_192.58    16_890.46       0.9153          1.0152         5.36
SOAROPQ-orth1-m32-nl158-np8                           10_853.71       881.44    11_735.15       0.9092          1.0182         6.32
IVFOPQ-m32-nl158-np12                                  9_599.82     1_011.69    10_611.51       0.9095          1.0176         3.84
IVFOPQ-m64-nl158-np12                                 15_697.88     1_610.45    17_308.33       0.9154          1.0152         5.36
SOAROPQ-orth1-m32-nl158-np12                          10_853.71     1_126.83    11_980.54       0.9094          1.0177         6.32
IVFOPQ-m32-nl223-np1                                   8_692.25       419.27     9_111.52       0.7330          1.1767         3.96
IVFOPQ-m64-nl223-np1                                  14_775.05       468.68    15_243.72       0.7347          1.1749         5.49
SOAROPQ-orth1-m32-nl223-np1                            9_907.59       433.42    10_341.01       0.8494          1.0687         6.45
IVFOPQ-m32-nl223-np2                                   8_692.25       471.74     9_163.99       0.8709          1.0451         3.96
IVFOPQ-m64-nl223-np2                                  14_775.05       560.62    15_335.67       0.8741          1.0434         5.49
SOAROPQ-orth1-m32-nl223-np2                            9_907.59       516.37    10_423.96       0.9000          1.0280         6.45
IVFOPQ-m32-nl223-np4                                   8_692.25       569.71     9_261.96       0.9099          1.0179         3.96
IVFOPQ-m64-nl223-np4                                  14_775.05       747.99    15_523.03       0.9149          1.0162         5.49
SOAROPQ-orth1-m32-nl223-np4                            9_907.59       601.14    10_508.73       0.9108          1.0198         6.45
IVFOPQ-m32-nl223-np8                                   8_692.25       762.68     9_454.93       0.9134          1.0162         3.96
IVFOPQ-m64-nl223-np8                                  14_775.05     1_121.13    15_896.18       0.9185          1.0144         5.49
SOAROPQ-orth1-m32-nl223-np8                            9_907.59       829.57    10_737.15       0.9132          1.0168         6.45
IVFOPQ-m32-nl223-np11                                  8_692.25       909.43     9_601.68       0.9136          1.0161         3.96
IVFOPQ-m64-nl223-np11                                 14_775.05     1_403.03    16_178.08       0.9187          1.0144         5.49
SOAROPQ-orth1-m32-nl223-np11                           9_907.59       998.22    10_905.81       0.9135          1.0163         6.45
IVFOPQ-m32-nl223-np14                                  8_692.25     1_073.78     9_766.03       0.9136          1.0161         3.96
IVFOPQ-m64-nl223-np14                                 14_775.05     1_690.06    16_465.11       0.9188          1.0143         5.49
SOAROPQ-orth1-m32-nl223-np14                           9_907.59     1_163.01    11_070.60       0.9135          1.0162         6.45
IVFOPQ-m32-nl316-np1                                   8_988.14       429.77     9_417.91       0.7080          1.2039         4.65
IVFOPQ-m64-nl316-np1                                  15_190.98       469.46    15_660.44       0.7097          1.2018         6.17
SOAROPQ-orth1-m32-nl316-np1                           10_348.62       439.77    10_788.39       0.8347          1.0799         7.13
IVFOPQ-m32-nl316-np2                                   8_988.14       472.17     9_460.31       0.8590          1.0536         4.65
IVFOPQ-m64-nl316-np2                                  15_190.98       559.44    15_750.42       0.8632          1.0515         6.17
SOAROPQ-orth1-m32-nl316-np2                           10_348.62       487.05    10_835.67       0.8984          1.0293         7.13
IVFOPQ-m32-nl316-np4                                   8_988.14       566.95     9_555.09       0.9114          1.0177         4.65
IVFOPQ-m64-nl316-np4                                  15_190.98       738.65    15_929.63       0.9173          1.0157         6.17
SOAROPQ-orth1-m32-nl316-np4                           10_348.62       608.43    10_957.05       0.9135          1.0191         7.13
IVFOPQ-m32-nl316-np8                                   8_988.14       754.31     9_742.46       0.9170          1.0148         4.65
IVFOPQ-m64-nl316-np8                                  15_190.98     1_093.80    16_284.79       0.9234          1.0128         6.17
SOAROPQ-orth1-m32-nl316-np8                           10_348.62       798.21    11_146.83       0.9166          1.0159         7.13
IVFOPQ-m32-nl316-np15                                  8_988.14     1_088.42    10_076.57       0.9174          1.0147         4.65
IVFOPQ-m64-nl316-np15                                 15_190.98     1_729.82    16_920.80       0.9239          1.0127         6.17
SOAROPQ-orth1-m32-nl316-np15                          10_348.62     1_191.09    11_539.71       0.9173          1.0148         7.13
IVFOPQ-m32-nl316-np17                                  8_988.14     1_190.18    10_178.32       0.9174          1.0147         4.65
IVFOPQ-m64-nl316-np17                                 15_190.98     1_915.57    17_106.55       0.9239          1.0127         6.17
SOAROPQ-orth1-m32-nl316-np17                          10_348.62     1_286.02    11_634.64       0.9174          1.0147         7.13
-----------------------------------------------------------------------------------------------------------------------------------

-----------------------------
Sweep B: rule comparison at nlist=158
-----------------------------
Building SOAR-OPQ (rule=near, nlist=158)...
  Querying rule=near, nprobe=1...
  Querying rule=near, nprobe=2...
  Querying rule=near, nprobe=4...
  Querying rule=near, nprobe=7...
  Querying rule=near, nprobe=8...
  Querying rule=near, nprobe=12...
Building SOAR-OPQ (rule=shift0.3, nlist=158)...
  Querying rule=shift0.3, nprobe=1...
  Querying rule=shift0.3, nprobe=2...
  Querying rule=shift0.3, nprobe=4...
  Querying rule=shift0.3, nprobe=7...
  Querying rule=shift0.3, nprobe=8...
  Querying rule=shift0.3, nprobe=12...
Building SOAR-OPQ (rule=shift0.7, nlist=158)...
  Querying rule=shift0.7, nprobe=1...
  Querying rule=shift0.7, nprobe=2...
  Querying rule=shift0.7, nprobe=4...
  Querying rule=shift0.7, nprobe=7...
  Querying rule=shift0.7, nprobe=8...
  Querying rule=shift0.7, nprobe=12...
Building SOAR-OPQ (rule=orth1, nlist=158)...
  Querying rule=orth1, nprobe=1...
  Querying rule=orth1, nprobe=2...
  Querying rule=orth1, nprobe=4...
  Querying rule=orth1, nprobe=7...
  Querying rule=orth1, nprobe=8...
  Querying rule=orth1, nprobe=12...

===================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        74.52     1_290.24     1_364.76       1.0000          1.0000        97.85
SOAROPQ-near-np1                                      10_846.51       441.02    11_287.53       0.8631          1.0549         6.32
SOAROPQ-near-np2                                      10_846.51       501.35    11_347.86       0.8998          1.0253         6.32
SOAROPQ-near-np4                                      10_846.51       628.87    11_475.38       0.9078          1.0194         6.32
SOAROPQ-near-np7                                      10_846.51       816.15    11_662.66       0.9092          1.0179         6.32
SOAROPQ-near-np8                                      10_846.51       883.29    11_729.80       0.9093          1.0178         6.32
SOAROPQ-near-np12                                     10_846.51     1_114.42    11_960.93       0.9095          1.0176         6.32
SOAROPQ-shift0.3-np1                                  10_879.37       442.62    11_321.99       0.8624          1.0567         6.32
SOAROPQ-shift0.3-np2                                  10_879.37       509.13    11_388.50       0.8984          1.0272         6.32
SOAROPQ-shift0.3-np4                                  10_879.37       628.91    11_508.27       0.9069          1.0204         6.32
SOAROPQ-shift0.3-np7                                  10_879.37       869.43    11_748.79       0.9089          1.0183         6.32
SOAROPQ-shift0.3-np8                                  10_879.37       888.66    11_768.03       0.9091          1.0181         6.32
SOAROPQ-shift0.3-np12                                 10_879.37     1_141.81    12_021.18       0.9094          1.0177         6.32
SOAROPQ-shift0.7-np1                                  10_766.78       447.42    11_214.20       0.8569          1.0638         6.32
SOAROPQ-shift0.7-np2                                  10_766.78       499.72    11_266.50       0.8955          1.0311         6.32
SOAROPQ-shift0.7-np4                                  10_766.78       629.60    11_396.38       0.9057          1.0223         6.32
SOAROPQ-shift0.7-np7                                  10_766.78       818.71    11_585.49       0.9085          1.0190         6.32
SOAROPQ-shift0.7-np8                                  10_766.78       877.22    11_644.00       0.9088          1.0187         6.32
SOAROPQ-shift0.7-np12                                 10_766.78     1_117.54    11_884.32       0.9094          1.0178         6.32
SOAROPQ-orth1-np1                                     10_892.51       439.73    11_332.24       0.8568          1.0634         6.32
SOAROPQ-orth1-np2                                     10_892.51       512.54    11_405.05       0.8979          1.0286         6.32
SOAROPQ-orth1-np4                                     10_892.51       628.74    11_521.25       0.9073          1.0207         6.32
SOAROPQ-orth1-np7                                     10_892.51       815.31    11_707.82       0.9090          1.0184         6.32
SOAROPQ-orth1-np8                                     10_892.51       876.16    11_768.67       0.9092          1.0182         6.32
SOAROPQ-orth1-np12                                    10_892.51     1_115.44    12_007.95       0.9094          1.0177         6.32
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
