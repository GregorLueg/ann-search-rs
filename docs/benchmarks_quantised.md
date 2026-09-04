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
Exhaustive (query)                                        11.28       634.34       645.62       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.28     6_418.01     6_429.29       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.27     1_244.96     1_257.24       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    12.27    12_462.48    12_474.75       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              319.54        89.46       409.00       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              319.54       100.62       420.17       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              319.54       137.07       456.61       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    319.54     1_410.31     1_729.85       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              585.35        93.37       678.71       0.9820          1.0001            1.0000         9.21
IVF-BF16-nl387-np27 (query)                              585.35       118.97       704.32       0.9828          1.0001            1.0000         9.21
IVF-BF16-nl387 (self)                                    585.35     1_225.48     1_810.82       0.9798          1.0001            1.0000         9.21
IVF-BF16-nl547-np23 (query)                            1_119.12        85.15     1_204.26       0.9773          1.0005            1.0000         9.23
IVF-BF16-nl547-np27 (query)                            1_119.12        92.28     1_211.39       0.9816          1.0002            1.0000         9.23
IVF-BF16-nl547-np33 (query)                            1_119.12       108.45     1_227.57       0.9828          1.0001            1.0000         9.23
IVF-BF16-nl547 (self)                                  1_119.12     1_145.87     2_264.99       0.9798          1.0001            1.0000         9.23
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
Exhaustive (query)                                        12.08       722.94       735.01       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.08     7_355.83     7_367.91       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   15.46     1_263.42     1_278.88       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    15.46    12_868.08    12_883.54       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              301.61        94.16       395.78       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              301.61       111.05       412.66       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              301.61       146.26       447.88       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    301.61     1_531.28     1_832.89       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              547.24        98.70       645.94       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              547.24       125.40       672.64       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    547.24     1_307.61     1_854.85       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                            1_047.09        86.56     1_133.65       0.8848          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                            1_047.09        96.56     1_143.65       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                            1_047.09       118.13     1_165.22       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                  1_047.09     1_185.32     2_232.41       0.8852          1.0073            1.0020         9.51
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
Exhaustive (query)                                        11.78       640.54       652.31       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.78     6_410.21     6_421.99       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.04     1_235.55     1_248.59       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    13.04    12_104.44    12_117.48       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              469.99        84.62       554.61       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              469.99        91.89       561.88       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              469.99       118.01       588.00       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    469.99     1_184.01     1_654.00       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              569.78        84.55       654.33       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387-np27 (query)                              569.78       104.42       674.20       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387 (self)                                    569.78     1_051.44     1_621.22       0.9184          1.0030            1.0021         9.21
IVF-BF16-nl547-np23 (query)                            1_056.59        82.41     1_139.00       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_056.59        85.72     1_142.31       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_056.59        98.28     1_154.88       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547 (self)                                  1_056.59       979.41     2_036.00       0.9184          1.0030            1.0021         9.23
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
Exhaustive (query)                                        11.04       647.97       659.01       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.04     6_311.46     6_322.50       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.75     1_201.01     1_214.76       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    13.75    12_059.65    12_073.40       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              333.57        78.47       412.04       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              333.57        98.28       431.85       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              333.57       119.41       452.99       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    333.57     1_197.22     1_530.79       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              571.51        82.27       653.77       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387-np27 (query)                              571.51       141.82       713.33       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387 (self)                                    571.51     1_052.63     1_624.14       0.9429          1.0017            1.0009         9.21
IVF-BF16-nl547-np23 (query)                            1_089.10        77.28     1_166.38       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np27 (query)                            1_089.10        84.49     1_173.59       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np33 (query)                            1_089.10        97.12     1_186.22       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547 (self)                                  1_089.10       968.56     2_057.66       0.9429          1.0017            1.0009         9.23
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
Exhaustive (query)                                        50.55     1_219.63     1_270.18       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.55    12_014.53    12_065.08       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   62.91     5_187.16     5_250.07       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    62.91    53_079.59    53_142.50       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              644.85       273.77       918.62       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              644.85       309.63       954.48       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              644.85       427.99     1_072.84       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    644.85     4_327.95     4_972.80       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                            1_148.18       279.60     1_427.78       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                            1_148.18       367.16     1_515.34       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                  1_148.18     3_701.93     4_850.11       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            2_309.16       264.96     2_574.12       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            2_309.16       292.76     2_601.92       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            2_309.16       341.43     2_650.59       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                  2_309.16     3_399.30     5_708.46       0.9679          1.0005            1.0000        36.89
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
*96 x 8 bits = 96 bytes*, a **4x reduction** plus the codebook. The codebook is
fixed overhead, so the realised saving is 3.5x at 32 dimensions and 3.9x at 128.

Whether the integer kernels also make the scan faster depends on the index
under them, and the tables below split. The **exhaustive** SQ8 scan is *slower*
than the `f32` one at 32 dimensions, by up to 1.5x, and only edges ahead at 128:
one byte per dimension does not buy enough per-element work to pay for the
widening when `dim` is small. Under **IVF** it wins everywhere, 1.15 to 1.5x on
every matched `nlist`/`nprobe` pairing, because the cell scan is the whole
cost there.

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
Exhaustive (query)                                        11.36       665.55       676.91       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.36     6_229.54     6_240.90       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.79       993.10     1_010.89       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     17.79    10_031.18    10_048.97       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               366.25        66.00       432.25       0.9244          1.0020            1.0009         6.33
IVF-SQ8-nl273-np16 (query)                               366.25        70.33       436.58       0.9258          1.0018            1.0009         6.33
IVF-SQ8-nl273-np23 (query)                               366.25        94.64       460.89       0.9260          1.0018            1.0009         6.33
IVF-SQ8-nl273 (self)                                     366.25       911.72     1_277.96       0.9253          1.0018            1.0009         6.33
IVF-SQ8-nl387-np19 (query)                               586.44        66.91       653.35       0.9243          1.0019            1.0009         6.35
IVF-SQ8-nl387-np27 (query)                               586.44        82.60       669.04       0.9248          1.0018            1.0009         6.35
IVF-SQ8-nl387 (self)                                     586.44       813.54     1_399.98       0.9252          1.0018            1.0009         6.35
IVF-SQ8-nl547-np23 (query)                             1_167.65        59.27     1_226.92       0.9215          1.0022            1.0010         6.37
IVF-SQ8-nl547-np27 (query)                             1_167.65        66.45     1_234.10       0.9244          1.0019            1.0009         6.37
IVF-SQ8-nl547-np33 (query)                             1_167.65        78.59     1_246.24       0.9251          1.0018            1.0009         6.37
IVF-SQ8-nl547 (self)                                   1_167.65       781.04     1_948.69       0.9252          1.0018            1.0009         6.37
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
Exhaustive (query)                                        14.25       705.51       719.77       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         14.25     7_133.04     7_147.29       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    21.73       922.35       944.08       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     21.73    10_351.59    10_373.32       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               315.89        61.08       376.96       0.7391          1.0362            1.0153         6.33
IVF-SQ8-nl273-np16 (query)                               315.89        68.76       384.65       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273-np23 (query)                               315.89        89.89       405.77       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273 (self)                                     315.89     1_001.70     1_317.59       0.7378          1.0362            1.0152         6.33
IVF-SQ8-nl387-np19 (query)                               584.17        63.79       647.97       0.7378          1.0360            1.0155         6.35
IVF-SQ8-nl387-np27 (query)                               584.17        82.36       666.53       0.7379          1.0360            1.0155         6.35
IVF-SQ8-nl387 (self)                                     584.17       824.91     1_409.08       0.7375          1.0362            1.0157         6.35
IVF-SQ8-nl547-np23 (query)                             1_109.08        58.57     1_167.65       0.7365          1.0365            1.0165         6.37
IVF-SQ8-nl547-np27 (query)                             1_109.08        63.04     1_172.11       0.7370          1.0364            1.0163         6.37
IVF-SQ8-nl547-np33 (query)                             1_109.08        73.37     1_182.45       0.7369          1.0364            1.0162         6.37
IVF-SQ8-nl547 (self)                                   1_109.08       766.13     1_875.21       0.7360          1.0365            1.0159         6.37
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
Exhaustive (query)                                        12.08       655.39       667.47       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.08     6_440.55     6_452.63       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    32.40       996.94     1_029.34       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     32.40     9_934.77     9_967.17       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               332.37        62.26       394.63       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np16 (query)                               332.37        65.89       398.26       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np23 (query)                               332.37        83.16       415.53       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273 (self)                                     332.37       816.32     1_148.69       0.8121          1.0174            1.0155         6.33
IVF-SQ8-nl387-np19 (query)                               586.90        61.01       647.91       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387-np27 (query)                               586.90        76.71       663.61       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387 (self)                                     586.90       726.08     1_312.98       0.8119          1.0175            1.0155         6.35
IVF-SQ8-nl547-np23 (query)                             1_074.05        55.91     1_129.95       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np27 (query)                             1_074.05        61.15     1_135.19       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np33 (query)                             1_074.05        70.53     1_144.57       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547 (self)                                   1_074.05       684.01     1_758.06       0.8118          1.0175            1.0155         6.37
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
Exhaustive (query)                                        11.67       644.96       656.62       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.67     6_404.85     6_416.52       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.20       998.34     1_015.54       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     17.20     9_945.76     9_962.96       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                               327.50        58.06       385.55       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273-np16 (query)                               327.50        64.51       392.00       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273-np23 (query)                               327.50        87.25       414.74       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273 (self)                                     327.50       836.05     1_163.55       0.7899          1.0281            1.0257         6.33
IVF-SQ8-nl387-np19 (query)                               582.31        60.39       642.70       0.7899          1.0265            1.0244         6.35
IVF-SQ8-nl387-np27 (query)                               582.31        75.47       657.79       0.7899          1.0265            1.0244         6.35
IVF-SQ8-nl387 (self)                                     582.31       722.31     1_304.62       0.7903          1.0280            1.0256         6.35
IVF-SQ8-nl547-np23 (query)                             1_090.30        55.69     1_145.99       0.7897          1.0265            1.0243         6.37
IVF-SQ8-nl547-np27 (query)                             1_090.30        63.65     1_153.95       0.7897          1.0265            1.0243         6.37
IVF-SQ8-nl547-np33 (query)                             1_090.30        70.24     1_160.54       0.7897          1.0265            1.0243         6.37
IVF-SQ8-nl547 (self)                                   1_090.30       675.58     1_765.88       0.7899          1.0280            1.0256         6.37
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
Exhaustive (query)                                        50.77     1_237.17     1_287.94       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.77    11_952.14    12_002.91       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    88.52     1_176.10     1_264.62       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     88.52    12_071.55    12_160.07       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               660.75        80.10       740.85       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np16 (query)                               660.75        85.36       746.11       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np23 (query)                               660.75       111.90       772.65       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273 (self)                                     660.75       909.40     1_570.14       0.8865          1.0073            1.0059        20.16
IVF-SQ8-nl387-np19 (query)                             1_207.88        85.66     1_293.54       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387-np27 (query)                             1_207.88       102.97     1_310.85       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387 (self)                                   1_207.88       820.18     2_028.06       0.8867          1.0073            1.0059        20.22
IVF-SQ8-nl547-np23 (query)                             2_403.44        88.15     2_491.59       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np27 (query)                             2_403.44        91.13     2_494.58       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np33 (query)                             2_403.44       101.28     2_504.73       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547 (self)                                   2_403.44       798.23     3_201.68       0.8865          1.0073            1.0059        20.30
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### HNSW on SQ8 codes

An HNSW built **and** searched entirely on the uniform 8-bit codes described
above, inspired by [pyglass](https://github.com/zilliztech/pyglass). Because the
shared scale makes the integer code distance order-preserving, one kernel serves
graph construction and query alike: the graph never sees a float. The build and
the query both get faster, since everything is integer arithmetic.

The *vector store* drops 4x, but the graph edges do not compress, so the index
as a whole lands at 0.44 to 0.80 of a plain HNSW depending on `M` and
dimensionality. The ratio is worst at 32 dimensions, where the edges dominate.

The grid runs the full-precision HNSW at matched `(M, ef_construction,
ef_search)` alongside it, plus an exhaustive scan over the same codec. The
exhaustive-SQ8 row is the ceiling the graph rows work against: whatever they
lose up to it is the codec, whatever they lose beyond it is the graph.

Read the recall columns before the memory ones. At matched `(M=16, ef=200,
s=200)` the codec costs about 0.07 recall on Euclidean and **0.26 to 0.33 on
cosine**, and the graph rows sit essentially on the exhaustive-SQ8 ceiling, so
that loss is all codec. Cosine is the case to check on your own data.

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
Exhaustive (query)                                        11.79       679.44       691.24       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.79     6_528.46     6_540.26       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.76       996.51     1_015.27       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               901.52        48.94       950.45       0.9299          1.0177            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              901.52        89.42       990.94       0.9632          1.0104            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              901.52       160.42     1_061.94       0.9813          1.0069            1.0000        38.52
HNSW-M16-ef100 (self)                                    901.52       824.14     1_725.66       0.9631          1.0082            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_575.67        80.60     1_656.27       0.9581          1.0183            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_575.67        93.03     1_668.70       0.9822          1.0091            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_575.67       176.44     1_752.11       0.9918          1.0019            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_575.67       885.66     2_461.33       0.9831          1.0071            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_672.13        56.36     1_728.50       0.9694          1.0088            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_672.13       104.18     1_776.32       0.9887          1.0026            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_672.13       182.66     1_854.79       0.9955          1.0007            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_672.13       976.96     2_649.09       0.9886          1.0034            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_768.83        60.92     1_829.74       0.9727          1.0137            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_768.83       105.15     1_873.98       0.9897          1.0057            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_768.83       191.89     1_960.71       0.9965          1.0013            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_768.83     1_042.46     2_811.29       0.9905          1.0026            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          725.31        38.02       763.33       0.8773          1.0221            1.0032        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         725.31        67.46       792.77       0.9021          1.0146            1.0020        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         725.31       125.75       851.06       0.9148          1.0098            1.0014        26.89
HNSW-SQ8U-M16-ef100 (self)                               725.31       656.09     1_381.40       0.9015          1.0122            1.0020        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_373.47        36.81     1_410.29       0.8984          1.0148            1.0021        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_373.47        69.56     1_443.04       0.9152          1.0038            1.0014        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_373.47       133.26     1_506.73       0.9209          1.0029            1.0011        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_373.47       714.34     2_087.81       0.9147          1.0059            1.0014        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_488.63        41.79     1_530.41       0.9068          1.0090            1.0017        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_488.63        80.01     1_568.64       0.9186          1.0052            1.0012        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_488.63       149.51     1_638.14       0.9229          1.0031            1.0010        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_488.63       739.45     2_228.08       0.9181          1.0059            1.0012        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_631.83        45.72     1_677.55       0.9078          1.0213            1.0016        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_631.83        79.86     1_711.69       0.9188          1.0137            1.0012        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_631.83       154.84     1_786.67       0.9230          1.0059            1.0010        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_631.83       789.59     2_421.42       0.9187          1.0097            1.0012        45.20
HNSW-SQ8U-drop0 (query)                                1_398.20        68.07     1_466.27       0.8951          1.0161            1.0024        26.89
HNSW-SQ8U-drop0.001 (query)                            1_384.76        70.17     1_454.93       0.9146          1.0063            1.0014        26.89
HNSW-SQ8U-drop0.01 (query)                             1_415.69        75.41     1_491.10       0.8986          1.0087            1.0018        26.89
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
Exhaustive (query)                                        11.88       711.23       723.10       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.88     6_987.80     6_999.68       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    20.70       922.73       943.44       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               875.63        50.66       926.30       0.9361          1.0127            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              875.63        91.26       966.89       0.9698          1.0080            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              875.63       183.52     1_059.15       0.9878          1.0047            1.0000        39.09
HNSW-M16-ef100 (self)                                    875.63       856.56     1_732.19       0.9690          1.0085            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_625.96        64.26     1_690.21       0.9656          1.0084            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_625.96        94.35     1_720.30       0.9877          1.0015            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_625.96       178.30     1_804.25       0.9954          1.0007            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_625.96       948.42     2_574.38       0.9872          1.0049            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_780.96        67.67     1_848.63       0.9736          1.0024            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_780.96       113.64     1_894.60       0.9912          1.0006            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_780.96       200.31     1_981.28       0.9970          1.0003            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_780.96     1_110.76     2_891.73       0.9910          1.0011            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_802.05        64.37     1_866.42       0.9760          1.0019            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_802.05       109.45     1_911.50       0.9918          1.0009            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_802.05       194.41     1_996.46       0.9973          1.0005            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_802.05     1_079.49     2_881.54       0.9919          1.0011            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          776.83        49.13       825.96       0.6854          1.0562            1.0289        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         776.83        68.41       845.24       0.7093          1.0463            1.0239        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         776.83       124.67       901.50       0.7228          1.0427            1.0206        26.89
HNSW-SQ8U-M16-ef100 (self)                               776.83       636.10     1_412.93       0.7088          1.0486            1.0238        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_450.63        38.41     1_489.04       0.7069          1.0447            1.0233        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_450.63        69.95     1_520.59       0.7245          1.0405            1.0199        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_450.63       132.22     1_582.85       0.7320          1.0387            1.0183        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_450.63       684.27     2_134.90       0.7236          1.0408            1.0198        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_555.85        45.42     1_601.27       0.7177          1.0405            1.0207        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_555.85        82.18     1_638.03       0.7302          1.0381            1.0183        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_555.85       145.58     1_701.43       0.7354          1.0372            1.0171        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_555.85       754.34     2_310.19       0.7296          1.0383            1.0181        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_692.11        44.30     1_736.41       0.7221          1.0387            1.0197        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_692.11        84.20     1_776.31       0.7329          1.0371            1.0177        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_692.11       144.82     1_836.92       0.7368          1.0364            1.0168        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_692.11       787.20     2_479.30       0.7321          1.0382            1.0173        45.20
HNSW-SQ8U-drop0 (query)                                1_444.30        97.38     1_541.69       0.6642          1.0649            1.0306        26.89
HNSW-SQ8U-drop0.001 (query)                            1_443.24        99.86     1_543.10       0.7240          1.0402            1.0201        26.89
HNSW-SQ8U-drop0.01 (query)                             1_409.95        75.07     1_485.02       0.6895          1.0538            1.0261        26.89
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
Exhaustive (query)                                        11.05       692.84       703.89       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.05     6_872.27     6_883.32       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.80     1_030.91     1_049.71       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               852.56        50.41       902.97       0.9670          2.2715            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              852.56        89.88       942.44       0.9699          1.0074            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              852.56       168.89     1_021.45       0.9702          1.0074            1.0000        38.52
HNSW-M16-ef100 (self)                                    852.56       841.46     1_694.02       0.9701          1.0076            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_468.90        54.57     1_523.47       0.9972          2.0503            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_468.90        91.55     1_560.44       0.9997          1.0001            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_468.90       156.61     1_625.51       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_468.90       886.13     2_355.03       0.9998          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_495.48        53.37     1_548.85       0.9991          1.0054            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_495.48        89.79     1_585.27       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_495.48       159.64     1_655.12       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_495.48       881.51     2_376.99       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_555.12        53.27     1_608.38       0.9993          1.0001            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_555.12        99.61     1_654.73       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_555.12       158.94     1_714.05       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_555.12       874.22     2_429.34       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          728.27        38.22       766.50       0.8134          1.3447            1.0149        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         728.27        66.78       795.05       0.8141          1.2755            1.0148        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         728.27       119.32       847.60       0.8143          1.0935            1.0148        26.89
HNSW-SQ8U-M16-ef100 (self)                               728.27       607.78     1_336.05       0.8116          1.1471            1.0156        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_286.27        35.73     1_322.00       0.8143          1.0166            1.0148        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_286.27        66.10     1_352.37       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_286.27       118.46     1_404.73       0.8146          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_286.27       618.75     1_905.02       0.8119          1.0175            1.0155        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_422.00        42.64     1_464.64       0.8143          1.2388            1.0148        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_422.00        72.44     1_494.44       0.8145          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_422.00       143.63     1_565.63       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_422.00       663.46     2_085.46       0.8119          1.0175            1.0155        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_460.59        42.30     1_502.89       0.8144          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_460.59        73.60     1_534.19       0.8145          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_460.59       128.45     1_589.04       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_460.59       689.74     2_150.33       0.8119          1.0175            1.0155        45.20
HNSW-SQ8U-drop0 (query)                                1_303.58        65.20     1_368.78       0.8081          1.0176            1.0157        26.89
HNSW-SQ8U-drop0.001 (query)                            1_319.19        66.89     1_386.08       0.8145          1.0936            1.0148        26.89
HNSW-SQ8U-drop0.01 (query)                             1_296.85        66.22     1_363.07       0.8049          1.0195            1.0163        26.89
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
Exhaustive (query)                                        11.78       683.28       695.06       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.78     6_945.43     6_957.21       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    23.14     1_029.14     1_052.28       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               907.65        56.54       964.20       0.9981          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              907.65        94.68     1_002.34       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              907.65       171.80     1_079.45       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    907.65       933.15     1_840.81       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_603.28        53.60     1_656.88       0.9986          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_603.28        94.67     1_697.95       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_603.28       173.95     1_777.22       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_603.28       963.11     2_566.39       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_708.72        60.47     1_769.19       0.9993          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_708.72       106.94     1_815.66       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_708.72       193.51     1_902.23       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_708.72     1_012.90     2_721.62       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_779.10        63.11     1_842.21       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_779.10       110.63     1_889.73       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_779.10       195.50     1_974.60       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_779.10     1_058.25     2_837.35       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          797.00        41.64       838.64       0.7889          1.0268            1.0246        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         797.00        77.54       874.53       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         797.00       134.96       931.96       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100 (self)                               797.00       686.01     1_483.01       0.7896          1.0282            1.0258        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_425.42        46.88     1_472.30       0.7891          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_425.42        85.01     1_510.43       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_425.42       136.92     1_562.34       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_425.42       701.13     2_126.55       0.7897          1.0281            1.0258        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_545.99        45.51     1_591.50       0.7891          1.0267            1.0245        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_545.99        80.20     1_626.19       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_545.99       146.50     1_692.50       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_545.99       771.72     2_317.71       0.7897          1.0281            1.0258        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_636.96        47.47     1_684.43       0.7891          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_636.96        87.05     1_724.02       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_636.96       152.68     1_789.64       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_636.96       802.79     2_439.76       0.7897          1.0281            1.0258        45.20
HNSW-SQ8U-drop0 (query)                                1_434.61        72.57     1_507.18       0.7860          1.0279            1.0254        26.89
HNSW-SQ8U-drop0.001 (query)                            1_454.31        74.06     1_528.37       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-drop0.01 (query)                             1_434.96        72.04     1_506.99       0.7830          1.0294            1.0262        26.89
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
Exhaustive (query)                                        53.20     1_318.61     1_371.81       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         53.20    13_011.85    13_065.04       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    85.04     1_228.37     1_313.40       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_387.63        80.62     1_468.25       0.9946          1.0269            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_387.63       137.98     1_525.61       0.9964          1.0139            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_387.63       246.15     1_633.77       0.9981          1.0043            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_387.63     1_377.60     2_765.23       0.9964          1.0159            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_513.55        84.68     2_598.23       0.9969          1.0205            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_513.55       147.33     2_660.89       0.9983          1.0092            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_513.55       255.22     2_768.77       0.9990          1.0046            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_513.55     1_439.97     3_953.52       0.9978          1.0135            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_663.04        89.62     2_752.66       0.9983          1.0095            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_663.04       153.16     2_816.20       0.9992          1.0035            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_663.04       266.96     2_930.00       0.9995          1.0018            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_663.04     1_514.66     4_177.69       0.9991          1.0040            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_748.19        96.32     2_844.51       0.9987          1.0080            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_748.19       159.78     2_907.98       0.9993          1.0038            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_748.19       274.72     3_022.91       0.9998          1.0003            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_748.19     1_569.61     4_317.80       0.9994          1.0030            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          860.13        40.89       901.02       0.9288          1.0339            1.0038        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         860.13        72.12       932.25       0.9302          1.0248            1.0038        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         860.13       134.74       994.86       0.9317          1.0167            1.0037        40.63
HNSW-SQ8U-M16-ef100 (self)                               860.13       683.47     1_543.60       0.9306          1.0223            1.0038        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_561.62        41.96     1_603.58       0.9321          1.0203            1.0036        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_561.62        76.95     1_638.57       0.9327          1.0161            1.0036        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_561.62       142.67     1_704.30       0.9335          1.0106            1.0036        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_561.62       720.68     2_282.30       0.9325          1.0169            1.0037        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_644.67        46.05     1_690.72       0.9322          1.0178            1.0036        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_644.67        80.26     1_724.93       0.9330          1.0132            1.0036        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_644.67       142.16     1_786.84       0.9337          1.0082            1.0036        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_644.67       763.44     2_408.11       0.9331          1.0111            1.0036        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_744.27        47.45     1_791.73       0.9330          1.0148            1.0036        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_744.27        86.95     1_831.23       0.9335          1.0115            1.0036        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_744.27       142.96     1_887.23       0.9339          1.0084            1.0036        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_744.27       779.60     2_523.87       0.9332          1.0117            1.0036        58.94
HNSW-SQ8U-drop0 (query)                                1_567.37        75.16     1_642.53       0.8638          1.0454            1.0221        40.63
HNSW-SQ8U-drop0.001 (query)                            1_546.93        75.87     1_622.80       0.9323          1.0169            1.0036        40.63
HNSW-SQ8U-drop0.01 (query)                             1_556.41        73.95     1_630.36       0.9317          1.0405            1.0020        40.63
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
Exhaustive (query)                                        57.88     1_330.77     1_388.65       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         57.88    13_349.40    13_407.28       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                   107.68     1_267.45     1_375.13       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_274.87        69.25     1_344.12       0.9934          1.1148            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_274.87       119.76     1_394.64       0.9965          1.0407            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_274.87       213.99     1_488.86       0.9977          1.0188            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_274.87     1_161.14     2_436.02       0.9961          1.0468            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_312.08        74.55     2_386.62       0.9944          1.1067            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_312.08       127.45     2_439.53       0.9972          1.0416            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_312.08       224.50     2_536.58       0.9986          1.0166            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_312.08     1_236.31     3_548.39       0.9970          1.0558            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_478.59        78.98     2_557.56       0.9984          1.0229            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_478.59       131.30     2_609.89       0.9989          1.0136            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_478.59       236.51     2_715.10       0.9996          1.0053            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_478.59     1_258.21     3_736.80       0.9993          1.0073            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_550.62        77.44     2_628.06       0.9983          1.0731            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_550.62       135.35     2_685.96       0.9990          1.0323            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_550.62       235.00     2_785.62       0.9997          1.0033            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_550.62     1_275.99     3_826.61       0.9991          1.0274            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          843.72        49.23       892.95       0.6633          1.4243            1.1646        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         843.72        66.71       910.43       0.6648          1.3810            1.1635        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         843.72       122.91       966.63       0.6661          1.3579            1.1624        40.63
HNSW-SQ8U-M16-ef100 (self)                               843.72       654.18     1_497.90       0.6644          1.3883            1.1640        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_517.63        46.26     1_563.89       0.6635          1.4376            1.1638        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_517.63        79.24     1_596.87       0.6655          1.3946            1.1624        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_517.63       132.11     1_649.74       0.6664          1.3708            1.1620        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_517.63       677.58     2_195.21       0.6649          1.3970            1.1631        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_624.70        45.55     1_670.25       0.6662          1.3684            1.1621        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_624.70        75.78     1_700.49       0.6669          1.3556            1.1616        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_624.70       131.33     1_756.04       0.6673          1.3497            1.1614        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_624.70       722.99     2_347.69       0.6664          1.3642            1.1620        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_691.93        54.73     1_746.65       0.6659          1.3924            1.1621        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_691.93        77.10     1_769.02       0.6664          1.3752            1.1619        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_691.93       134.26     1_826.19       0.6672          1.3534            1.1614        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_691.93       741.15     2_433.08       0.6663          1.3810            1.1619        58.94
HNSW-SQ8U-drop0 (query)                                1_523.82        69.39     1_593.21       0.6192          1.5345            1.2368        40.63
HNSW-SQ8U-drop0.001 (query)                            1_539.57        64.56     1_604.13       0.6654          1.3923            1.1627        40.63
HNSW-SQ8U-drop0.01 (query)                             1_516.47        71.94     1_588.41       0.6782          1.3596            1.1508        40.63
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
Exhaustive (query)                                        33.17       710.47       743.64       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.17     2_343.80     2_376.97       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                643.10       665.22     1_308.32       0.2581          1.1827            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 643.10     2_197.72     2_840.81       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_167.89     1_521.91     2_689.80       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_167.89     5_055.12     6_223.01       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              1_928.67     3_602.32     5_530.98       0.3611          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               1_928.67    12_035.88    13_964.54       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_412.71       201.20     1_613.91       0.3712          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_412.71       311.52     1_724.23       0.3712          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_412.71       425.16     1_837.86       0.3712          1.0979            1.1001         1.17
IVF-PQ-nl158-m16 (self)                                1_412.71     1_426.87     2_839.58       0.3042          1.1282            1.1332         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_851.23       371.48     2_222.70       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_851.23       563.51     2_414.74       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_851.23       749.00     2_600.22       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32 (self)                                1_851.23     2_502.10     4_353.32       0.4068          1.0804            1.0800         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_503.36       647.06     3_150.42       0.6905          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_503.36       988.68     3_492.04       0.6905          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_503.36     1_334.79     3_838.15       0.6905          1.0199            1.0166         3.46
IVF-PQ-nl158-m64 (self)                                2_503.36     4_445.47     6_948.83       0.6338          1.0271            1.0243         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_191.69       293.90     1_485.59       0.3868          1.0887            1.0895         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_191.69       357.87     1_549.56       0.3868          1.0888            1.0896         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_191.69       519.84     1_711.52       0.3868          1.0888            1.0896         1.23
IVF-PQ-nl223-m16 (self)                                1_191.69     1_746.39     2_938.08       0.3098          1.1230            1.1272         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_648.48       501.70     2_150.18       0.4975          1.0564            1.0517         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_648.48       627.79     2_276.27       0.4975          1.0565            1.0517         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_648.48       905.31     2_553.78       0.4975          1.0565            1.0517         2.00
IVF-PQ-nl223-m32 (self)                                1_648.48     3_001.98     4_650.46       0.4146          1.0780            1.0755         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_325.70       893.97     3_219.67       0.6981          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_325.70     1_107.44     3_433.13       0.6982          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_325.70     1_640.00     3_965.70       0.6982          1.0199            1.0155         3.52
IVF-PQ-nl223-m64 (self)                                2_325.70     5_305.13     7_630.83       0.6387          1.0273            1.0235         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_432.58       371.22     1_803.80       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_432.58       414.78     1_847.36       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_432.58       589.30     2_021.88       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16 (self)                                1_432.58     1_974.70     3_407.29       0.3156          1.1188            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_861.22       640.62     2_501.83       0.5113          1.0520            1.0487         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_861.22       718.12     2_579.34       0.5113          1.0520            1.0487         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_861.22     1_040.11     2_901.33       0.5113          1.0520            1.0487         2.09
IVF-PQ-nl316-m32 (self)                                1_861.22     3_417.68     5_278.90       0.4237          1.0742            1.0728         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_437.07     1_146.83     3_583.90       0.7072          1.0175            1.0146         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_437.07     1_287.54     3_724.61       0.7072          1.0174            1.0146         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_437.07     1_851.73     4_288.79       0.7072          1.0174            1.0146         3.61
IVF-PQ-nl316-m64 (self)                                2_437.07     6_281.13     8_718.19       0.6490          1.0248            1.0221         3.61
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
Exhaustive (query)                                        68.39     1_286.99     1_355.38       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.39     4_212.01     4_280.40       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                881.73       681.33     1_563.06       0.2443          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                                 881.73     2_242.90     3_124.63       0.2277          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_289.58     1_534.28     2_823.85       0.2649          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_289.58     5_054.32     6_343.90       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_153.70     3_618.45     5_772.14       0.2958          1.0991            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_153.70    11_973.75    14_127.45       0.2627          1.1103            1.1143         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_441.87       272.58     2_714.45       0.3075          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_441.87       422.85     2_864.72       0.3075          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_441.87       601.46     3_043.33       0.3075          1.0878            1.0922         1.57
IVF-PQ-nl158-m16 (self)                                2_441.87     1_929.76     4_371.64       0.2625          1.1081            1.1146         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_864.24       398.60     3_262.84       0.3543          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_864.24       625.53     3_489.77       0.3543          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_864.24       835.35     3_699.59       0.3543          1.0712            1.0721         2.34
IVF-PQ-nl158-m32 (self)                                2_864.24     2_765.30     5_629.53       0.2914          1.0914            1.0953         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_751.54       722.60     4_474.14       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_751.54     1_124.83     4_876.37       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_751.54     1_550.02     5_301.56       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64 (self)                                3_751.54     5_135.48     8_887.02       0.3902          1.0584            1.0572         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_861.16       414.06     2_275.22       0.3166          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_861.16       495.71     2_356.87       0.3166          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_861.16       721.15     2_582.31       0.3166          1.0827            1.0852         1.70
IVF-PQ-nl223-m16 (self)                                1_861.16     2_397.97     4_259.13       0.2659          1.1044            1.1096         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_229.30       576.57     2_805.87       0.3685          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_229.30       706.46     2_935.76       0.3686          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_229.30     1_048.54     3_277.84       0.3686          1.0656            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                2_229.30     3_392.54     5_621.83       0.2945          1.0892            1.0917         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_133.61     1_054.47     4_188.08       0.4769          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_133.61     1_277.11     4_410.72       0.4769          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_133.61     1_861.30     4_994.91       0.4769          1.0429            1.0387         3.99
IVF-PQ-nl223-m64 (self)                                3_133.61     6_172.17     9_305.78       0.3958          1.0576            1.0553         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_134.11       535.34     2_669.46       0.3273          1.0764            1.0806         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_134.11       591.25     2_725.36       0.3273          1.0764            1.0806         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_134.11       847.86     2_981.97       0.3273          1.0765            1.0806         1.88
IVF-PQ-nl316-m16 (self)                                2_134.11     2_802.85     4_936.96       0.2709          1.0999            1.1061         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_535.41       748.76     3_284.17       0.3790          1.0612            1.0620         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_535.41       822.65     3_358.06       0.3790          1.0612            1.0620         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_535.41     1_166.53     3_701.93       0.3790          1.0612            1.0620         2.65
IVF-PQ-nl316-m32 (self)                                2_535.41     3_857.32     6_392.73       0.2992          1.0859            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_480.79     1_335.62     4_816.42       0.4883          1.0396            1.0363         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_480.79     1_499.40     4_980.20       0.4883          1.0396            1.0363         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_480.79     2_131.72     5_612.52       0.4883          1.0396            1.0363         4.17
IVF-PQ-nl316-m64 (self)                                3_480.79     7_056.42    10_537.22       0.4046          1.0543            1.0533         4.17
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
Exhaustive (query)                                       100.60     1_823.10     1_923.71       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.60     6_062.36     6_162.97       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_161.28       695.14     1_856.42       0.2345          1.1095            1.1000         1.51
Exhaustive-PQ-m16 (self)                               1_161.28     2_243.70     3_404.98       0.2206          1.1180            1.1048         1.51
Exhaustive-PQ-m32 (query)                              1_693.05     1_564.86     3_257.91       0.2567          1.0943            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_693.05     5_142.91     6_835.96       0.2391          1.1012            1.1021         2.28
Exhaustive-PQ-m64 (query)                              2_630.64     3_659.13     6_289.78       0.2775          1.0855            1.0910         3.80
Exhaustive-PQ-m64 (self)                               2_630.64    12_078.47    14_709.11       0.2515          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_555.09     7_960.24    12_515.32       0.3162          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_555.09    26_461.98    31_017.07       0.2755          1.0817            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_344.42       366.31     3_710.73       0.2858          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_344.42       568.03     3_912.45       0.2858          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_344.42       771.79     4_116.20       0.2858          1.0780            1.0843         1.98
IVF-PQ-nl158-m16 (self)                                3_344.42     2_559.22     5_903.64       0.2511          1.0938            1.1010         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_849.39       538.13     4_387.52       0.3151          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_849.39       841.71     4_691.11       0.3151          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_849.39     1_161.40     5_010.79       0.3151          1.0674            1.0715         2.74
IVF-PQ-nl158-m32 (self)                                3_849.39     3_857.60     7_707.00       0.2628          1.0855            1.0913         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_783.71       821.70     5_605.41       0.3782          1.0519            1.0512         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_783.71     1_304.78     6_088.49       0.3782          1.0519            1.0512         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_783.71     1_777.76     6_561.47       0.3782          1.0519            1.0512         4.27
IVF-PQ-nl158-m64 (self)                                4_783.71     5_879.48    10_663.19       0.3104          1.0662            1.0675         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_761.00     1_578.71     8_339.72       0.5354          1.0270            1.0231         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_761.00     2_481.56     9_242.56       0.5354          1.0270            1.0231         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_761.00     3_395.21    10_156.21       0.5354          1.0270            1.0231         7.32
IVF-PQ-nl158-m128 (self)                               6_761.00    11_253.78    18_014.78       0.4638          1.0343            1.0320         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_542.06       532.19     3_074.25       0.2961          1.0726            1.0765         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_542.06       615.42     3_157.48       0.2961          1.0726            1.0765         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_542.06       896.54     3_438.60       0.2961          1.0726            1.0765         2.17
IVF-PQ-nl223-m16 (self)                                2_542.06     2_962.76     5_504.82       0.2553          1.0893            1.0954         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_952.46       740.16     3_692.62       0.3306          1.0612            1.0633         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_952.46       911.08     3_863.54       0.3306          1.0612            1.0633         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_952.46     1_324.59     4_277.05       0.3306          1.0612            1.0633         2.93
IVF-PQ-nl223-m32 (self)                                2_952.46     4_363.82     7_316.28       0.2674          1.0815            1.0862         2.93
IVF-PQ-nl223-m64-np11 (query)                          4_035.79     1_152.97     5_188.75       0.3932          1.0479            1.0459         4.46
IVF-PQ-nl223-m64-np14 (query)                          4_035.79     1_429.36     5_465.15       0.3932          1.0479            1.0459         4.46
IVF-PQ-nl223-m64-np21 (query)                          4_035.79     2_075.63     6_111.42       0.3932          1.0479            1.0459         4.46
IVF-PQ-nl223-m64 (self)                                4_035.79     6_929.66    10_965.45       0.3132          1.0650            1.0654         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_958.85     2_272.20     8_231.05       0.5467          1.0259            1.0213         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_958.85     2_825.89     8_784.74       0.5467          1.0259            1.0213         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_958.85     4_111.89    10_070.74       0.5467          1.0259            1.0213         7.51
IVF-PQ-nl223-m128 (self)                               5_958.85    13_664.95    19_623.80       0.4702          1.0336            1.0307         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_907.84       666.36     3_574.20       0.3049          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_907.84       736.48     3_644.31       0.3049          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_907.84     1_055.49     3_963.32       0.3049          1.0680            1.0728         2.44
IVF-PQ-nl316-m16 (self)                                2_907.84     3_540.71     6_448.54       0.2596          1.0853            1.0916         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_412.08     1_004.97     4_417.05       0.3356          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_412.08     1_110.21     4_522.29       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_412.08     1_607.71     5_019.80       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32 (self)                                3_412.08     5_356.34     8_768.43       0.2692          1.0794            1.0842         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_410.64     1_542.60     5_953.24       0.4015          1.0454            1.0438         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_410.64     1_729.53     6_140.17       0.4015          1.0454            1.0438         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_410.64     2_482.15     6_892.80       0.4015          1.0454            1.0438         4.73
IVF-PQ-nl316-m64 (self)                                4_410.64     8_229.80    12_640.44       0.3194          1.0625            1.0635         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_246.60     2_989.16     9_235.76       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_246.60     3_331.62     9_578.21       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_246.60     4_770.31    11_016.91       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128 (self)                               6_246.60    15_980.41    22_227.01       0.4776          1.0316            1.0299         7.78
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
Exhaustive (query)                                        33.01       714.49       747.49       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.01     2_368.65     2_401.66       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                658.36       664.04     1_322.40       0.2932          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 658.36     2_193.90     2_852.27       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                              1_161.37     1_518.71     2_680.08       0.4008          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                               1_161.37     5_045.69     6_207.06       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                              1_845.67     3_829.29     5_674.96       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                               1_845.67    11_968.31    13_813.98       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_489.47       194.11     1_683.57       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_489.47       301.11     1_790.58       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_489.47       422.48     1_911.94       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16 (self)                                1_489.47     1_392.46     2_881.93       0.4285          1.1644            1.1604         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_969.84       349.64     2_319.48       0.6747          1.0398            1.0375         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_969.84       553.26     2_523.10       0.6747          1.0398            1.0375         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_969.84       768.32     2_738.16       0.6747          1.0398            1.0375         1.93
IVF-PQ-nl158-m32 (self)                                1_969.84     2_573.67     4_543.51       0.6053          1.0692            1.0642         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_614.73       628.57     3_243.30       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_614.73       990.48     3_605.21       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_614.73     1_372.37     3_987.10       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                2_614.73     4_551.90     7_166.63       0.7986          1.0164            1.0142         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_229.41       288.77     1_518.18       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_229.41       353.73     1_583.13       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_229.41       524.36     1_753.77       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16 (self)                                1_229.41     1_751.73     2_981.14       0.4250          1.1675            1.1633         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_726.10       530.74     2_256.84       0.6754          1.0394            1.0372         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_726.10       644.57     2_370.67       0.6755          1.0394            1.0372         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_726.10       939.80     2_665.89       0.6755          1.0394            1.0372         2.00
IVF-PQ-nl223-m32 (self)                                1_726.10     3_120.63     4_846.72       0.6039          1.0699            1.0652         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_320.43       895.73     3_216.15       0.8359          1.0092            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_320.43     1_110.75     3_431.18       0.8360          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_320.43     1_647.08     3_967.51       0.8360          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                2_320.43     5_420.37     7_740.79       0.8003          1.0160            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_442.10       375.07     1_817.17       0.5373          1.0875            1.0847         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_442.10       418.96     1_861.06       0.5373          1.0875            1.0847         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_442.10       596.85     2_038.95       0.5373          1.0875            1.0847         1.32
IVF-PQ-nl316-m16 (self)                                1_442.10     1_989.63     3_431.73       0.4156          1.1742            1.1702         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_915.81       663.77     2_579.58       0.6784          1.0387            1.0364         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_915.81       743.65     2_659.46       0.6784          1.0387            1.0364         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_915.81     1_074.47     2_990.28       0.6784          1.0387            1.0364         2.09
IVF-PQ-nl316-m32 (self)                                1_915.81     3_489.08     5_404.89       0.6009          1.0711            1.0665         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_551.56     1_142.80     3_694.36       0.8384          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_551.56     1_286.71     3_838.27       0.8385          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_551.56     1_864.37     4_415.93       0.8385          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                2_551.56     6_189.78     8_741.34       0.8030          1.0155            1.0136         3.61
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
Exhaustive (query)                                        68.71     1_280.13     1_348.84       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.71     4_173.12     4_241.84       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                886.76       682.46     1_569.22       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 886.76     2_218.87     3_105.63       0.1772          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_267.54     1_538.87     2_806.41       0.2802          1.1736            1.1699         2.03
Exhaustive-PQ-m32 (self)                               1_267.54     5_064.87     6_332.41       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                              2_164.16     3_616.05     5_780.21       0.3752          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                               2_164.16    12_000.06    14_164.22       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_329.29       266.06     2_595.35       0.3774          1.1193            1.1182         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_329.29       414.45     2_743.74       0.3774          1.1193            1.1182         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_329.29       570.09     2_899.37       0.3774          1.1193            1.1182         1.57
IVF-PQ-nl158-m16 (self)                                2_329.29     1_876.06     4_205.34       0.2721          1.2081            1.2110         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_757.18       413.47     3_170.65       0.4902          1.0728            1.0712         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_757.18       613.72     3_370.90       0.4902          1.0728            1.0712         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_757.18       841.63     3_598.81       0.4902          1.0728            1.0712         2.34
IVF-PQ-nl158-m32 (self)                                2_757.18     2_753.94     5_511.13       0.3931          1.1248            1.1233         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_902.42       711.58     4_614.00       0.6308          1.0351            1.0335         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_902.42     1_144.78     5_047.20       0.6308          1.0351            1.0335         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_902.42     1_559.89     5_462.31       0.6308          1.0351            1.0335         3.86
IVF-PQ-nl158-m64 (self)                                3_902.42     5_122.58     9_025.00       0.5743          1.0543            1.0502         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_869.17       408.62     2_277.79       0.3791          1.1177            1.1178         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_869.17       495.61     2_364.77       0.3791          1.1177            1.1178         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_869.17       722.99     2_592.16       0.3791          1.1177            1.1178         1.70
IVF-PQ-nl223-m16 (self)                                1_869.17     2_425.76     4_294.93       0.2686          1.2119            1.2160         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_316.60       575.25     2_891.85       0.4903          1.0726            1.0713         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_316.60       702.84     3_019.44       0.4903          1.0726            1.0713         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_316.60     1_034.17     3_350.77       0.4903          1.0726            1.0713         2.46
IVF-PQ-nl223-m32 (self)                                2_316.60     3_391.65     5_708.25       0.3846          1.1297            1.1284         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_206.78     1_026.73     4_233.51       0.6326          1.0345            1.0330         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_206.78     1_270.14     4_476.92       0.6326          1.0345            1.0330         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_206.78     1_868.26     5_075.03       0.6326          1.0345            1.0330         3.99
IVF-PQ-nl223-m64 (self)                                3_206.78     6_192.13     9_398.91       0.5726          1.0548            1.0506         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_330.98       529.71     2_860.69       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_330.98       582.05     2_913.04       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_330.98       839.77     3_170.75       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16 (self)                                2_330.98     2_752.84     5_083.82       0.2624          1.2171            1.2220         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_748.84       734.65     3_483.49       0.4886          1.0729            1.0718         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_748.84       814.26     3_563.09       0.4886          1.0729            1.0718         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_748.84     1_167.85     3_916.68       0.4886          1.0729            1.0718         2.65
IVF-PQ-nl316-m32 (self)                                2_748.84     3_856.64     6_605.48       0.3735          1.1353            1.1345         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_664.57     1_325.78     4_990.35       0.6354          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_664.57     1_482.40     5_146.97       0.6354          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_664.57     2_132.37     5_796.94       0.6354          1.0339            1.0324         4.17
IVF-PQ-nl316-m64 (self)                                3_664.57     7_028.69    10_693.26       0.5697          1.0554            1.0517         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.23     1_858.41     1_959.65       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.23     6_026.59     6_127.82       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_165.80       700.08     1_865.88       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                               1_165.80     2_254.88     3_420.67       0.1758          1.3086            1.3090         1.51
Exhaustive-PQ-m32 (query)                              1_668.38     1_554.61     3_222.99       0.2712          1.1686            1.1636         2.28
Exhaustive-PQ-m32 (self)                               1_668.38     5_111.55     6_779.93       0.2191          1.2527            1.2505         2.28
Exhaustive-PQ-m64 (query)                              2_592.24     3_629.68     6_221.92       0.3546          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                               2_592.24    12_087.40    14_679.64       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             4_729.66     7_976.98    12_706.65       0.4597          1.0781            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_729.66    26_495.84    31_225.50       0.3908          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_315.20       352.59     3_667.79       0.3632          1.1174            1.1165         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_315.20       545.37     3_860.57       0.3632          1.1174            1.1165         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_315.20       753.19     4_068.39       0.3632          1.1174            1.1165         1.98
IVF-PQ-nl158-m16 (self)                                3_315.20     2_516.30     5_831.50       0.2608          1.2172            1.2210         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_939.40       531.03     4_470.43       0.4652          1.0759            1.0742         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_939.40       835.40     4_774.80       0.4652          1.0759            1.0742         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_939.40     1_150.95     5_090.34       0.4652          1.0759            1.0742         2.74
IVF-PQ-nl158-m32 (self)                                3_939.40     3_831.31     7_770.71       0.3690          1.1372            1.1366         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_940.26       809.98     5_750.24       0.5782          1.0438            1.0420         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_940.26     1_283.79     6_224.05       0.5782          1.0438            1.0420         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_940.26     1_768.23     6_708.49       0.5782          1.0438            1.0420         4.27
IVF-PQ-nl158-m64 (self)                                4_940.26     5_829.05    10_769.31       0.5192          1.0707            1.0668         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_387.58     1_523.22     7_910.79       0.7381          1.0160            1.0142         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_387.58     2_535.51     8_923.08       0.7381          1.0160            1.0142         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_387.58     3_441.16     9_828.74       0.7381          1.0160            1.0142         7.32
IVF-PQ-nl158-m128 (self)                               6_387.58    11_046.57    17_434.14       0.7141          1.0239            1.0193         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_512.80       498.85     3_011.64       0.3621          1.1176            1.1180         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_512.80       601.97     3_114.76       0.3621          1.1176            1.1180         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_512.80       874.83     3_387.63       0.3621          1.1176            1.1180         2.17
IVF-PQ-nl223-m16 (self)                                2_512.80     2_886.97     5_399.77       0.2537          1.2248            1.2300         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_955.67       731.26     3_686.92       0.4607          1.0772            1.0760         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_955.67       905.24     3_860.90       0.4607          1.0772            1.0760         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_955.67     1_317.86     4_273.52       0.4607          1.0772            1.0760         2.93
IVF-PQ-nl223-m32 (self)                                2_955.67     4_313.56     7_269.22       0.3487          1.1494            1.1490         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_846.14     1_135.82     4_981.96       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_846.14     1_417.41     5_263.55       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_846.14     2_077.98     5_924.12       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64 (self)                                3_846.14     6_780.60    10_626.74       0.5020          1.0767            1.0727         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_655.89     2_236.89     7_892.78       0.7402          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_655.89     2_791.30     8_447.20       0.7402          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_655.89     4_070.35     9_726.24       0.7402          1.0155            1.0139         7.51
IVF-PQ-nl223-m128 (self)                               5_655.89    13_476.90    19_132.79       0.7127          1.0239            1.0198         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_407.14       703.68     4_110.81       0.3562          1.1200            1.1207         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_407.14       738.67     4_145.80       0.3562          1.1200            1.1207         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_407.14     1_045.99     4_453.13       0.3562          1.1200            1.1207         2.44
IVF-PQ-nl316-m16 (self)                                3_407.14     3_487.12     6_894.26       0.2440          1.2329            1.2393         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_630.62     1_041.60     4_672.22       0.4529          1.0793            1.0785         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_630.62     1_102.82     4_733.44       0.4529          1.0793            1.0785         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_630.62     1_578.70     5_209.32       0.4529          1.0793            1.0785         3.21
IVF-PQ-nl316-m32 (self)                                3_630.62     5_295.33     8_925.95       0.3299          1.1595            1.1607         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_467.53     1_516.25     5_983.78       0.5728          1.0449            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_467.53     1_700.15     6_167.67       0.5728          1.0449            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_467.53     2_441.13     6_908.66       0.5728          1.0449            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                4_467.53     8_085.07    12_552.60       0.4860          1.0822            1.0788         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_276.58     2_916.72     9_193.30       0.7430          1.0152            1.0136         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_276.58     3_264.50     9_541.08       0.7430          1.0152            1.0136         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_276.58     4_720.30    10_996.87       0.7430          1.0152            1.0136         7.78
IVF-PQ-nl316-m128 (self)                               6_276.58    15_658.89    21_935.47       0.7104          1.0241            1.0204         7.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
Exhaustive (query)                                        32.87       734.06       766.93       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.87     2_370.74     2_403.61       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                666.44       666.18     1_332.62       0.7118          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 666.44     2_199.00     2_865.44       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_182.95     1_538.11     2_721.06       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_182.95     5_057.80     6_240.75       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              1_874.26     3_589.19     5_463.45       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               1_874.26    11_939.13    13_813.39       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_547.67       210.08     1_757.75       0.8272          1.0522            1.0448         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_547.67       346.69     1_894.36       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_547.67       470.07     2_017.74       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16 (self)                                1_547.67     1_556.84     3_104.51       0.7669          1.0989            1.0836         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_976.10       388.19     2_364.29       0.8746          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_976.10       652.79     2_628.89       0.8752          1.0262            1.0217         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_976.10       905.57     2_881.67       0.8752          1.0262            1.0217         1.93
IVF-PQ-nl158-m32 (self)                                1_976.10     3_012.66     4_988.76       0.8288          1.0511            1.0423         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_618.50       707.18     3_325.68       0.9048          1.0151            1.0118         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_618.50     1_190.53     3_809.02       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_618.50     1_677.45     4_295.95       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64 (self)                                2_618.50     5_579.11     8_197.61       0.8704          1.0288            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_127.90       298.63     1_426.53       0.8428          1.0430            1.0365         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_127.90       369.14     1_497.04       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_127.90       544.63     1_672.53       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16 (self)                                1_127.90     1_832.54     2_960.44       0.7841          1.0842            1.0704         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_578.92       537.57     2_116.49       0.8837          1.0224            1.0183         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_578.92       673.23     2_252.14       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_578.92       998.88     2_577.80       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32 (self)                                1_578.92     3_332.69     4_911.61       0.8403          1.0440            1.0356         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_190.52       941.29     3_131.81       0.9100          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_190.52     1_199.00     3_389.52       0.9102          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_190.52     1_784.38     3_974.90       0.9102          1.0133            1.0102         3.52
IVF-PQ-nl223-m64 (self)                                2_190.52     5_956.03     8_146.55       0.8765          1.0259            1.0200         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_349.90       381.28     1_731.18       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_349.90       427.94     1_777.84       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_349.90       616.63     1_966.52       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16 (self)                                1_349.90     2_045.58     3_395.48       0.7922          1.0785            1.0637         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_798.27       681.14     2_479.40       0.8867          1.0214            1.0175         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_798.27       780.54     2_578.80       0.8867          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_798.27     1_113.47     2_911.74       0.8867          1.0214            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_798.27     3_695.98     5_494.25       0.8425          1.0433            1.0344         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_384.19     1_190.93     3_575.12       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_384.19     1_350.02     3_734.21       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_384.19     1_970.33     4_354.52       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64 (self)                                2_384.19     6_553.03     8_937.21       0.8791          1.0247            1.0188         3.61
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
Exhaustive (query)                                        67.93     1_288.44     1_356.37       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.93     4_211.81     4_279.74       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                879.03       686.44     1_565.47       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 879.03     2_224.77     3_103.80       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_277.65     1_539.05     2_816.70       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_277.65     5_067.87     6_345.52       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_210.02     3_634.11     5_844.14       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_210.02    12_080.75    14_290.78       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_565.96       281.43     2_847.38       0.8455          1.0448            1.0357         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_565.96       448.52     3_014.48       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_565.96       626.28     3_192.24       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16 (self)                                2_565.96     2_078.29     4_644.25       0.7844          1.0913            1.0648         1.57
IVF-PQ-nl158-m32-np7 (query)                           3_012.30       424.07     3_436.38       0.8726          1.0297            1.0231         2.34
IVF-PQ-nl158-m32-np12 (query)                          3_012.30       722.45     3_734.76       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32-np17 (query)                          3_012.30       975.62     3_987.92       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32 (self)                                3_012.30     3_220.61     6_232.92       0.8208          1.0615            1.0426         2.34
IVF-PQ-nl158-m64-np7 (query)                           4_101.76       785.47     4_887.23       0.8936          1.0202            1.0150         3.86
IVF-PQ-nl158-m64-np12 (query)                          4_101.76     1_320.69     5_422.45       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64-np17 (query)                          4_101.76     1_833.34     5_935.10       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64 (self)                                4_101.76     6_105.32    10_207.08       0.8494          1.0420            1.0290         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_750.90       422.99     2_173.89       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_750.90       512.51     2_263.41       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_750.90       752.76     2_503.66       0.8544          1.0397            1.0313         1.70
IVF-PQ-nl223-m16 (self)                                1_750.90     2_502.44     4_253.34       0.7965          1.0807            1.0566         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_090.81       609.89     2_700.70       0.8795          1.0266            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_090.81       756.88     2_847.69       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_090.81     1_121.43     3_212.23       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                2_090.81     3_687.16     5_777.97       0.8306          1.0550            1.0377         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_168.19     1_082.02     4_250.20       0.9002          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_168.19     1_368.73     4_536.92       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_168.19     2_034.59     5_202.78       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64 (self)                                3_168.19     6_754.04     9_922.23       0.8568          1.0378            1.0256         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_021.26       549.98     2_571.24       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_021.26       603.65     2_624.91       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_021.26       864.29     2_885.55       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16 (self)                                2_021.26     2_860.58     4_881.84       0.8149          1.0655            1.0461         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_374.51       772.10     3_146.61       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_374.51       859.86     3_234.38       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_374.51     1_273.52     3_648.03       0.8917          1.0214            1.0159         2.65
IVF-PQ-nl316-m32 (self)                                2_374.51     4_109.38     6_483.89       0.8455          1.0452            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_277.91     1_369.18     4_647.09       0.9064          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_277.91     1_557.20     4_835.12       0.9065          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_277.91     2_257.48     5_535.40       0.9065          1.0155            1.0112         4.17
IVF-PQ-nl316-m64 (self)                                3_277.91     7_478.93    10_756.85       0.8655          1.0331            1.0218         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.66     1_800.97     1_902.63       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.66     5_876.66     5_978.32       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_122.13       685.87     1_808.00       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_122.13     2_216.86     3_338.98       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                              1_585.30     1_537.89     3_123.19       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                               1_585.30     5_050.13     6_635.42       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                              2_459.18     3_571.59     6_030.77       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                               2_459.18    11_809.95    14_269.13       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             4_319.64     7_836.14    12_155.78       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              4_319.64    25_948.57    30_268.21       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_531.21       368.63     3_899.84       0.8522          1.0420            1.0322         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_531.21       586.51     4_117.73       0.8524          1.0420            1.0321         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_531.21       795.86     4_327.07       0.8524          1.0420            1.0321         1.98
IVF-PQ-nl158-m16 (self)                                3_531.21     2_636.24     6_167.46       0.7910          1.0835            1.0598         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_997.14       548.35     4_545.49       0.9000          1.0207            1.0126         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_997.14       896.36     4_893.50       0.9001          1.0206            1.0126         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_997.14     1_240.39     5_237.53       0.9001          1.0206            1.0126         2.74
IVF-PQ-nl158-m32 (self)                                3_997.14     4_117.09     8_114.23       0.8546          1.0436            1.0236         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_991.82       899.08     5_890.90       0.9202          1.0131            1.0071         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_991.82     1_547.22     6_539.04       0.9204          1.0130            1.0070         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_991.82     2_037.96     7_029.78       0.9204          1.0130            1.0070         4.27
IVF-PQ-nl158-m64 (self)                                4_991.82     6_696.75    11_688.57       0.8831          1.0284            1.0135         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_693.61     1_721.43     8_415.04       0.9393          1.0072            1.0031         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_693.61     2_873.69     9_567.29       0.9395          1.0071            1.0031         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_693.61     4_012.96    10_706.57       0.9395          1.0071            1.0031         7.32
IVF-PQ-nl158-m128 (self)                               6_693.61    13_292.52    19_986.13       0.9071          1.0171            1.0071         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_243.80       517.78     2_761.58       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_243.80       648.71     2_892.51       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_243.80       925.25     3_169.05       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16 (self)                                2_243.80     3_062.26     5_306.06       0.8061          1.0703            1.0513         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_806.69       769.34     3_576.03       0.9087          1.0172            1.0103         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_806.69       963.26     3_769.95       0.9088          1.0172            1.0103         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_806.69     1_417.47     4_224.16       0.9088          1.0172            1.0103         2.93
IVF-PQ-nl223-m32 (self)                                2_806.69     4_668.74     7_475.43       0.8678          1.0351            1.0191         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_612.22     1_205.08     4_817.30       0.9272          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_612.22     1_526.45     5_138.67       0.9272          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_612.22     2_248.50     5_860.72       0.9273          1.0111            1.0057         4.46
IVF-PQ-nl223-m64 (self)                                3_612.22     7_497.63    11_109.85       0.8925          1.0236            1.0111         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_387.40     2_364.46     7_751.86       0.9439          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_387.40     2_990.33     8_377.73       0.9440          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_387.40     4_449.79     9_837.20       0.9440          1.0059            1.0023         7.51
IVF-PQ-nl223-m128 (self)                               5_387.40    14_756.27    20_143.67       0.9133          1.0148            1.0058         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_600.28       673.04     3_273.32       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_600.28       744.81     3_345.09       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_600.28     1_052.78     3_653.06       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16 (self)                                2_600.28     3_503.86     6_104.13       0.8141          1.0644            1.0466         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_063.75       982.35     4_046.10       0.9129          1.0154            1.0094         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_063.75     1_097.54     4_161.29       0.9129          1.0154            1.0094         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_063.75     1_592.37     4_656.12       0.9129          1.0154            1.0093         3.21
IVF-PQ-nl316-m32 (self)                                3_063.75     5_267.48     8_331.24       0.8726          1.0331            1.0177         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_061.98     1_534.47     5_596.44       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_061.98     1_733.61     5_795.59       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_061.98     2_510.37     6_572.35       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64 (self)                                4_061.98     8_312.64    12_374.62       0.8964          1.0221            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_760.43     3_017.26     8_777.69       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_760.43     3_429.51     9_189.94       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_760.43     4_971.95    10_732.38       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128 (self)                               5_760.43    16_561.11    22_321.55       0.9164          1.0139            1.0052         7.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.49       691.60       725.09       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.49     2_201.75     2_235.24       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_533.19       724.53     4_257.72       0.2865          1.1528            1.1332         1.26
Exhaustive-OPQ-m16 (self)                              3_533.19     2_709.60     6_242.80       0.2585          1.1711            1.1497         1.26
Exhaustive-OPQ-m32 (query)                             5_701.10     1_571.24     7_272.35       0.3260          1.1208            1.1171         2.03
Exhaustive-OPQ-m32 (self)                              5_701.10     5_478.69    11_179.79       0.2831          1.1440            1.1382         2.03
Exhaustive-OPQ-m64 (query)                             8_810.76     3_618.01    12_428.77       0.3797          1.0983            1.0951         3.55
Exhaustive-OPQ-m64 (self)                              8_810.76    12_325.69    21_136.45       0.3219          1.1205            1.1171         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_174.04       272.46     4_446.50       0.3870          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_174.04       374.64     4_548.68       0.3870          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_174.04       484.90     4_658.94       0.3870          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16 (self)                               4_174.04     1_944.81     6_118.85       0.3184          1.1180            1.1227         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_164.38       429.05     6_593.44       0.4937          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_164.38       629.74     6_794.13       0.4937          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_164.38       831.91     6_996.29       0.4937          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32 (self)                               6_164.38     3_043.73     9_208.11       0.4170          1.0758            1.0764         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_741.81       702.06     9_443.87       0.6945          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_741.81     1_059.93     9_801.74       0.6945          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_741.81     1_433.30    10_175.11       0.6945          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64 (self)                               8_741.81     5_064.37    13_806.18       0.6382          1.0260            1.0239         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_883.85       360.25     4_244.10       0.3976          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_883.85       423.30     4_307.15       0.3976          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_883.85       586.59     4_470.44       0.3976          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16 (self)                               3_883.85     2_296.39     6_180.24       0.3209          1.1161            1.1204         1.73
IVF-OPQ-nl223-m32-np11 (query)                         5_970.89       569.31     6_540.20       0.5055          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32-np14 (query)                         5_970.89       703.56     6_674.45       0.5055          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32-np21 (query)                         5_970.89       987.82     6_958.72       0.5055          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32 (self)                               5_970.89     3_616.29     9_587.18       0.4235          1.0742            1.0736         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_722.35       972.55     9_694.90       0.7016          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_722.35     1_194.57     9_916.92       0.7017          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_722.35     1_703.26    10_425.61       0.7017          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64 (self)                               8_722.35     6_327.75    15_050.10       0.6436          1.0256            1.0229         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_172.68       444.03     4_616.71       0.4070          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_172.68       505.66     4_678.34       0.4071          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_172.68       674.90     4_847.58       0.4071          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16 (self)                               4_172.68     2_599.30     6_771.99       0.3255          1.1134            1.1175         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_127.15       740.94     6_868.09       0.5174          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_127.15       804.89     6_932.04       0.5175          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_127.15     1_134.68     7_261.83       0.5175          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32 (self)                               6_127.15     4_121.33    10_248.48       0.4332          1.0702            1.0706         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_917.00     1_254.97    10_171.97       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_917.00     1_380.37    10_297.37       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_917.00     1_959.46    10_876.46       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64 (self)                               8_917.00     6_897.93    15_814.94       0.6529          1.0237            1.0215         4.36
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
Exhaustive (query)                                        67.76     1_275.43     1_343.19       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.76     4_266.13     4_333.89       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_734.40     1_006.77     6_741.16       0.2659          1.1129            1.1008         2.26
Exhaustive-OPQ-m16 (self)                              5_734.40     4_700.86    10_435.25       0.2458          1.1245            1.1094         2.26
Exhaustive-OPQ-m32 (query)                             7_575.00     1_853.78     9_428.78       0.2865          1.0971            1.0962         3.03
Exhaustive-OPQ-m32 (self)                              7_575.00     7_434.38    15_009.37       0.2608          1.1083            1.1060         3.03
Exhaustive-OPQ-m64 (query)                            11_660.67     3_869.45    15_530.12       0.3196          1.0822            1.0844         4.55
Exhaustive-OPQ-m64 (self)                             11_660.67    14_216.59    25_877.26       0.2789          1.0967            1.0990         4.55
Exhaustive-OPQ-m128 (query)                           17_305.79     8_139.05    25_444.83       0.3687          1.0680            1.0687         7.61
Exhaustive-OPQ-m128 (self)                            17_305.79    28_451.09    45_756.88       0.3154          1.0825            1.0831         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_030.87       592.80     7_623.67       0.3241          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_030.87       742.00     7_772.87       0.3241          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_030.87       901.28     7_932.15       0.3241          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16 (self)                               7_030.87     4_426.50    11_457.37       0.2768          1.0976            1.1034         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_889.86       729.73     9_619.59       0.3685          1.0656            1.0671         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_889.86       953.00     9_842.86       0.3685          1.0656            1.0671         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_889.86     1_176.99    10_066.85       0.3685          1.0656            1.0671         3.84
IVF-OPQ-nl158-m32 (self)                               8_889.86     5_366.96    14_256.82       0.3026          1.0856            1.0895         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_946.10     1_054.74    14_000.85       0.4762          1.0414            1.0399         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_946.10     1_461.99    14_408.09       0.4762          1.0414            1.0399         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_946.10     1_884.07    14_830.17       0.4762          1.0414            1.0399         5.36
IVF-OPQ-nl158-m64 (self)                              12_946.10     7_681.69    20_627.79       0.4023          1.0545            1.0549         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_427.22     1_599.76    20_026.98       0.6818          1.0146            1.0117         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_427.22     2_302.58    20_729.80       0.6818          1.0146            1.0117         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_427.22     3_004.14    21_431.36       0.6818          1.0146            1.0117         8.42
IVF-OPQ-nl158-m128 (self)                             18_427.22    11_638.37    30_065.59       0.6259          1.0193            1.0170         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_586.86       737.00     7_323.86       0.3298          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_586.86       814.35     7_401.21       0.3299          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_586.86     1_058.12     7_644.98       0.3299          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16 (self)                               6_586.86     4_822.78    11_409.64       0.2774          1.0971            1.1019         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_629.54       925.26     9_554.80       0.3782          1.0618            1.0630         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_629.54     1_044.89     9_674.42       0.3782          1.0618            1.0630         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_629.54     1_366.78     9_996.32       0.3782          1.0618            1.0630         3.96
IVF-OPQ-nl223-m32 (self)                               8_629.54     5_938.94    14_568.48       0.3048          1.0841            1.0875         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_622.18     1_395.15    14_017.33       0.4881          1.0384            1.0372         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_622.18     1_639.76    14_261.94       0.4882          1.0384            1.0372         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_622.18     2_237.60    14_859.78       0.4882          1.0384            1.0372         5.49
IVF-OPQ-nl223-m64 (self)                              12_622.18     8_800.33    21_422.51       0.4073          1.0532            1.0530         5.49
IVF-OPQ-nl223-m128-np11 (query)                       19_679.49     2_155.67    21_835.17       0.6882          1.0134            1.0113         8.54
IVF-OPQ-nl223-m128-np14 (query)                       19_679.49     2_590.25    22_269.74       0.6883          1.0134            1.0113         8.54
IVF-OPQ-nl223-m128-np21 (query)                       19_679.49     3_585.74    23_265.23       0.6883          1.0134            1.0113         8.54
IVF-OPQ-nl223-m128 (self)                             19_679.49    13_463.88    33_143.38       0.6309          1.0187            1.0165         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_473.79       841.09     8_314.88       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_473.79       952.25     8_426.04       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_473.79     1_139.88     8_613.68       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16 (self)                               7_473.79     5_202.45    12_676.24       0.2806          1.0944            1.1002         3.88
IVF-OPQ-nl316-m32-np15 (query)                         9_542.92     1_101.41    10_644.33       0.3859          1.0585            1.0601         4.65
IVF-OPQ-nl316-m32-np17 (query)                         9_542.92     1_172.77    10_715.69       0.3859          1.0585            1.0601         4.65
IVF-OPQ-nl316-m32-np25 (query)                         9_542.92     1_569.54    11_112.46       0.3859          1.0586            1.0601         4.65
IVF-OPQ-nl316-m32 (self)                               9_542.92     6_489.09    16_032.01       0.3068          1.0823            1.0861         4.65
IVF-OPQ-nl316-m64-np15 (query)                        14_059.77     1_688.65    15_748.42       0.4960          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64-np17 (query)                        14_059.77     1_859.75    15_919.52       0.4960          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64-np25 (query)                        14_059.77     2_552.72    16_612.50       0.4960          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64 (self)                              14_059.77     9_869.82    23_929.60       0.4129          1.0514            1.0520         6.17
IVF-OPQ-nl316-m128-np15 (query)                       20_446.78     2_819.62    23_266.40       0.6972          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np17 (query)                       20_446.78     2_956.44    23_403.22       0.6972          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np25 (query)                       20_446.78     4_055.85    24_502.63       0.6972          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128 (self)                             20_446.78    14_942.17    35_388.95       0.6383          1.0176            1.0158         9.23
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
Exhaustive (query)                                       100.51     1_823.65     1_924.17       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.51     6_073.10     6_173.62       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_389.18     1_513.64    10_902.82       0.2602          1.0925            1.0831         3.76
Exhaustive-OPQ-m16 (self)                              9_389.18     8_191.92    17_581.10       0.2417          1.1021            1.0892         3.76
Exhaustive-OPQ-m32 (query)                            11_474.17     2_339.43    13_813.60       0.2792          1.0787            1.0797         4.53
Exhaustive-OPQ-m32 (self)                             11_474.17    10_974.08    22_448.25       0.2568          1.0877            1.0870         4.53
Exhaustive-OPQ-m64 (query)                            16_442.60     4_375.50    20_818.11       0.2961          1.0725            1.0758         6.05
Exhaustive-OPQ-m64 (self)                             16_442.60    17_698.89    34_141.49       0.2656          1.0825            1.0855         6.05
Exhaustive-OPQ-m128 (query)                           24_433.04     8_830.18    33_263.22       0.3311          1.0632            1.0658         9.11
Exhaustive-OPQ-m128 (self)                            24_433.04    32_100.42    56_533.46       0.2834          1.0754            1.0784         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_110.46     1_152.33    12_262.79       0.3037          1.0684            1.0736         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_110.46     1_313.90    12_424.36       0.3037          1.0684            1.0736         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_110.46     1_497.91    12_608.38       0.3037          1.0684            1.0736         4.98
IVF-OPQ-nl158-m16 (self)                              11_110.46     8_190.82    19_301.28       0.2669          1.0820            1.0877         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_326.54     1_318.68    14_645.22       0.3315          1.0606            1.0638         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_326.54     1_607.31    14_933.85       0.3315          1.0606            1.0638         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_326.54     1_900.18    15_226.72       0.3315          1.0606            1.0638         5.74
IVF-OPQ-nl158-m32 (self)                              13_326.54     9_507.36    22_833.90       0.2761          1.0776            1.0826         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_473.18     1_623.59    19_096.77       0.3898          1.0477            1.0482         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_473.18     2_090.50    19_563.68       0.3898          1.0477            1.0482         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_473.18     2_556.86    20_030.04       0.3898          1.0477            1.0482         7.27
IVF-OPQ-nl158-m64 (self)                              17_473.18    11_870.15    29_343.33       0.3204          1.0625            1.0646         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_952.20     2_411.69    28_363.89       0.5419          1.0242            1.0224        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_952.20     3_344.27    29_296.47       0.5419          1.0242            1.0224        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_952.20     4_272.30    30_224.50       0.5419          1.0242            1.0224        10.32
IVF-OPQ-nl158-m128 (self)                             25_952.20    17_457.08    43_409.27       0.4724          1.0318            1.0309        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_834.31     1_277.76    12_112.07       0.3071          1.0669            1.0710         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_834.31     1_393.07    12_227.38       0.3071          1.0669            1.0710         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_834.31     1_646.57    12_480.89       0.3071          1.0669            1.0710         5.17
IVF-OPQ-nl223-m16 (self)                              10_834.31     8_682.42    19_516.73       0.2669          1.0819            1.0877         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_096.08     1_556.19    14_652.27       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_096.08     1_729.20    14_825.29       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_096.08     2_133.13    15_229.22       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32 (self)                              13_096.08    10_307.75    23_403.83       0.2769          1.0772            1.0817         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_742.34     1_986.27    19_728.61       0.4029          1.0445            1.0446         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_742.34     2_270.35    20_012.69       0.4029          1.0445            1.0446         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_742.34     2_926.71    20_669.05       0.4029          1.0445            1.0446         7.46
IVF-OPQ-nl223-m64 (self)                              17_742.34    13_088.16    30_830.50       0.3230          1.0614            1.0633         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_235.16     3_132.92    29_368.07       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_235.16     3_679.87    29_915.02       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_235.16     5_007.25    31_242.40       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128 (self)                             26_235.16    19_845.39    46_080.55       0.4789          1.0309            1.0300        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_092.81     1_436.93    12_529.74       0.3137          1.0642            1.0691         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_092.81     1_501.63    12_594.44       0.3137          1.0642            1.0691         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_092.81     1_787.51    12_880.33       0.3137          1.0642            1.0691         6.19
IVF-OPQ-nl316-m16 (self)                              11_092.81     9_160.60    20_253.41       0.2694          1.0802            1.0862         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_575.08     1_785.00    15_360.08       0.3445          1.0559            1.0587         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_575.08     1_897.48    15_472.55       0.3446          1.0559            1.0587         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_575.08     2_359.10    15_934.18       0.3446          1.0559            1.0587         6.96
IVF-OPQ-nl316-m32 (self)                              13_575.08    11_062.99    24_638.07       0.2782          1.0756            1.0805         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_962.62     2_345.84    20_308.46       0.4111          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_962.62     2_529.69    20_492.31       0.4111          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_962.62     3_287.03    21_249.65       0.4111          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64 (self)                              17_962.62    14_190.75    32_153.37       0.3274          1.0596            1.0619         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_286.08     3_822.36    30_108.44       0.5615          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_286.08     4_188.43    30_474.50       0.5615          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_286.08     5_665.26    31_951.33       0.5615          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128 (self)                             26_286.08    22_146.26    48_432.34       0.4851          1.0296            1.0291        11.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.81       704.36       737.16       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.81     2_363.69     2_396.50       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_364.39       717.86     4_082.25       0.3009          1.2503            1.2433         1.26
Exhaustive-OPQ-m16 (self)                              3_364.39     2_701.36     6_065.75       0.2368          1.3778            1.3714         1.26
Exhaustive-OPQ-m32 (query)                             5_457.83     1_548.69     7_006.52       0.4204          1.1526            1.1474         2.03
Exhaustive-OPQ-m32 (self)                              5_457.83     5_438.65    10_896.48       0.3378          1.2478            1.2416         2.03
Exhaustive-OPQ-m64 (query)                             8_318.37     3_586.39    11_904.76       0.5662          1.0765            1.0733         3.55
Exhaustive-OPQ-m64 (self)                              8_318.37    12_308.58    20_626.95       0.4876          1.1287            1.1241         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_194.09       266.38     4_460.47       0.7004          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_194.09       370.01     4_564.10       0.7004          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_194.09       488.37     4_682.46       0.7004          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16 (self)                               4_194.09     1_981.87     6_175.96       0.6205          1.0625            1.0598         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_091.93       420.21     6_512.15       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_091.93       624.62     6_716.56       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_091.93       850.87     6_942.80       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32 (self)                               6_091.93     3_187.82     9_279.76       0.7479          1.0257            1.0237         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_775.68       694.68     9_470.36       0.8604          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_775.68     1_074.92     9_850.61       0.8604          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_775.68     1_452.98    10_228.66       0.8604          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64 (self)                               8_775.68     5_168.87    13_944.55       0.8308          1.0112            1.0097         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_927.23       351.98     4_279.22       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_927.23       421.45     4_348.68       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_927.23       592.36     4_519.59       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16 (self)                               3_927.23     2_310.60     6_237.84       0.6283          1.0597            1.0570         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_028.77       576.91     6_605.68       0.8038          1.0132            1.0120         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_028.77       705.96     6_734.74       0.8038          1.0132            1.0120         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_028.77     1_014.52     7_043.30       0.8038          1.0132            1.0120         2.50
IVF-OPQ-nl223-m32 (self)                               6_028.77     3_713.04     9_741.81       0.7548          1.0243            1.0224         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_784.48       969.24     9_753.72       0.8636          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_784.48     1_183.78     9_968.27       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_784.48     1_732.20    10_516.69       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64 (self)                               8_784.48     6_056.96    14_841.44       0.8350          1.0106            1.0092         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_207.71       439.79     4_647.50       0.7130          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_207.71       481.61     4_689.32       0.7131          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_207.71       670.35     4_878.06       0.7131          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16 (self)                               4_207.71     2_582.88     6_790.60       0.6356          1.0570            1.0544         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_211.61       736.98     6_948.59       0.8078          1.0127            1.0114         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_211.61       819.39     7_031.00       0.8079          1.0126            1.0114         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_211.61     1_159.69     7_371.30       0.8079          1.0126            1.0114         2.84
IVF-OPQ-nl316-m32 (self)                               6_211.61     4_224.06    10_435.66       0.7590          1.0235            1.0217         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_993.95     1_230.70    10_224.65       0.8657          1.0060            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_993.95     1_371.66    10_365.61       0.8658          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_993.95     1_959.85    10_953.80       0.8658          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               8_993.95     6_863.18    15_857.13       0.8369          1.0103            1.0090         4.36
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
Exhaustive (query)                                        68.00     1_273.83     1_341.84       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.00     4_233.50     4_301.50       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_257.00     1_017.45     7_274.46       0.2317          1.2142            1.2107         2.26
Exhaustive-OPQ-m16 (self)                              6_257.00     4_761.13    11_018.14       0.1879          1.2983            1.2985         2.26
Exhaustive-OPQ-m32 (query)                             8_091.86     1_845.78     9_937.64       0.3189          1.1505            1.1472         3.03
Exhaustive-OPQ-m32 (self)                              8_091.86     7_479.23    15_571.09       0.2588          1.2171            1.2145         3.03
Exhaustive-OPQ-m64 (query)                            12_855.29     3_877.38    16_732.67       0.4332          1.0939            1.0912         4.55
Exhaustive-OPQ-m64 (self)                             12_855.29    14_189.66    27_044.94       0.3620          1.1417            1.1393         4.55
Exhaustive-OPQ-m128 (query)                           17_207.90     8_128.35    25_336.25       0.5699          1.0489            1.0476         7.61
Exhaustive-OPQ-m128 (self)                            17_207.90    28_476.33    45_684.23       0.4998          1.0773            1.0755         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_984.42       592.77     7_577.19       0.5392          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_984.42       726.73     7_711.14       0.5392          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_984.42       881.87     7_866.28       0.5392          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16 (self)                               6_984.42     4_378.44    11_362.86       0.4372          1.1019            1.0999         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_781.76       719.61     9_501.36       0.6839          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_781.76       938.28     9_720.04       0.6839          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_781.76     1_170.80     9_952.55       0.6839          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32 (self)                               8_781.76     5_303.43    14_085.18       0.6092          1.0440            1.0417         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_925.30     1_040.31    13_965.62       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_925.30     1_450.67    14_375.97       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_925.30     1_859.83    14_785.14       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64 (self)                              12_925.30     7_624.57    20_549.87       0.7293          1.0202            1.0180         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_367.04     1_568.90    19_935.94       0.8371          1.0063            1.0052         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_367.04     2_258.76    20_625.80       0.8371          1.0063            1.0052         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_367.04     2_957.51    21_324.55       0.8371          1.0063            1.0052         8.42
IVF-OPQ-nl158-m128 (self)                             18_367.04    11_259.15    29_626.19       0.8087          1.0101            1.0081         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_776.46       709.63     7_486.09       0.5465          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_776.46       790.31     7_566.77       0.5465          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_776.46       996.70     7_773.17       0.5465          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16 (self)                               6_776.46     4_722.89    11_499.35       0.4480          1.0973            1.0954         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_502.84       902.03     9_404.87       0.6914          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_502.84     1_030.49     9_533.33       0.6914          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_502.84     1_352.71     9_855.55       0.6914          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32 (self)                               8_502.84     5_887.92    14_390.75       0.6190          1.0414            1.0395         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_731.00     1_375.16    14_106.16       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_731.00     1_616.96    14_347.96       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_731.00     2_213.61    14_944.61       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64 (self)                              12_731.00     8_808.37    21_539.37       0.7361          1.0190            1.0173         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_668.43     2_279.62    20_948.05       0.8410          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_668.43     2_556.31    21_224.74       0.8410          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_668.43     3_533.51    22_201.94       0.8410          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128 (self)                             18_668.43    13_244.50    31_912.93       0.8125          1.0095            1.0079         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_034.60       818.30     7_852.90       0.5527          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_034.60       880.67     7_915.27       0.5527          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_034.60     1_110.48     8_145.08       0.5527          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16 (self)                               7_034.60     5_071.06    12_105.66       0.4526          1.0952            1.0936         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_990.48     1_074.72    10_065.19       0.6937          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_990.48     1_161.81    10_152.28       0.6937          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_990.48     1_515.58    10_506.05       0.6937          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32 (self)                               8_990.48     6_434.77    15_425.25       0.6228          1.0405            1.0387         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_132.23     1_677.18    14_809.40       0.7859          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_132.23     1_842.90    14_975.13       0.7859          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_132.23     2_504.97    15_637.20       0.7859          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64 (self)                              13_132.23     9_804.38    22_936.60       0.7390          1.0184            1.0168         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_751.94     2_626.40    21_378.34       0.8423          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_751.94     2_900.73    21_652.67       0.8423          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_751.94     4_054.30    22_806.24       0.8423          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128 (self)                             18_751.94    15_230.77    33_982.71       0.8144          1.0092            1.0078         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.30     1_787.57     1_886.87       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.30     6_169.30     6_268.60       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_367.07     1_503.90    10_870.97       0.2295          1.2024            1.1985         3.76
Exhaustive-OPQ-m16 (self)                              9_367.07     8_169.33    17_536.40       0.1868          1.2974            1.2971         3.76
Exhaustive-OPQ-m32 (query)                            11_479.56     2_355.40    13_834.96       0.3123          1.1452            1.1413         4.53
Exhaustive-OPQ-m32 (self)                             11_479.56    10_958.18    22_437.74       0.2574          1.2173            1.2150         4.53
Exhaustive-OPQ-m64 (query)                            16_080.86     4_389.33    20_470.19       0.4084          1.0974            1.0946         6.05
Exhaustive-OPQ-m64 (self)                             16_080.86    17_724.53    33_805.39       0.3479          1.1498            1.1473         6.05
Exhaustive-OPQ-m128 (query)                           24_547.64     8_683.39    33_231.03       0.5283          1.0567            1.0548         9.11
Exhaustive-OPQ-m128 (self)                            24_547.64    32_162.80    56_710.44       0.4662          1.0907            1.0887         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_149.49     1_143.90    12_293.39       0.5318          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_149.49     1_310.63    12_460.12       0.5318          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_149.49     1_493.05    12_642.54       0.5318          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16 (self)                              11_149.49     8_172.42    19_321.91       0.4302          1.1043            1.1024         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_354.38     1_310.16    14_664.55       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_354.38     1_603.44    14_957.82       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_354.38     1_890.72    15_245.10       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32 (self)                              13_354.38     9_486.99    22_841.38       0.6038          1.0450            1.0428         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_551.17     1_624.01    19_175.18       0.7740          1.0113            1.0102         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_551.17     2_094.01    19_645.18       0.7740          1.0113            1.0102         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_551.17     2_557.28    20_108.45       0.7740          1.0113            1.0102         7.27
IVF-OPQ-nl158-m64 (self)                              17_551.17    12_551.45    30_102.62       0.7253          1.0207            1.0184         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_996.20     2_388.66    28_384.86       0.8325          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_996.20     3_349.93    29_346.13       0.8325          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_996.20     4_258.71    30_254.91       0.8325          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128 (self)                             25_996.20    17_409.86    43_406.06       0.8042          1.0105            1.0084        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_896.59     1_291.42    12_188.01       0.5394          1.0523            1.0512         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_896.59     1_388.90    12_285.50       0.5394          1.0523            1.0512         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_896.59     1_638.78    12_535.37       0.5394          1.0523            1.0512         5.17
IVF-OPQ-nl223-m16 (self)                              10_896.59     8_673.91    19_570.51       0.4397          1.1000            1.0983         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_219.55     1_554.26    14_773.80       0.6831          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_219.55     1_721.93    14_941.48       0.6831          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_219.55     2_157.11    15_376.66       0.6831          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32 (self)                              13_219.55    10_305.64    23_525.19       0.6094          1.0435            1.0415         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_549.42     1_998.15    19_547.58       0.7788          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_549.42     2_265.01    19_814.43       0.7788          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_549.42     2_945.52    20_494.94       0.7788          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64 (self)                              17_549.42    12_984.48    30_533.90       0.7310          1.0196            1.0178         7.46
IVF-OPQ-nl223-m128-np11 (query)                       25_969.82     3_115.91    29_085.73       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np14 (query)                       25_969.82     3_680.86    29_650.67       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np21 (query)                       25_969.82     5_026.71    30_996.52       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128 (self)                             25_969.82    19_872.58    45_842.40       0.8077          1.0100            1.0083        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_624.13     1_432.80    13_056.93       0.5459          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_624.13     1_516.18    13_140.31       0.5459          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_624.13     1_793.00    13_417.13       0.5459          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16 (self)                              11_624.13     9_262.99    20_887.12       0.4462          1.0971            1.0955         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_670.87     1_779.61    15_450.48       0.6866          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_670.87     1_894.23    15_565.10       0.6866          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_670.87     2_359.85    16_030.71       0.6866          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32 (self)                              13_670.87    11_060.61    24_731.48       0.6125          1.0427            1.0408         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_033.38     2_349.70    20_383.08       0.7793          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_033.38     2_540.88    20_574.26       0.7793          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_033.38     3_298.64    21_332.02       0.7793          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64 (self)                              18_033.38    14_184.58    32_217.96       0.7332          1.0192            1.0176         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_744.86     3_840.10    30_584.96       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_744.86     4_186.87    30_931.73       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_744.86     5_668.83    32_413.69       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128 (self)                             26_744.86    22_426.47    49_171.33       0.8089          1.0097            1.0082        11.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Cell embeddings

Lastly, also here the synthetic data that resembles the embeddings generated by
single cell models.

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.47       711.09       743.56       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.47     2_351.51     2_383.98       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_347.74       716.02     4_063.76       0.7911          1.0819            1.0684         1.26
Exhaustive-OPQ-m16 (self)                              3_347.74     2_695.31     6_043.05       0.7232          1.1502            1.1255         1.26
Exhaustive-OPQ-m32 (query)                             5_430.51     1_560.21     6_990.72       0.8303          1.0536            1.0424         2.03
Exhaustive-OPQ-m32 (self)                              5_430.51     5_527.57    10_958.08       0.7763          1.0975            1.0767         2.03
Exhaustive-OPQ-m64 (query)                             8_209.82     3_579.46    11_789.28       0.8562          1.0398            1.0292         3.55
Exhaustive-OPQ-m64 (self)                              8_209.82    12_257.34    20_467.16       0.8092          1.0723            1.0534         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_158.66       279.06     4_437.72       0.8895          1.0208            1.0163         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_158.66       415.34     4_574.00       0.8903          1.0203            1.0161         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_158.66       564.37     4_723.03       0.8903          1.0203            1.0161         1.67
IVF-OPQ-nl158-m16 (self)                               4_158.66     2_186.72     6_345.38       0.8475          1.0405            1.0325         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_180.35       457.01     6_637.35       0.9110          1.0133            1.0098         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_180.35       717.90     6_898.24       0.9118          1.0129            1.0096         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_180.35       980.24     7_160.59       0.9118          1.0128            1.0096         2.43
IVF-OPQ-nl158-m32 (self)                               6_180.35     3_622.51     9_802.86       0.8774          1.0255            1.0196         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_873.14       776.81     9_649.95       0.9240          1.0098            1.0066         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_873.14     1_263.04    10_136.18       0.9248          1.0094            1.0064         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_873.14     1_751.55    10_624.69       0.9248          1.0094            1.0064         3.96
IVF-OPQ-nl158-m64 (self)                               8_873.14     6_180.87    15_054.01       0.8961          1.0185            1.0133         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_845.99       388.15     4_234.14       0.8979          1.0178            1.0138         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_845.99       446.43     4_292.42       0.8980          1.0177            1.0138         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_845.99       628.26     4_474.25       0.8980          1.0177            1.0138         1.73
IVF-OPQ-nl223-m16 (self)                               3_845.99     2_441.97     6_287.96       0.8581          1.0352            1.0271         1.73
IVF-OPQ-nl223-m32-np11 (query)                         5_891.61       603.78     6_495.39       0.9152          1.0118            1.0088         2.50
IVF-OPQ-nl223-m32-np14 (query)                         5_891.61       748.04     6_639.66       0.9153          1.0117            1.0087         2.50
IVF-OPQ-nl223-m32-np21 (query)                         5_891.61     1_069.35     6_960.96       0.9153          1.0117            1.0087         2.50
IVF-OPQ-nl223-m32 (self)                               5_891.61     3_917.91     9_809.52       0.8827          1.0234            1.0176         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_651.81     1_027.28     9_679.09       0.9269          1.0092            1.0060         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_651.81     1_276.58     9_928.39       0.9271          1.0091            1.0060         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_651.81     1_863.86    10_515.67       0.9271          1.0091            1.0060         4.02
IVF-OPQ-nl223-m64 (self)                               8_651.81     6_535.87    15_187.68       0.8987          1.0178            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_082.26       446.30     4_528.56       0.9021          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_082.26       489.62     4_571.88       0.9022          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_082.26       682.28     4_764.54       0.9022          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16 (self)                               4_082.26     2_638.45     6_720.71       0.8629          1.0334            1.0247         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_142.15       771.66     6_913.81       0.9172          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_142.15       842.93     6_985.08       0.9173          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_142.15     1_208.30     7_350.45       0.9174          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32 (self)                               6_142.15     4_355.46    10_497.61       0.8839          1.0230            1.0170         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_914.54     1_279.42    10_193.96       0.9281          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_914.54     1_425.45    10_339.99       0.9281          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_914.54     2_055.78    10_970.32       0.9282          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64 (self)                               8_914.54     7_187.78    16_102.32       0.9005          1.0171            1.0118         4.36
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
Exhaustive (query)                                        70.27     1_283.22     1_353.49       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.27     4_223.97     4_294.25       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_635.41     1_019.30     6_654.71       0.7546          1.1136            1.0983         2.26
Exhaustive-OPQ-m16 (self)                              5_635.41     4_712.54    10_347.95       0.6788          1.2037            1.1739         2.26
Exhaustive-OPQ-m32 (query)                             7_454.26     1_844.17     9_298.44       0.8064          1.0692            1.0572         3.03
Exhaustive-OPQ-m32 (self)                              7_454.26     7_435.14    14_889.40       0.7455          1.1245            1.1019         3.03
Exhaustive-OPQ-m64 (query)                            11_701.31     3_869.26    15_570.58       0.8413          1.0455            1.0364         4.55
Exhaustive-OPQ-m64 (self)                             11_701.31    14_191.12    25_892.43       0.7916          1.0819            1.0654         4.55
Exhaustive-OPQ-m128 (query)                           17_442.39     8_183.82    25_626.20       0.9198          1.0107            1.0069         7.61
Exhaustive-OPQ-m128 (self)                            17_442.39    28_572.78    46_015.17       0.8933          1.0192            1.0139         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_519.03       619.73     8_138.76       0.8909          1.0219            1.0164         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_519.03       787.99     8_307.02       0.8913          1.0217            1.0163         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_519.03       954.81     8_473.84       0.8913          1.0217            1.0163         3.07
IVF-OPQ-nl158-m16 (self)                               7_519.03     4_572.10    12_091.13       0.8464          1.0446            1.0310         3.07
IVF-OPQ-nl158-m32-np7 (query)                          9_637.88       771.13    10_409.02       0.9021          1.0171            1.0125         3.84
IVF-OPQ-nl158-m32-np12 (query)                         9_637.88     1_102.09    10_739.97       0.9026          1.0169            1.0124         3.84
IVF-OPQ-nl158-m32-np17 (query)                         9_637.88     1_302.60    10_940.48       0.9026          1.0169            1.0124         3.84
IVF-OPQ-nl158-m32 (self)                               9_637.88     5_753.94    15_391.82       0.8629          1.0344            1.0239         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_175.66     1_136.74    14_312.40       0.9096          1.0145            1.0102         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_175.66     1_650.56    14_826.22       0.9102          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_175.66     2_175.58    15_351.24       0.9102          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64 (self)                              13_175.66     8_605.62    21_781.28       0.8749          1.0286            1.0197         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_263.70     1_742.96    21_006.66       0.9631          1.0027            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_263.70     2_692.91    21_956.60       0.9638          1.0025            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_263.70     3_667.46    22_931.16       0.9639          1.0025            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                             19_263.70    13_542.72    32_806.42       0.9454          1.0056            1.0013         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_633.24       719.02     7_352.25       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_633.24       808.79     7_442.03       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_633.24     1_035.98     7_669.22       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16 (self)                               6_633.24     5_282.61    11_915.85       0.8555          1.0393            1.0269         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_498.22       932.96     9_431.18       0.9075          1.0154            1.0109         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_498.22     1_075.01     9_573.23       0.9075          1.0154            1.0109         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_498.22     1_444.49     9_942.71       0.9076          1.0154            1.0108         3.96
IVF-OPQ-nl223-m32 (self)                               8_498.22     6_140.68    14_638.90       0.8701          1.0310            1.0212         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_610.77     1_427.25    14_038.02       0.9168          1.0124            1.0083         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_610.77     1_710.38    14_321.14       0.9169          1.0123            1.0083         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_610.77     2_382.11    14_992.88       0.9169          1.0123            1.0083         5.49
IVF-OPQ-nl223-m64 (self)                              12_610.77     9_436.87    22_047.64       0.8824          1.0249            1.0169         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_102.00     2_237.26    20_339.26       0.9660          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_102.00     2_743.84    20_845.84       0.9661          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_102.00     3_915.40    22_017.40       0.9661          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                             18_102.00    14_450.91    32_552.91       0.9491          1.0049            1.0008         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_834.13       837.30     7_671.42       0.9054          1.0162            1.0116         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_834.13       902.18     7_736.31       0.9053          1.0162            1.0116         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_834.13     1_149.19     7_983.32       0.9054          1.0161            1.0116         3.88
IVF-OPQ-nl316-m16 (self)                               6_834.13     5_198.80    12_032.93       0.8661          1.0330            1.0225         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_880.94     1_095.20     9_976.14       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_880.94     1_188.79    10_069.73       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_880.94     1_566.70    10_447.64       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32 (self)                               8_880.94     6_658.96    15_539.90       0.8786          1.0267            1.0180         4.65
IVF-OPQ-nl316-m64-np15 (query)                        12_876.27     1_718.32    14_594.59       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64-np17 (query)                        12_876.27     1_910.43    14_786.70       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64-np25 (query)                        12_876.27     2_615.21    15_491.48       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64 (self)                              12_876.27    10_143.87    23_020.14       0.8884          1.0222            1.0150         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_439.97     2_908.37    21_348.34       0.9690          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_439.97     3_047.23    21_487.20       0.9691          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_439.97     4_287.51    22_727.49       0.9691          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                             18_439.97    15_692.38    34_132.35       0.9520          1.0044            1.0004         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.10     1_824.94     1_926.04       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.10     5_975.86     6_076.95       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_328.71     1_512.18    10_840.89       0.7383          1.1306            1.1121         3.76
Exhaustive-OPQ-m16 (self)                              9_328.71     8_163.52    17_492.24       0.6595          1.2295            1.1962         3.76
Exhaustive-OPQ-m32 (query)                            11_456.56     2_336.60    13_793.16       0.8493          1.0411            1.0321         4.53
Exhaustive-OPQ-m32 (self)                             11_456.56    11_131.97    22_588.53       0.8006          1.0714            1.0580         4.53
Exhaustive-OPQ-m64 (query)                            15_952.28     4_388.61    20_340.89       0.8796          1.0255            1.0186         6.05
Exhaustive-OPQ-m64 (self)                             15_952.28    17_840.72    33_793.00       0.8413          1.0441            1.0341         6.05
Exhaustive-OPQ-m128 (query)                           24_558.72     8_653.27    33_211.99       0.9051          1.0148            1.0104         9.11
Exhaustive-OPQ-m128 (self)                            24_558.72    32_139.37    56_698.09       0.8741          1.0264            1.0201         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_521.69     1_162.25    12_683.94       0.8920          1.0223            1.0161         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_521.69     1_380.49    12_902.18       0.8922          1.0222            1.0161         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_521.69     1_579.19    13_100.87       0.8922          1.0222            1.0161         4.98
IVF-OPQ-nl158-m16 (self)                              11_521.69     8_426.20    19_947.89       0.8473          1.0438            1.0304         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_899.31     1_358.77    15_258.08       0.9369          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_899.31     1_694.69    15_594.00       0.9371          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_899.31     2_030.47    15_929.79       0.9371          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32 (self)                              13_899.31     9_968.45    23_867.76       0.9074          1.0183            1.0075         5.74
IVF-OPQ-nl158-m64-np7 (query)                         18_196.06     1_724.21    19_920.26       0.9509          1.0052            1.0013         7.27
IVF-OPQ-nl158-m64-np12 (query)                        18_196.06     2_290.23    20_486.29       0.9512          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64-np17 (query)                        18_196.06     2_862.53    21_058.59       0.9512          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64 (self)                              18_196.06    12_732.80    30_928.86       0.9272          1.0115            1.0036         7.27
IVF-OPQ-nl158-m128-np7 (query)                        26_468.83     2_576.39    29_045.23       0.9607          1.0034            1.0000        10.32
IVF-OPQ-nl158-m128-np12 (query)                       26_468.83     3_758.26    30_227.09       0.9610          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128-np17 (query)                       26_468.83     4_956.16    31_424.99       0.9610          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                             26_468.83    20_762.37    47_231.20       0.9396          1.0077            1.0017        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_653.55     1_327.76    11_981.30       0.9000          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_653.55     1_429.18    12_082.73       0.9001          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_653.55     1_701.24    12_354.78       0.9001          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16 (self)                              10_653.55     8_847.90    19_501.44       0.8592          1.0363            1.0252         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_830.62     1_581.96    14_412.59       0.9422          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_830.62     1_769.94    14_600.57       0.9423          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_830.62     2_214.69    15_045.32       0.9423          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32 (self)                              12_830.62    10_562.69    23_393.31       0.9156          1.0148            1.0059         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_479.60     2_056.59    19_536.19       0.9544          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_479.60     2_368.89    19_848.49       0.9545          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_479.60     3_117.67    20_597.28       0.9545          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64 (self)                              17_479.60    13_605.57    31_085.17       0.9330          1.0097            1.0027         7.46
IVF-OPQ-nl223-m128-np11 (query)                       25_790.04     3_241.26    29_031.30       0.9640          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                       25_790.04     3_893.73    29_683.77       0.9641          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                       25_790.04     5_375.73    31_165.77       0.9641          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                             25_790.04    21_107.41    46_897.45       0.9433          1.0068            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_910.57     1_443.07    12_353.64       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_910.57     1_514.30    12_424.87       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_910.57     1_818.62    12_729.19       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16 (self)                              10_910.57     9_353.73    20_264.30       0.8640          1.0338            1.0232         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_625.91     1_817.81    15_443.73       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_625.91     1_936.60    15_562.51       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_625.91     2_422.01    16_047.92       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32 (self)                              13_625.91    11_229.70    24_855.61       0.9197          1.0135            1.0051         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_435.59     2_423.87    20_859.46       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_435.59     2_602.16    21_037.74       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_435.59     3_419.51    21_855.09       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64 (self)                              18_435.59    14_631.90    33_067.48       0.9353          1.0090            1.0024         8.48
IVF-OPQ-nl316-m128-np15 (query)                       27_642.09     3_967.04    31_609.13       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np17 (query)                       27_642.09     4_357.22    31_999.31       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np25 (query)                       27_642.09     6_010.70    33_652.79       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128 (self)                             27_642.09    23_183.58    50_825.67       0.9460          1.0062            1.0009        11.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.79     1_282.01     1_353.81       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_897.68       127.39     2_025.07       0.3442          1.0764            1.0758         2.24
IVFPQ-m64-nl111-np1                                    2_777.70       225.69     3_003.39       0.4480          1.0516            1.0446         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_993.87       138.56     2_132.43       0.3186          1.3218            1.0785         4.72
IVFPQ-m32-nl111-np2                                    1_897.68       176.17     2_073.85       0.3455          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np2                                    2_777.70       319.48     3_097.18       0.4505          1.0508            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_993.87       198.39     2_192.27       0.3451          1.0766            1.0756         4.72
IVFPQ-m32-nl111-np4                                    1_897.68       264.88     2_162.56       0.3455          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np4                                    2_777.70       488.61     3_266.31       0.4506          1.0508            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_993.87       281.01     2_274.88       0.3455          1.0760            1.0755         4.72
IVFPQ-m32-nl111-np5                                    1_897.68       311.22     2_208.90       0.3455          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np5                                    2_777.70       572.55     3_350.25       0.4506          1.0508            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_993.87       340.65     2_334.52       0.3455          1.0759            1.0755         4.72
IVFPQ-m32-nl111-np8                                    1_897.68       447.11     2_344.79       0.3455          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np8                                    2_777.70       841.56     3_619.27       0.4506          1.0508            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_993.87       474.24     2_468.12       0.3455          1.0759            1.0755         4.72
IVFPQ-m32-nl111-np10                                   1_897.68       538.08     2_435.76       0.3455          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np10                                   2_777.70     1_003.17     3_780.87       0.4506          1.0508            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_993.87       571.37     2_565.25       0.3455          1.0759            1.0755         4.72
IVFPQ-m32-nl158-np1                                    2_914.02       123.72     3_037.74       0.3498          1.0728            1.0732         2.34
IVFPQ-m64-nl158-np1                                    3_804.35       216.90     4_021.24       0.4544          1.0480            1.0436         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_072.95       132.44     3_205.39       0.3116          1.2089            1.0777         4.82
IVFPQ-m32-nl158-np2                                    2_914.02       175.55     3_089.58       0.3541          1.0713            1.0722         2.34
IVFPQ-m64-nl158-np2                                    3_804.35       310.21     4_114.55       0.4622          1.0459            1.0424         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_072.95       180.69     3_253.65       0.3535          1.0726            1.0723         4.82
IVFPQ-m32-nl158-np4                                    2_914.02       264.12     3_178.15       0.3543          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np4                                    3_804.35       476.38     4_280.72       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_072.95       276.65     3_349.61       0.3543          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np7                                    2_914.02       396.82     3_310.84       0.3543          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np7                                    3_804.35       731.50     4_535.85       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_072.95       410.12     3_483.07       0.3543          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np8                                    2_914.02       452.07     3_366.09       0.3543          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np8                                    3_804.35       810.25     4_614.60       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_072.95       459.68     3_532.64       0.3543          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np12                                   2_914.02       630.32     3_544.34       0.3543          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np12                                   3_804.35     1_144.19     4_948.54       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_072.95       642.25     3_715.21       0.3543          1.0712            1.0721         4.82
IVFPQ-m32-nl223-np1                                    2_243.70       102.10     2_345.80       0.3514          1.0705            1.0703         2.46
IVFPQ-m64-nl223-np1                                    3_190.19       161.04     3_351.23       0.4368          1.0504            1.0449         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_529.71       112.84     2_642.56       0.3276          1.1792            1.0736         4.95
IVFPQ-m32-nl223-np2                                    2_243.70       155.08     2_398.78       0.3648          1.0664            1.0669         2.46
IVFPQ-m64-nl223-np2                                    3_190.19       258.71     3_448.91       0.4660          1.0446            1.0405         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_529.71       169.20     2_698.91       0.3616          1.0687            1.0687         4.95
IVFPQ-m32-nl223-np4                                    2_243.70       254.11     2_497.81       0.3682          1.0656            1.0658         2.46
IVFPQ-m64-nl223-np4                                    3_190.19       455.06     3_645.25       0.4756          1.0431            1.0389         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_529.71       267.59     2_797.30       0.3661          1.0670            1.0666         4.95
IVFPQ-m32-nl223-np8                                    2_243.70       446.00     2_689.70       0.3685          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np8                                    3_190.19       800.44     3_990.63       0.4769          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_529.71       471.61     3_001.33       0.3683          1.0658            1.0657         4.95
IVFPQ-m32-nl223-np11                                   2_243.70       590.07     2_833.77       0.3685          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np11                                   3_190.19     1_059.87     4_250.06       0.4769          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_529.71       607.29     3_137.01       0.3685          1.0656            1.0657         4.95
IVFPQ-m32-nl223-np14                                   2_243.70       745.24     2_988.94       0.3686          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np14                                   3_190.19     1_320.49     4_510.68       0.4769          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_529.71       755.03     3_284.74       0.3686          1.0656            1.0657         4.95
IVFPQ-m32-nl316-np1                                    2_602.91       101.66     2_704.58       0.3521          1.0693            1.0683         2.65
IVFPQ-m64-nl316-np1                                    3_525.54       150.55     3_676.08       0.4284          1.0513            1.0458         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_823.43       112.44     2_935.87       0.3251          1.1522            1.0724         5.13
IVFPQ-m32-nl316-np2                                    2_602.91       152.07     2_754.98       0.3722          1.0628            1.0638         2.65
IVFPQ-m64-nl316-np2                                    3_525.54       243.58     3_769.12       0.4688          1.0426            1.0392         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_823.43       168.03     2_991.46       0.3687          1.0654            1.0658         5.13
IVFPQ-m32-nl316-np4                                    2_602.91       270.78     2_873.70       0.3782          1.0614            1.0622         2.65
IVFPQ-m64-nl316-np4                                    3_525.54       434.42     3_959.96       0.4846          1.0401            1.0368         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_823.43       267.43     3_090.86       0.3747          1.0631            1.0635         5.13
IVFPQ-m32-nl316-np8                                    2_602.91       459.39     3_062.30       0.3790          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np8                                    3_525.54       789.66     4_315.19       0.4880          1.0396            1.0363         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_823.43       469.03     3_292.46       0.3782          1.0617            1.0622         5.13
IVFPQ-m32-nl316-np15                                   2_602.91       799.21     3_402.12       0.3790          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np15                                   3_525.54     1_396.52     4_922.05       0.4883          1.0396            1.0363         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_823.43       816.39     3_639.82       0.3790          1.0612            1.0620         5.13
IVFPQ-m32-nl316-np17                                   2_602.91       894.69     3_497.60       0.3790          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np17                                   3_525.54     1_636.48     5_162.02       0.4883          1.0396            1.0363         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_823.43       904.25     3_727.68       0.3790          1.0612            1.0620         5.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.79     1_282.01     1_353.81       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        3_235.15       151.81     3_386.96       0.3123          1.2056            1.0776         4.82
SOARPQ-near-np2                                        3_235.15       185.47     3_420.62       0.3536          1.0724            1.0723         4.82
SOARPQ-near-np4                                        3_235.15       293.48     3_528.63       0.3543          1.0712            1.0721         4.82
SOARPQ-near-np7                                        3_235.15       422.18     3_657.33       0.3543          1.0712            1.0721         4.82
SOARPQ-near-np8                                        3_235.15       478.55     3_713.70       0.3543          1.0712            1.0721         4.82
SOARPQ-near-np12                                       3_235.15       669.63     3_904.78       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.3-np1                                    3_167.81       132.97     3_300.78       0.3117          1.2080            1.0777         4.82
SOARPQ-shift0.3-np2                                    3_167.81       180.48     3_348.29       0.3535          1.0725            1.0723         4.82
SOARPQ-shift0.3-np4                                    3_167.81       272.30     3_440.11       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.3-np7                                    3_167.81       418.84     3_586.65       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.3-np8                                    3_167.81       490.55     3_658.36       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.3-np12                                   3_167.81       662.45     3_830.26       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.7-np1                                    3_158.49       135.10     3_293.59       0.3115          1.2102            1.0777         4.82
SOARPQ-shift0.7-np2                                    3_158.49       180.24     3_338.73       0.3535          1.0726            1.0723         4.82
SOARPQ-shift0.7-np4                                    3_158.49       295.16     3_453.65       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.7-np7                                    3_158.49       420.57     3_579.06       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.7-np8                                    3_158.49       473.40     3_631.88       0.3543          1.0712            1.0721         4.82
SOARPQ-shift0.7-np12                                   3_158.49       672.18     3_830.67       0.3543          1.0712            1.0721         4.82
SOARPQ-orth1-np1                                       3_178.11       137.73     3_315.84       0.3128          1.2071            1.0776         4.82
SOARPQ-orth1-np2                                       3_178.11       180.08     3_358.19       0.3536          1.0722            1.0723         4.82
SOARPQ-orth1-np4                                       3_178.11       282.59     3_460.70       0.3543          1.0712            1.0721         4.82
SOARPQ-orth1-np7                                       3_178.11       423.67     3_601.77       0.3543          1.0712            1.0721         4.82
SOARPQ-orth1-np8                                       3_178.11       478.44     3_656.54       0.3543          1.0712            1.0721         4.82
SOARPQ-orth1-np12                                      3_178.11       665.99     3_844.09       0.3543          1.0712            1.0721         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.84     1_359.77     1_429.60       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_938.00       121.88     2_059.87       0.4771          1.0785            1.0741         2.24
IVFPQ-m64-nl111-np1                                    2_849.38       217.19     3_066.57       0.6137          1.0403            1.0355         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_086.42       128.59     2_215.00       0.4636          1.1011            1.0768         4.72
IVFPQ-m32-nl111-np2                                    1_938.00       171.79     2_109.78       0.4826          1.0763            1.0733         2.24
IVFPQ-m64-nl111-np2                                    2_849.38       303.00     3_152.39       0.6232          1.0370            1.0349         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_086.42       184.73     2_271.15       0.4816          1.0779            1.0736         4.72
IVFPQ-m32-nl111-np4                                    1_938.00       275.62     2_213.62       0.4828          1.0762            1.0733         2.24
IVFPQ-m64-nl111-np4                                    2_849.38       479.83     3_329.21       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_086.42       284.05     2_370.47       0.4827          1.0762            1.0733         4.72
IVFPQ-m32-nl111-np5                                    1_938.00       320.70     2_258.70       0.4828          1.0762            1.0733         2.24
IVFPQ-m64-nl111-np5                                    2_849.38       568.20     3_417.59       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_086.42       337.33     2_423.75       0.4828          1.0762            1.0733         4.72
IVFPQ-m32-nl111-np8                                    1_938.00       461.08     2_399.08       0.4828          1.0762            1.0733         2.24
IVFPQ-m64-nl111-np8                                    2_849.38       835.22     3_684.61       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_086.42       528.45     2_614.86       0.4828          1.0762            1.0733         4.72
IVFPQ-m32-nl111-np10                                   1_938.00       561.15     2_499.15       0.4828          1.0762            1.0733         2.24
IVFPQ-m64-nl111-np10                                   2_849.38     1_027.49     3_876.87       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_086.42       639.47     2_725.89       0.4828          1.0762            1.0733         4.72
IVFPQ-m32-nl158-np1                                    2_894.83       125.10     3_019.93       0.4825          1.0759            1.0725         2.34
IVFPQ-m64-nl158-np1                                    3_873.31       221.75     4_095.06       0.6179          1.0396            1.0346         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_114.88       131.48     3_246.36       0.4825          1.0800            1.0735         4.82
IVFPQ-m32-nl158-np2                                    2_894.83       169.42     3_064.25       0.4897          1.0730            1.0714         2.34
IVFPQ-m64-nl158-np2                                    3_873.31       309.71     4_183.03       0.6297          1.0355            1.0337         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_114.88       185.06     3_299.94       0.4894          1.0738            1.0717         4.82
IVFPQ-m32-nl158-np4                                    2_894.83       263.29     3_158.12       0.4902          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np4                                    3_873.31       484.56     4_357.87       0.6308          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_114.88       278.25     3_393.13       0.4901          1.0729            1.0713         4.82
IVFPQ-m32-nl158-np7                                    2_894.83       412.94     3_307.77       0.4902          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np7                                    3_873.31       732.68     4_606.00       0.6308          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_114.88       421.09     3_535.97       0.4902          1.0728            1.0712         4.82
IVFPQ-m32-nl158-np8                                    2_894.83       457.90     3_352.73       0.4902          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np8                                    3_873.31       824.12     4_697.43       0.6308          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_114.88       480.96     3_595.84       0.4902          1.0728            1.0712         4.82
IVFPQ-m32-nl158-np12                                   2_894.83       643.74     3_538.57       0.4902          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np12                                   3_873.31     1_175.78     5_049.09       0.6308          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_114.88       677.22     3_792.10       0.4902          1.0728            1.0712         4.82
IVFPQ-m32-nl223-np1                                    2_363.33       105.73     2_469.06       0.3972          1.1048            1.1003         2.46
IVFPQ-m64-nl223-np1                                    3_419.24       167.24     3_586.47       0.4714          1.0736            1.0670         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_553.76       116.31     2_670.07       0.4484          1.0881            1.0841         4.95
IVFPQ-m32-nl223-np2                                    2_363.33       151.81     2_515.14       0.4564          1.0828            1.0807         2.46
IVFPQ-m64-nl223-np2                                    3_419.24       263.42     3_682.66       0.5690          1.0476            1.0436         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_553.76       172.08     2_725.84       0.4779          1.0767            1.0755         4.95
IVFPQ-m32-nl223-np4                                    2_363.33       257.84     2_621.17       0.4834          1.0746            1.0732         2.46
IVFPQ-m64-nl223-np4                                    3_419.24       456.42     3_875.66       0.6176          1.0374            1.0356         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_553.76       273.80     2_827.56       0.4887          1.0732            1.0718         4.95
IVFPQ-m32-nl223-np8                                    2_363.33       447.06     2_810.39       0.4900          1.0727            1.0714         2.46
IVFPQ-m64-nl223-np8                                    3_419.24       827.40     4_246.64       0.6318          1.0346            1.0332         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_553.76       476.50     3_030.26       0.4901          1.0726            1.0713         4.95
IVFPQ-m32-nl223-np11                                   2_363.33       594.89     2_958.22       0.4903          1.0726            1.0713         2.46
IVFPQ-m64-nl223-np11                                   3_419.24     1_084.31     4_503.55       0.6326          1.0345            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_553.76       624.55     3_178.31       0.4903          1.0726            1.0713         4.95
IVFPQ-m32-nl223-np14                                   2_363.33       748.40     3_111.73       0.4903          1.0726            1.0713         2.46
IVFPQ-m64-nl223-np14                                   3_419.24     1_349.49     4_768.73       0.6326          1.0345            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_553.76       791.23     3_344.99       0.4903          1.0726            1.0713         4.95
IVFPQ-m32-nl316-np1                                    2_818.03       108.80     2_926.84       0.3505          1.1213            1.1193         2.65
IVFPQ-m64-nl316-np1                                    3_753.73       148.52     3_902.25       0.3990          1.0929            1.0887         4.17
SOARPQ-shift0.5-m32-nl316-np1                          3_007.03       111.57     3_118.60       0.4211          1.0954            1.0940         5.13
IVFPQ-m32-nl316-np2                                    2_818.03       148.15     2_966.19       0.4243          1.0925            1.0914         2.65
IVFPQ-m64-nl316-np2                                    3_753.73       240.79     3_994.51       0.5158          1.0590            1.0566         4.17
SOARPQ-shift0.5-m32-nl316-np2                          3_007.03       166.22     3_173.25       0.4622          1.0812            1.0801         5.13
IVFPQ-m32-nl316-np4                                    2_818.03       244.13     3_062.17       0.4690          1.0785            1.0774         2.65
IVFPQ-m64-nl316-np4                                    3_753.73       428.28     4_182.01       0.5953          1.0419            1.0398         4.17
SOARPQ-shift0.5-m32-nl316-np4                          3_007.03       270.35     3_277.38       0.4830          1.0745            1.0733         5.13
IVFPQ-m32-nl316-np8                                    2_818.03       456.99     3_275.02       0.4869          1.0734            1.0723         2.65
IVFPQ-m64-nl316-np8                                    3_753.73       787.72     4_541.45       0.6311          1.0347            1.0333         4.17
SOARPQ-shift0.5-m32-nl316-np8                          3_007.03       465.69     3_472.73       0.4878          1.0732            1.0722         5.13
IVFPQ-m32-nl316-np15                                   2_818.03       772.42     3_590.45       0.4886          1.0729            1.0718         2.65
IVFPQ-m64-nl316-np15                                   3_753.73     1_391.47     5_145.20       0.6354          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np15                         3_007.03       810.97     3_818.00       0.4886          1.0729            1.0718         5.13
IVFPQ-m32-nl316-np17                                   2_818.03       870.99     3_689.03       0.4886          1.0729            1.0718         2.65
IVFPQ-m64-nl316-np17                                   3_753.73     1_566.79     5_320.51       0.6354          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np17                         3_007.03       913.89     3_920.92       0.4886          1.0729            1.0718         5.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.84     1_359.77     1_429.60       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_941.07       126.00     3_067.06       0.4830          1.0795            1.0733         4.82
SOARPQ-near-np2                                        2_941.07       177.13     3_118.20       0.4896          1.0736            1.0717         4.82
SOARPQ-near-np4                                        2_941.07       265.65     3_206.71       0.4902          1.0729            1.0712         4.82
SOARPQ-near-np7                                        2_941.07       413.99     3_355.05       0.4902          1.0728            1.0712         4.82
SOARPQ-near-np8                                        2_941.07       455.79     3_396.86       0.4902          1.0728            1.0712         4.82
SOARPQ-near-np12                                       2_941.07       661.45     3_602.51       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.3-np1                                    2_931.16       126.40     3_057.56       0.4829          1.0798            1.0734         4.82
SOARPQ-shift0.3-np2                                    2_931.16       174.91     3_106.08       0.4894          1.0737            1.0717         4.82
SOARPQ-shift0.3-np4                                    2_931.16       265.12     3_196.28       0.4902          1.0729            1.0713         4.82
SOARPQ-shift0.3-np7                                    2_931.16       407.26     3_338.42       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.3-np8                                    2_931.16       452.71     3_383.87       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.3-np12                                   2_931.16       652.69     3_583.85       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.7-np1                                    2_915.77       125.66     3_041.44       0.4820          1.0803            1.0736         4.82
SOARPQ-shift0.7-np2                                    2_915.77       174.54     3_090.31       0.4893          1.0739            1.0718         4.82
SOARPQ-shift0.7-np4                                    2_915.77       264.88     3_180.66       0.4901          1.0729            1.0713         4.82
SOARPQ-shift0.7-np7                                    2_915.77       406.28     3_322.06       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.7-np8                                    2_915.77       451.59     3_367.36       0.4902          1.0728            1.0712         4.82
SOARPQ-shift0.7-np12                                   2_915.77       652.88     3_568.65       0.4902          1.0728            1.0712         4.82
SOARPQ-orth1-np1                                       2_950.46       126.51     3_076.97       0.4822          1.0802            1.0736         4.82
SOARPQ-orth1-np2                                       2_950.46       174.03     3_124.49       0.4894          1.0738            1.0718         4.82
SOARPQ-orth1-np4                                       2_950.46       264.21     3_214.68       0.4901          1.0729            1.0713         4.82
SOARPQ-orth1-np7                                       2_950.46       407.19     3_357.66       0.4902          1.0728            1.0712         4.82
SOARPQ-orth1-np8                                       2_950.46       454.73     3_405.19       0.4902          1.0728            1.0712         4.82
SOARPQ-orth1-np12                                      2_950.46       654.51     3_604.98       0.4902          1.0728            1.0712         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.55     1_318.54     1_389.08       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_989.25        98.74     2_087.99       0.7055          1.1845            1.0982         2.24
IVFPQ-m64-nl111-np1                                    2_813.24       161.55     2_974.79       0.7175          1.1739            1.0866         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_136.31       122.30     2_258.61       0.8142          1.0740            1.0458         4.72
IVFPQ-m32-nl111-np2                                    1_989.25       158.87     2_148.13       0.8212          1.0638            1.0402         2.24
IVFPQ-m64-nl111-np2                                    2_813.24       285.11     3_098.35       0.8434          1.0521            1.0275         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_136.31       209.87     2_346.18       0.8488          1.0446            1.0337         4.72
IVFPQ-m32-nl111-np4                                    1_989.25       281.13     2_270.39       0.8540          1.0384            1.0306         2.24
IVFPQ-m64-nl111-np4                                    2_813.24       527.97     3_341.22       0.8803          1.0260            1.0194         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_136.31       367.37     2_503.67       0.8555          1.0387            1.0306         4.72
IVFPQ-m32-nl111-np5                                    1_989.25       343.95     2_333.20       0.8556          1.0373            1.0301         2.24
IVFPQ-m64-nl111-np5                                    2_813.24       653.34     3_466.58       0.8821          1.0249            1.0190         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_136.31       444.93     2_581.23       0.8559          1.0382            1.0302         4.72
IVFPQ-m32-nl111-np8                                    1_989.25       531.01     2_520.27       0.8566          1.0367            1.0298         2.24
IVFPQ-m64-nl111-np8                                    2_813.24     1_038.35     3_851.60       0.8833          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_136.31       669.94     2_806.25       0.8565          1.0371            1.0299         4.72
IVFPQ-m32-nl111-np10                                   1_989.25       659.20     2_648.46       0.8566          1.0367            1.0298         2.24
IVFPQ-m64-nl111-np10                                   2_813.24     1_276.02     4_089.26       0.8834          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_136.31       792.01     2_928.31       0.8566          1.0369            1.0299         4.72
IVFPQ-m32-nl158-np1                                    3_023.38        95.74     3_119.12       0.7125          1.1767            1.0939         2.34
IVFPQ-m64-nl158-np1                                    3_795.23       148.43     3_943.66       0.7212          1.1686            1.0854         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_151.24       110.67     3_261.91       0.8237          1.0695            1.0397         4.82
IVFPQ-m32-nl158-np2                                    3_023.38       149.47     3_172.85       0.8339          1.0567            1.0334         2.34
IVFPQ-m64-nl158-np2                                    3_795.23       254.79     4_050.02       0.8510          1.0477            1.0238         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_151.24       181.08     3_332.32       0.8617          1.0397            1.0274         4.82
IVFPQ-m32-nl158-np4                                    3_023.38       259.39     3_282.77       0.8687          1.0320            1.0241         2.34
IVFPQ-m64-nl158-np4                                    3_795.23       477.35     4_272.58       0.8892          1.0225            1.0159         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_151.24       317.67     3_468.91       0.8703          1.0331            1.0244         4.82
IVFPQ-m32-nl158-np7                                    3_023.38       434.57     3_457.95       0.8726          1.0297            1.0231         2.34
IVFPQ-m64-nl158-np7                                    3_795.23       809.26     4_604.49       0.8936          1.0202            1.0150         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_151.24       512.62     3_663.86       0.8724          1.0305            1.0234         4.82
IVFPQ-m32-nl158-np8                                    3_023.38       487.50     3_510.88       0.8730          1.0295            1.0230         2.34
IVFPQ-m64-nl158-np8                                    3_795.23       915.17     4_710.40       0.8939          1.0201            1.0150         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_151.24       582.95     3_734.20       0.8727          1.0302            1.0233         4.82
IVFPQ-m32-nl158-np12                                   3_023.38       722.27     3_745.65       0.8731          1.0294            1.0230         2.34
IVFPQ-m64-nl158-np12                                   3_795.23     1_348.51     5_143.74       0.8941          1.0200            1.0149         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_151.24       834.24     3_985.48       0.8731          1.0296            1.0231         4.82
IVFPQ-m32-nl223-np1                                    2_025.52        95.55     2_121.07       0.6875          1.1973            1.1237         2.46
IVFPQ-m64-nl223-np1                                    2_934.44       141.05     3_075.49       0.6935          1.1897            1.1152         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_222.65       104.26     2_326.92       0.8134          1.0794            1.0465         4.95
IVFPQ-m32-nl223-np2                                    2_025.52       144.10     2_169.62       0.8245          1.0642            1.0373         2.46
IVFPQ-m64-nl223-np2                                    2_934.44       235.50     3_169.94       0.8396          1.0562            1.0281         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_222.65       163.65     2_386.30       0.8646          1.0397            1.0263         4.95
IVFPQ-m32-nl223-np4                                    2_025.52       244.25     2_269.77       0.8726          1.0303            1.0220         2.46
IVFPQ-m64-nl223-np4                                    2_934.44       446.29     3_380.74       0.8924          1.0218            1.0146         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_222.65       283.39     2_506.04       0.8758          1.0314            1.0219         4.95
IVFPQ-m32-nl223-np8                                    2_025.52       453.89     2_479.41       0.8792          1.0267            1.0203         2.46
IVFPQ-m64-nl223-np8                                    2_934.44       824.59     3_759.04       0.8998          1.0180            1.0130         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_222.65       554.42     2_777.07       0.8786          1.0279            1.0206         4.95
IVFPQ-m32-nl223-np11                                   2_025.52       616.13     2_641.65       0.8795          1.0266            1.0202         2.46
IVFPQ-m64-nl223-np11                                   2_934.44     1_122.46     4_056.90       0.9002          1.0178            1.0129         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_222.65       698.18     2_920.83       0.8792          1.0271            1.0203         4.95
IVFPQ-m32-nl223-np14                                   2_025.52       775.48     2_801.00       0.8795          1.0265            1.0202         2.46
IVFPQ-m64-nl223-np14                                   2_934.44     1_420.44     4_354.89       0.9003          1.0178            1.0129         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_222.65       862.23     3_084.89       0.8794          1.0268            1.0202         4.95
IVFPQ-m32-nl316-np1                                    2_256.45        99.26     2_355.71       0.6725          1.2102            1.1380         2.65
IVFPQ-m64-nl316-np1                                    3_098.20       139.28     3_237.49       0.6767          1.2049            1.1332         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_516.54       102.16     2_618.70       0.8094          1.0840            1.0496         5.13
IVFPQ-m32-nl316-np2                                    2_256.45       143.85     2_400.30       0.8228          1.0663            1.0373         2.65
IVFPQ-m64-nl316-np2                                    3_098.20       227.97     3_326.18       0.8331          1.0605            1.0306         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_516.54       159.35     2_675.89       0.8708          1.0389            1.0233         5.13
IVFPQ-m32-nl316-np4                                    2_256.45       240.29     2_496.73       0.8817          1.0267            1.0187         2.65
IVFPQ-m64-nl316-np4                                    3_098.20       431.01     3_529.22       0.8960          1.0207            1.0134         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_516.54       267.39     2_783.93       0.8858          1.0287            1.0184         5.13
IVFPQ-m32-nl316-np8                                    2_256.45       446.17     2_702.62       0.8910          1.0218            1.0162         2.65
IVFPQ-m64-nl316-np8                                    3_098.20       778.87     3_877.08       0.9056          1.0158            1.0114         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_516.54       483.01     2_999.56       0.8900          1.0240            1.0167         5.13
IVFPQ-m32-nl316-np15                                   2_256.45       801.98     3_058.43       0.8917          1.0215            1.0159         2.65
IVFPQ-m64-nl316-np15                                   3_098.20     1_431.74     4_529.95       0.9064          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_516.54       863.07     3_379.61       0.8915          1.0218            1.0161         5.13
IVFPQ-m32-nl316-np17                                   2_256.45       906.07     3_162.52       0.8917          1.0215            1.0159         2.65
IVFPQ-m64-nl316-np17                                   3_098.20     1_623.00     4_721.20       0.9065          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_516.54       976.28     3_492.82       0.8916          1.0216            1.0160         5.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.55     1_318.54     1_389.08       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        3_189.09       113.04     3_302.12       0.8232          1.0711            1.0373         4.82
SOARPQ-near-np2                                        3_189.09       179.18     3_368.26       0.8631          1.0379            1.0263         4.82
SOARPQ-near-np4                                        3_189.09       316.07     3_505.16       0.8714          1.0316            1.0238         4.82
SOARPQ-near-np7                                        3_189.09       509.95     3_699.04       0.8727          1.0300            1.0232         4.82
SOARPQ-near-np8                                        3_189.09       575.33     3_764.42       0.8729          1.0298            1.0232         4.82
SOARPQ-near-np12                                       3_189.09       824.66     4_013.75       0.8731          1.0295            1.0230         4.82
SOARPQ-shift0.3-np1                                    3_164.76       109.67     3_274.43       0.8257          1.0677            1.0384         4.82
SOARPQ-shift0.3-np2                                    3_164.76       191.63     3_356.39       0.8627          1.0386            1.0268         4.82
SOARPQ-shift0.3-np4                                    3_164.76       313.17     3_477.92       0.8707          1.0325            1.0242         4.82
SOARPQ-shift0.3-np7                                    3_164.76       509.17     3_673.93       0.8725          1.0304            1.0233         4.82
SOARPQ-shift0.3-np8                                    3_164.76       570.08     3_734.84       0.8727          1.0301            1.0233         4.82
SOARPQ-shift0.3-np12                                   3_164.76       816.33     3_981.08       0.8731          1.0296            1.0230         4.82
SOARPQ-shift0.7-np1                                    3_155.63       110.58     3_266.21       0.8207          1.0725            1.0413         4.82
SOARPQ-shift0.7-np2                                    3_155.63       178.21     3_333.84       0.8608          1.0409            1.0279         4.82
SOARPQ-shift0.7-np4                                    3_155.63       312.31     3_467.93       0.8699          1.0337            1.0246         4.82
SOARPQ-shift0.7-np7                                    3_155.63       507.69     3_663.32       0.8723          1.0308            1.0234         4.82
SOARPQ-shift0.7-np8                                    3_155.63       569.72     3_725.34       0.8726          1.0304            1.0233         4.82
SOARPQ-shift0.7-np12                                   3_155.63       828.90     3_984.53       0.8730          1.0296            1.0231         4.82
SOARPQ-orth1-np1                                       3_159.44       110.24     3_269.68       0.8233          1.0708            1.0386         4.82
SOARPQ-orth1-np2                                       3_159.44       180.19     3_339.63       0.8623          1.0392            1.0268         4.82
SOARPQ-orth1-np4                                       3_159.44       311.89     3_471.33       0.8707          1.0327            1.0241         4.82
SOARPQ-orth1-np7                                       3_159.44       510.12     3_669.56       0.8725          1.0303            1.0233         4.82
SOARPQ-orth1-np8                                       3_159.44       573.20     3_732.64       0.8727          1.0301            1.0232         4.82
SOARPQ-orth1-np12                                      3_159.44       834.67     3_994.11       0.8731          1.0296            1.0231         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.28     1_331.62     1_404.90       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                    1_846.85        97.18     1_944.02       0.7688          1.1400            1.0602         2.24
IVFPQ-m64-nl111-np1                                    2_697.81       153.47     2_851.27       0.7773          1.1327            1.0508         3.77
SOARPQ-orth1-m32-nl111-np1                             2_038.27       113.38     2_151.65       0.8462          1.0725            1.0359         4.72
IVFPQ-m32-nl111-np2                                    1_846.85       152.64     1_999.48       0.8642          1.0449            1.0274         2.24
IVFPQ-m64-nl111-np2                                    2_697.81       266.98     2_964.78       0.8774          1.0371            1.0203         3.77
SOARPQ-orth1-m32-nl111-np2                             2_038.27       197.65     2_235.93       0.8750          1.0430            1.0260         4.72
IVFPQ-m32-nl111-np4                                    1_846.85       266.54     2_113.39       0.8829          1.0309            1.0223         2.24
IVFPQ-m64-nl111-np4                                    2_697.81       502.15     3_199.95       0.8972          1.0232            1.0162         3.77
SOARPQ-orth1-m32-nl111-np4                             2_038.27       330.48     2_368.76       0.8821          1.0345            1.0231         4.72
IVFPQ-m32-nl111-np5                                    1_846.85       326.32     2_173.17       0.8839          1.0304            1.0221         2.24
IVFPQ-m64-nl111-np5                                    2_697.81       617.27     3_315.08       0.8981          1.0227            1.0160         3.77
SOARPQ-orth1-m32-nl111-np5                             2_038.27       399.75     2_438.02       0.8830          1.0332            1.0227         4.72
IVFPQ-m32-nl111-np8                                    1_846.85       549.09     2_395.93       0.8844          1.0302            1.0219         2.24
IVFPQ-m64-nl111-np8                                    2_697.81       967.12     3_664.92       0.8987          1.0225            1.0158         3.77
SOARPQ-orth1-m32-nl111-np8                             2_038.27       601.53     2_639.80       0.8841          1.0311            1.0221         4.72
IVFPQ-m32-nl111-np10                                   1_846.85       626.49     2_473.34       0.8844          1.0301            1.0219         2.24
IVFPQ-m64-nl111-np10                                   2_697.81     1_195.61     3_893.42       0.8987          1.0225            1.0158         3.77
SOARPQ-orth1-m32-nl111-np10                            2_038.27       726.28     2_764.56       0.8843          1.0305            1.0220         4.72
IVFPQ-m32-nl158-np1                                    2_862.42        91.37     2_953.79       0.7527          1.1567            1.0752         2.34
IVFPQ-m64-nl158-np1                                    3_718.34       140.68     3_859.02       0.7586          1.1514            1.0685         3.86
SOARPQ-orth1-m32-nl158-np1                             3_096.83       103.12     3_199.95       0.8448          1.0760            1.0352         4.82
IVFPQ-m32-nl158-np2                                    2_862.42       142.53     3_004.95       0.8658          1.0456            1.0252         2.34
IVFPQ-m64-nl158-np2                                    3_718.34       240.94     3_959.28       0.8753          1.0404            1.0199         3.86
SOARPQ-orth1-m32-nl158-np2                             3_096.83       166.45     3_263.28       0.8833          1.0399            1.0221         4.82
IVFPQ-m32-nl158-np4                                    2_862.42       247.66     3_110.08       0.8933          1.0250            1.0182         2.34
IVFPQ-m64-nl158-np4                                    3_718.34       445.14     4_163.47       0.9045          1.0197            1.0138         3.86
SOARPQ-orth1-m32-nl158-np4                             3_096.83       292.04     3_388.87       0.8929          1.0292            1.0188         4.82
IVFPQ-m32-nl158-np7                                    2_862.42       410.96     3_273.38       0.8955          1.0238            1.0178         2.34
IVFPQ-m64-nl158-np7                                    3_718.34       758.17     4_476.51       0.9069          1.0185            1.0134         3.86
SOARPQ-orth1-m32-nl158-np7                             3_096.83       477.04     3_573.88       0.8951          1.0254            1.0180         4.82
IVFPQ-m32-nl158-np8                                    2_862.42       470.21     3_332.63       0.8956          1.0238            1.0177         2.34
IVFPQ-m64-nl158-np8                                    3_718.34       867.56     4_585.90       0.9070          1.0184            1.0133         3.86
SOARPQ-orth1-m32-nl158-np8                             3_096.83       538.89     3_635.73       0.8952          1.0250            1.0179         4.82
IVFPQ-m32-nl158-np12                                   2_862.42       686.57     3_548.99       0.8957          1.0238            1.0177         2.34
IVFPQ-m64-nl158-np12                                   3_718.34     1_301.03     5_019.37       0.9071          1.0184            1.0133         3.86
SOARPQ-orth1-m32-nl158-np12                            3_096.83       781.66     3_878.50       0.8956          1.0241            1.0178         4.82
IVFPQ-m32-nl223-np1                                    2_162.35        91.01     2_253.36       0.7283          1.1815            1.1041         2.46
IVFPQ-m64-nl223-np1                                    2_974.18       135.41     3_109.59       0.7318          1.1772            1.1004         3.99
SOARPQ-orth1-m32-nl223-np1                             2_374.84        98.40     2_473.25       0.8393          1.0777            1.0371         4.95
IVFPQ-m32-nl223-np2                                    2_162.35       140.83     2_303.18       0.8613          1.0501            1.0263         2.46
IVFPQ-m64-nl223-np2                                    2_974.18       228.97     3_203.15       0.8679          1.0460            1.0217         3.99
SOARPQ-orth1-m32-nl223-np2                             2_374.84       154.01     2_528.85       0.8871          1.0370            1.0205         4.95
IVFPQ-m32-nl223-np4                                    2_162.35       236.51     2_398.86       0.8979          1.0231            1.0163         2.46
IVFPQ-m64-nl223-np4                                    2_974.18       420.47     3_394.65       0.9065          1.0191            1.0133         3.99
SOARPQ-orth1-m32-nl223-np4                             2_374.84       264.01     2_638.86       0.8976          1.0276            1.0170         4.95
IVFPQ-m32-nl223-np8                                    2_162.35       437.73     2_600.08       0.9010          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np8                                    2_974.18       791.83     3_766.02       0.9099          1.0173            1.0126         3.99
SOARPQ-orth1-m32-nl223-np8                             2_374.84       484.36     2_859.21       0.9006          1.0227            1.0160         4.95
IVFPQ-m32-nl223-np11                                   2_162.35       595.94     2_758.29       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np11                                   2_974.18     1_082.31     4_056.49       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np11                            2_374.84       654.18     3_029.03       0.9009          1.0218            1.0158         4.95
IVFPQ-m32-nl223-np14                                   2_162.35       752.15     2_914.50       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np14                                   2_974.18     1_368.76     4_342.95       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np14                            2_374.84       828.18     3_203.03       0.9011          1.0216            1.0157         4.95
IVFPQ-m32-nl316-np1                                    2_485.52        97.44     2_582.96       0.7037          1.2091            1.1328         2.65
IVFPQ-m64-nl316-np1                                    3_319.63       135.94     3_455.57       0.7071          1.2044            1.1272         4.17
SOARPQ-orth1-m32-nl316-np1                             2_770.98       100.65     2_871.63       0.8250          1.0882            1.0452         5.13
IVFPQ-m32-nl316-np2                                    2_485.52       140.06     2_625.58       0.8495          1.0587            1.0313         2.65
IVFPQ-m64-nl316-np2                                    3_319.63       224.10     3_543.73       0.8573          1.0541            1.0258         4.17
SOARPQ-orth1-m32-nl316-np2                             2_770.98       154.87     2_925.85       0.8855          1.0375            1.0212         5.13
IVFPQ-m32-nl316-np4                                    2_485.52       234.45     2_719.97       0.8982          1.0231            1.0160         2.65
IVFPQ-m64-nl316-np4                                    3_319.63       400.34     3_719.97       0.9091          1.0184            1.0119         4.17
SOARPQ-orth1-m32-nl316-np4                             2_770.98       258.54     3_029.53       0.8994          1.0269            1.0166         5.13
IVFPQ-m32-nl316-np8                                    2_485.52       428.96     2_914.49       0.9033          1.0203            1.0148         2.65
IVFPQ-m64-nl316-np8                                    3_319.63       785.06     4_104.69       0.9144          1.0156            1.0109         4.17
SOARPQ-orth1-m32-nl316-np8                             2_770.98       458.78     3_229.77       0.9028          1.0220            1.0152         5.13
IVFPQ-m32-nl316-np15                                   2_485.52       777.05     3_262.57       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np15                                   3_319.63     1_395.73     4_715.36       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np15                            2_770.98       829.67     3_600.65       0.9035          1.0204            1.0148         5.13
IVFPQ-m32-nl316-np17                                   2_485.52       893.38     3_378.90       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np17                                   3_319.63     1_577.79     4_897.42       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np17                            2_770.98       941.58     3_712.56       0.9035          1.0203            1.0148         5.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.28     1_331.62     1_404.90       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        3_046.30       106.10     3_152.40       0.8512          1.0652            1.0326         4.82
SOARPQ-near-np2                                        3_046.30       166.80     3_213.10       0.8858          1.0343            1.0213         4.82
SOARPQ-near-np4                                        3_046.30       288.80     3_335.10       0.8938          1.0267            1.0185         4.82
SOARPQ-near-np7                                        3_046.30       477.28     3_523.58       0.8953          1.0244            1.0179         4.82
SOARPQ-near-np8                                        3_046.30       539.81     3_586.11       0.8954          1.0242            1.0178         4.82
SOARPQ-near-np12                                       3_046.30       773.96     3_820.26       0.8957          1.0239            1.0177         4.82
SOARPQ-shift0.3-np1                                    3_082.58       104.18     3_186.76       0.8500          1.0681            1.0342         4.82
SOARPQ-shift0.3-np2                                    3_082.58       170.71     3_253.29       0.8839          1.0373            1.0220         4.82
SOARPQ-shift0.3-np4                                    3_082.58       289.71     3_372.29       0.8927          1.0284            1.0189         4.82
SOARPQ-shift0.3-np7                                    3_082.58       477.73     3_560.30       0.8950          1.0250            1.0181         4.82
SOARPQ-shift0.3-np8                                    3_082.58       537.22     3_619.79       0.8952          1.0247            1.0180         4.82
SOARPQ-shift0.3-np12                                   3_082.58       775.23     3_857.81       0.8956          1.0240            1.0178         4.82
SOARPQ-shift0.7-np1                                    3_047.38       106.03     3_153.41       0.8445          1.0765            1.0365         4.82
SOARPQ-shift0.7-np2                                    3_047.38       166.61     3_213.99       0.8805          1.0432            1.0229         4.82
SOARPQ-shift0.7-np4                                    3_047.38       290.35     3_337.73       0.8912          1.0314            1.0195         4.82
SOARPQ-shift0.7-np7                                    3_047.38       475.07     3_522.45       0.8945          1.0262            1.0183         4.82
SOARPQ-shift0.7-np8                                    3_047.38       547.06     3_594.43       0.8949          1.0255            1.0181         4.82
SOARPQ-shift0.7-np12                                   3_047.38       775.79     3_823.17       0.8955          1.0242            1.0178         4.82
SOARPQ-orth1-np1                                       3_089.43       103.69     3_193.12       0.8448          1.0760            1.0352         4.82
SOARPQ-orth1-np2                                       3_089.43       170.49     3_259.92       0.8833          1.0399            1.0221         4.82
SOARPQ-orth1-np4                                       3_089.43       290.50     3_379.93       0.8929          1.0292            1.0188         4.82
SOARPQ-orth1-np7                                       3_089.43       476.22     3_565.65       0.8951          1.0254            1.0180         4.82
SOARPQ-orth1-np8                                       3_089.43       536.56     3_625.99       0.8952          1.0250            1.0179         4.82
SOARPQ-orth1-np12                                      3_089.43       772.02     3_861.45       0.8956          1.0241            1.0178         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.81     1_219.19     1_287.00       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_548.68       454.59     8_003.26       0.3575          1.0701            1.0699         3.49
IVFOPQ-m64-nl111-np1                                  11_497.99       546.87    12_044.86       0.4573          1.0465            1.0431         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_785.99       458.18     9_244.17       0.3377          1.2404            1.0721         5.98
IVFOPQ-m32-nl111-np2                                   7_548.68       513.85     8_062.52       0.3594          1.0694            1.0696         3.49
IVFOPQ-m64-nl111-np2                                  11_497.99       647.83    12_145.81       0.4604          1.0454            1.0428         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_785.99       512.43     9_298.42       0.3591          1.0699            1.0696         5.98
IVFOPQ-m32-nl111-np4                                   7_548.68       599.13     8_147.81       0.3595          1.0694            1.0695         3.49
IVFOPQ-m64-nl111-np4                                  11_497.99       815.87    12_313.86       0.4604          1.0454            1.0428         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_785.99       614.12     9_400.11       0.3595          1.0694            1.0695         5.98
IVFOPQ-m32-nl111-np5                                   7_548.68       640.47     8_189.15       0.3595          1.0694            1.0695         3.49
IVFOPQ-m64-nl111-np5                                  11_497.99       953.47    12_451.46       0.4604          1.0454            1.0428         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_785.99       663.42     9_449.41       0.3595          1.0694            1.0695         5.98
IVFOPQ-m32-nl111-np8                                   7_548.68       783.41     8_332.09       0.3595          1.0694            1.0695         3.49
IVFOPQ-m64-nl111-np8                                  11_497.99     1_177.18    12_675.16       0.4604          1.0454            1.0428         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_785.99       822.91     9_608.90       0.3595          1.0694            1.0695         5.98
IVFOPQ-m32-nl111-np10                                  7_548.68       900.74     8_449.42       0.3595          1.0694            1.0695         3.49
IVFOPQ-m64-nl111-np10                                 11_497.99     1_354.23    12_852.22       0.4604          1.0454            1.0428         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_785.99       926.34     9_712.33       0.3595          1.0694            1.0695         5.98
IVFOPQ-m32-nl158-np1                                   9_297.35       451.01     9_748.36       0.3632          1.0674            1.0680         3.84
IVFOPQ-m64-nl158-np1                                  12_976.39       562.99    13_539.37       0.4674          1.0437            1.0411         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_105.43       457.50    10_562.93       0.3319          1.1498            1.0715         6.32
IVFOPQ-m32-nl158-np2                                   9_297.35       499.74     9_797.09       0.3683          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np2                                  12_976.39       645.64    13_622.02       0.4759          1.0415            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_105.43       509.75    10_615.18       0.3678          1.0666            1.0672         6.32
IVFOPQ-m32-nl158-np4                                   9_297.35       593.32     9_890.67       0.3685          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np4                                  12_976.39       805.64    13_782.02       0.4762          1.0414            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_105.43       606.53    10_711.96       0.3685          1.0656            1.0671         6.32
IVFOPQ-m32-nl158-np7                                   9_297.35       734.90    10_032.25       0.3685          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np7                                  12_976.39     1_063.78    14_040.17       0.4762          1.0414            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_105.43       763.09    10_868.52       0.3685          1.0656            1.0671         6.32
IVFOPQ-m32-nl158-np8                                   9_297.35       792.73    10_090.08       0.3685          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np8                                  12_976.39     1_146.79    14_123.18       0.4762          1.0414            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_105.43       815.46    10_920.89       0.3685          1.0656            1.0671         6.32
IVFOPQ-m32-nl158-np12                                  9_297.35       964.86    10_262.21       0.3685          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np12                                 12_976.39     1_489.36    14_465.75       0.4762          1.0414            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_105.43     1_003.00    11_108.43       0.3685          1.0656            1.0671         6.32
IVFOPQ-m32-nl223-np1                                   8_509.77       430.75     8_940.52       0.3598          1.0669            1.0675         3.96
IVFOPQ-m64-nl223-np1                                  12_642.37       498.10    13_140.47       0.4473          1.0462            1.0434         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_795.17       439.25    10_234.42       0.3445          1.1272            1.0695         6.45
IVFOPQ-m32-nl223-np2                                   8_509.77       507.22     9_016.99       0.3740          1.0627            1.0641         3.96
IVFOPQ-m64-nl223-np2                                  12_642.37       595.64    13_238.01       0.4771          1.0401            1.0389         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_795.17       499.34    10_294.51       0.3722          1.0642            1.0655         6.45
IVFOPQ-m32-nl223-np4                                   8_509.77       580.69     9_090.46       0.3776          1.0619            1.0631         3.96
IVFOPQ-m64-nl223-np4                                  12_642.37       779.42    13_421.79       0.4869          1.0386            1.0374         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_795.17       601.92    10_397.09       0.3763          1.0627            1.0636         6.45
IVFOPQ-m32-nl223-np8                                   8_509.77       799.45     9_309.22       0.3782          1.0618            1.0630         3.96
IVFOPQ-m64-nl223-np8                                  12_642.37     1_132.47    13_774.84       0.4881          1.0384            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_795.17       796.98    10_592.15       0.3780          1.0619            1.0630         6.45
IVFOPQ-m32-nl223-np11                                  8_509.77       958.64     9_468.42       0.3782          1.0618            1.0630         3.96
IVFOPQ-m64-nl223-np11                                 12_642.37     1_388.05    14_030.43       0.4881          1.0384            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_795.17       940.29    10_735.46       0.3782          1.0618            1.0630         6.45
IVFOPQ-m32-nl223-np14                                  8_509.77     1_055.53     9_565.30       0.3782          1.0618            1.0630         3.96
IVFOPQ-m64-nl223-np14                                 12_642.37     1_645.94    14_288.31       0.4882          1.0384            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_795.17     1_092.52    10_887.69       0.3782          1.0618            1.0630         6.45
IVFOPQ-m32-nl316-np1                                   8_736.22       429.52     9_165.74       0.3581          1.0669            1.0662         4.65
IVFOPQ-m64-nl316-np1                                  13_035.40       476.26    13_511.66       0.4342          1.0484            1.0450         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_392.53       437.01    10_829.55       0.3374          1.1218            1.0693         7.13
IVFOPQ-m32-nl316-np2                                   8_736.22       499.75     9_235.97       0.3784          1.0603            1.0617         4.65
IVFOPQ-m64-nl316-np2                                  13_035.40       574.05    13_609.44       0.4759          1.0395            1.0385         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_392.53       493.94    10_886.47       0.3760          1.0622            1.0634         7.13
IVFOPQ-m32-nl316-np4                                   8_736.22       581.55     9_317.77       0.3850          1.0587            1.0602         4.65
IVFOPQ-m64-nl316-np4                                  13_035.40       764.67    13_800.07       0.4922          1.0370            1.0360         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_392.53       594.08    10_986.61       0.3819          1.0602            1.0613         7.13
IVFOPQ-m32-nl316-np8                                   8_736.22       766.66     9_502.88       0.3859          1.0585            1.0601         4.65
IVFOPQ-m64-nl316-np8                                  13_035.40     1_119.61    14_155.00       0.4958          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_392.53       785.37    11_177.90       0.3853          1.0589            1.0603         7.13
IVFOPQ-m32-nl316-np15                                  8_736.22     1_105.73     9_841.94       0.3859          1.0585            1.0601         4.65
IVFOPQ-m64-nl316-np15                                 13_035.40     1_733.65    14_769.05       0.4960          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_392.53     1_114.85    11_507.38       0.3859          1.0586            1.0601         7.13
IVFOPQ-m32-nl316-np17                                  8_736.22     1_198.84     9_935.06       0.3859          1.0585            1.0601         4.65
IVFOPQ-m64-nl316-np17                                 13_035.40     1_898.42    14_933.81       0.4960          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_392.53     1_223.39    11_615.92       0.3859          1.0586            1.0601         7.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.81     1_219.19     1_287.00       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_157.85       467.72    10_625.57       0.3326          1.1482            1.0715         6.32
SOAROPQ-near-np2                                      10_157.85       512.96    10_670.81       0.3679          1.0664            1.0672         6.32
SOAROPQ-near-np4                                      10_157.85       603.86    10_761.71       0.3685          1.0656            1.0671         6.32
SOAROPQ-near-np7                                      10_157.85       746.16    10_904.01       0.3685          1.0656            1.0671         6.32
SOAROPQ-near-np8                                      10_157.85       790.04    10_947.89       0.3685          1.0656            1.0671         6.32
SOAROPQ-near-np12                                     10_157.85       975.05    11_132.89       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.3-np1                                  10_154.87       461.19    10_616.06       0.3320          1.1493            1.0715         6.32
SOAROPQ-shift0.3-np2                                  10_154.87       512.25    10_667.12       0.3679          1.0665            1.0672         6.32
SOAROPQ-shift0.3-np4                                  10_154.87       603.18    10_758.06       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.3-np7                                  10_154.87       747.71    10_902.58       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.3-np8                                  10_154.87       791.85    10_946.72       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.3-np12                                 10_154.87       976.86    11_131.74       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.7-np1                                  10_156.29       458.85    10_615.14       0.3319          1.1503            1.0715         6.32
SOAROPQ-shift0.7-np2                                  10_156.29       509.16    10_665.45       0.3678          1.0666            1.0672         6.32
SOAROPQ-shift0.7-np4                                  10_156.29       604.90    10_761.20       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.7-np7                                  10_156.29       745.17    10_901.46       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.7-np8                                  10_156.29       790.72    10_947.01       0.3685          1.0656            1.0671         6.32
SOAROPQ-shift0.7-np12                                 10_156.29       978.71    11_135.00       0.3685          1.0656            1.0671         6.32
SOAROPQ-orth1-np1                                     10_125.56       457.91    10_583.47       0.3331          1.1478            1.0714         6.32
SOAROPQ-orth1-np2                                     10_125.56       510.21    10_635.77       0.3680          1.0662            1.0672         6.32
SOAROPQ-orth1-np4                                     10_125.56       603.10    10_728.66       0.3685          1.0656            1.0671         6.32
SOAROPQ-orth1-np7                                     10_125.56       745.94    10_871.50       0.3685          1.0656            1.0671         6.32
SOAROPQ-orth1-np8                                     10_125.56       789.21    10_914.77       0.3685          1.0656            1.0671         6.32
SOAROPQ-orth1-np12                                    10_125.56       975.41    11_100.97       0.3685          1.0656            1.0671         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.56     1_301.36     1_368.92       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_625.12       451.88     8_076.99       0.6645          1.0303            1.0250         3.49
IVFOPQ-m64-nl111-np1                                  11_603.07       541.06    12_144.13       0.7569          1.0171            1.0113         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_823.72       454.89     9_278.61       0.6653          1.0333            1.0253         5.98
IVFOPQ-m32-nl111-np2                                   7_625.12       497.93     8_123.05       0.6772          1.0262            1.0246         3.49
IVFOPQ-m64-nl111-np2                                  11_603.07       629.39    12_232.46       0.7723          1.0126            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_823.72       505.14     9_328.87       0.6773          1.0262            1.0246         5.98
IVFOPQ-m32-nl111-np4                                   7_625.12       592.59     8_217.71       0.6780          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np4                                  11_603.07       804.18    12_407.26       0.7734          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_823.72       603.27     9_426.99       0.6780          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np5                                   7_625.12       639.86     8_264.98       0.6780          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np5                                  11_603.07       892.12    12_495.19       0.7734          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_823.72       655.68     9_479.40       0.6780          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np8                                   7_625.12       790.09     8_415.21       0.6780          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np8                                  11_603.07     1_149.65    12_752.73       0.7734          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_823.72       851.58     9_675.31       0.6780          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np10                                  7_625.12       888.91     8_514.02       0.6780          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np10                                 11_603.07     1_336.77    12_939.85       0.7734          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_823.72       946.33     9_770.06       0.6780          1.0259            1.0245         5.98
IVFOPQ-m32-nl158-np1                                   8_775.05       459.18     9_234.23       0.6658          1.0303            1.0244         3.84
IVFOPQ-m64-nl158-np1                                  12_954.30       541.26    13_495.56       0.7560          1.0178            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_085.85       453.68    10_539.53       0.6765          1.0276            1.0242         6.32
IVFOPQ-m32-nl158-np2                                   8_775.05       495.94     9_270.99       0.6818          1.0251            1.0236         3.84
IVFOPQ-m64-nl158-np2                                  12_954.30       629.05    13_583.35       0.7752          1.0123            1.0107         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_085.85       503.40    10_589.24       0.6832          1.0249            1.0235         6.32
IVFOPQ-m32-nl158-np4                                   8_775.05       595.83     9_370.87       0.6839          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np4                                  12_954.30       798.51    13_752.81       0.7777          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_085.85       595.75    10_681.60       0.6839          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np7                                   8_775.05       728.73     9_503.78       0.6839          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np7                                  12_954.30     1_050.60    14_004.91       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_085.85       737.67    10_823.51       0.6839          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np8                                   8_775.05       771.87     9_546.92       0.6839          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np8                                  12_954.30     1_134.70    14_089.00       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_085.85       777.73    10_863.58       0.6839          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np12                                  8_775.05       956.53     9_731.58       0.6839          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np12                                 12_954.30     1_472.17    14_426.47       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_085.85       971.88    11_057.72       0.6839          1.0246            1.0234         6.32
IVFOPQ-m32-nl223-np1                                   8_610.48       432.23     9_042.71       0.4958          1.0650            1.0575         3.96
IVFOPQ-m64-nl223-np1                                  12_736.14       491.01    13_227.16       0.5347          1.0542            1.0468         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_002.08       444.65    10_446.73       0.6015          1.0413            1.0349         6.45
IVFOPQ-m32-nl223-np2                                   8_610.48       486.45     9_096.93       0.6139          1.0371            1.0319         3.96
IVFOPQ-m64-nl223-np2                                  12_736.14       590.67    13_326.81       0.6810          1.0254            1.0187         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_002.08       498.91    10_500.99       0.6632          1.0282            1.0260         6.45
IVFOPQ-m32-nl223-np4                                   8_610.48       583.07     9_193.55       0.6728          1.0264            1.0245         3.96
IVFOPQ-m64-nl223-np4                                  12_736.14       778.81    13_514.95       0.7579          1.0143            1.0120         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_002.08       603.88    10_605.96       0.6868          1.0240            1.0228         6.45
IVFOPQ-m32-nl223-np8                                   8_610.48       770.69     9_381.18       0.6902          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np8                                  12_736.14     1_123.13    13_859.28       0.7800          1.0113            1.0104         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_002.08       781.51    10_783.58       0.6911          1.0233            1.0222         6.45
IVFOPQ-m32-nl223-np11                                  8_610.48       911.24     9_521.72       0.6914          1.0233            1.0222         3.96
IVFOPQ-m64-nl223-np11                                 12_736.14     1_378.75    14_114.89       0.7814          1.0111            1.0102         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_002.08       916.53    10_918.61       0.6914          1.0233            1.0222         6.45
IVFOPQ-m32-nl223-np14                                  8_610.48     1_054.16     9_664.64       0.6914          1.0233            1.0222         3.96
IVFOPQ-m64-nl223-np14                                 12_736.14     1_644.34    14_380.48       0.7814          1.0111            1.0102         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_002.08     1_067.67    11_069.75       0.6914          1.0233            1.0222         6.45
IVFOPQ-m32-nl316-np1                                   8_930.09       431.46     9_361.56       0.4098          1.0851            1.0798         4.65
IVFOPQ-m64-nl316-np1                                  13_218.67       477.79    13_696.47       0.4287          1.0751            1.0695         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_556.29       435.91    10_992.20       0.5374          1.0529            1.0490         7.13
IVFOPQ-m32-nl316-np2                                   8_930.09       492.36     9_422.46       0.5478          1.0493            1.0457         4.65
IVFOPQ-m64-nl316-np2                                  13_218.67       573.36    13_792.03       0.5951          1.0382            1.0341         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_556.29       493.06    11_049.35       0.6304          1.0339            1.0312         7.13
IVFOPQ-m32-nl316-np4                                   8_930.09       577.84     9_507.93       0.6431          1.0314            1.0287         4.65
IVFOPQ-m64-nl316-np4                                  13_218.67       759.08    13_977.75       0.7198          1.0197            1.0162         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_556.29       594.38    11_150.67       0.6786          1.0253            1.0239         7.13
IVFOPQ-m32-nl316-np8                                   8_930.09       772.50     9_702.59       0.6881          1.0238            1.0224         4.65
IVFOPQ-m64-nl316-np8                                  13_218.67     1_108.30    14_326.98       0.7781          1.0118            1.0105         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_556.29       781.03    11_337.33       0.6926          1.0231            1.0217         7.13
IVFOPQ-m32-nl316-np15                                  8_930.09     1_101.51    10_031.61       0.6937          1.0229            1.0216         4.65
IVFOPQ-m64-nl316-np15                                 13_218.67     1_715.99    14_934.67       0.7859          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_556.29     1_170.43    11_726.72       0.6937          1.0229            1.0216         7.13
IVFOPQ-m32-nl316-np17                                  8_930.09     1_195.70    10_125.79       0.6937          1.0229            1.0216         4.65
IVFOPQ-m64-nl316-np17                                 13_218.67     1_882.61    15_101.29       0.7859          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_556.29     1_202.28    11_758.57       0.6937          1.0229            1.0216         7.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.56     1_301.36     1_368.92       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_088.38       455.74    10_544.11       0.6765          1.0275            1.0242         6.32
SOAROPQ-near-np2                                      10_088.38       503.75    10_592.13       0.6833          1.0248            1.0235         6.32
SOAROPQ-near-np4                                      10_088.38       595.13    10_683.50       0.6840          1.0246            1.0234         6.32
SOAROPQ-near-np7                                      10_088.38       735.74    10_824.12       0.6839          1.0246            1.0234         6.32
SOAROPQ-near-np8                                      10_088.38       780.48    10_868.85       0.6839          1.0246            1.0234         6.32
SOAROPQ-near-np12                                     10_088.38       969.66    11_058.04       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np1                                  10_051.62       463.51    10_515.12       0.6767          1.0275            1.0242         6.32
SOAROPQ-shift0.3-np2                                  10_051.62       510.83    10_562.45       0.6832          1.0248            1.0235         6.32
SOAROPQ-shift0.3-np4                                  10_051.62       594.08    10_645.70       0.6840          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np7                                  10_051.62       735.56    10_787.18       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np8                                  10_051.62       780.94    10_832.56       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np12                                 10_051.62       971.47    11_023.09       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np1                                  10_087.44       456.79    10_544.23       0.6762          1.0277            1.0243         6.32
SOAROPQ-shift0.7-np2                                  10_087.44       530.90    10_618.34       0.6830          1.0249            1.0236         6.32
SOAROPQ-shift0.7-np4                                  10_087.44       594.50    10_681.95       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np7                                  10_087.44       734.23    10_821.67       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np8                                  10_087.44       780.20    10_867.64       0.6839          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np12                                 10_087.44       968.63    11_056.07       0.6839          1.0246            1.0234         6.32
SOAROPQ-orth1-np1                                     10_124.26       460.71    10_584.97       0.6759          1.0278            1.0243         6.32
SOAROPQ-orth1-np2                                     10_124.26       504.62    10_628.88       0.6830          1.0249            1.0236         6.32
SOAROPQ-orth1-np4                                     10_124.26       596.84    10_721.10       0.6839          1.0246            1.0234         6.32
SOAROPQ-orth1-np7                                     10_124.26       735.64    10_859.90       0.6839          1.0246            1.0234         6.32
SOAROPQ-orth1-np8                                     10_124.26       778.61    10_902.87       0.6839          1.0246            1.0234         6.32
SOAROPQ-orth1-np12                                    10_124.26       970.33    11_094.59       0.6839          1.0246            1.0234         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.51     1_294.21     1_362.72       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   8_052.72       433.64     8_486.36       0.7216          1.1712            1.0832         3.49
IVFOPQ-m64-nl111-np1                                  11_785.46       488.30    12_273.76       0.7255          1.1681            1.0809         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_188.04       446.93     9_634.97       0.8436          1.0561            1.0277         5.98
IVFOPQ-m32-nl111-np2                                   8_052.72       489.66     8_542.38       0.8515          1.0485            1.0239         3.49
IVFOPQ-m64-nl111-np2                                  11_785.46       612.46    12_397.92       0.8597          1.0453            1.0198         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_188.04       533.10     9_721.15       0.8854          1.0261            1.0173         5.98
IVFOPQ-m32-nl111-np4                                   8_052.72       641.52     8_694.25       0.8900          1.0220            1.0159         3.49
IVFOPQ-m64-nl111-np4                                  11_785.46       860.44    12_645.90       0.9006          1.0185            1.0126         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_188.04       694.06     9_882.10       0.8923          1.0213            1.0156         5.98
IVFOPQ-m32-nl111-np5                                   8_052.72       676.58     8_729.30       0.8919          1.0209            1.0155         3.49
IVFOPQ-m64-nl111-np5                                  11_785.46       992.52    12_777.98       0.9028          1.0174            1.0121         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_188.04       767.90     9_955.94       0.8927          1.0209            1.0155         5.98
IVFOPQ-m32-nl111-np8                                   8_052.72       866.79     8_919.51       0.8932          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np8                                  11_785.46     1_360.85    13_146.31       0.9042          1.0167            1.0118         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_188.04       976.88    10_164.92       0.8931          1.0205            1.0153         5.98
IVFOPQ-m32-nl111-np10                                  8_052.72       987.58     9_040.30       0.8933          1.0202            1.0152         3.49
IVFOPQ-m64-nl111-np10                                 11_785.46     1_612.28    13_397.74       0.9042          1.0167            1.0118         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_188.04     1_112.84    10_300.88       0.8932          1.0203            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   9_198.49       424.20     9_622.69       0.7234          1.1670            1.0836         3.84
IVFOPQ-m64-nl158-np1                                  13_172.68       479.68    13_652.36       0.7265          1.1644            1.0812         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_243.55       438.07    10_681.63       0.8467          1.0546            1.0256         6.32
IVFOPQ-m32-nl158-np2                                   9_198.49       478.28     9_676.76       0.8571          1.0452            1.0213         3.84
IVFOPQ-m64-nl158-np2                                  13_172.68       585.88    13_758.56       0.8632          1.0426            1.0180         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_243.55       509.14    10_752.69       0.8916          1.0239            1.0151         6.32
IVFOPQ-m32-nl158-np4                                   9_198.49       590.52     9_789.01       0.8972          1.0195            1.0134         3.84
IVFOPQ-m64-nl158-np4                                  13_172.68       803.21    13_975.89       0.9048          1.0170            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_243.55       648.65    10_892.20       0.9004          1.0186            1.0130         6.32
IVFOPQ-m32-nl158-np7                                   9_198.49       758.40     9_956.88       0.9021          1.0171            1.0125         3.84
IVFOPQ-m64-nl158-np7                                  13_172.68     1_129.51    14_302.19       0.9096          1.0145            1.0102         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_243.55       841.87    11_085.42       0.9021          1.0173            1.0126         6.32
IVFOPQ-m32-nl158-np8                                   9_198.49       814.99    10_013.48       0.9024          1.0169            1.0125         3.84
IVFOPQ-m64-nl158-np8                                  13_172.68     1_338.61    14_511.30       0.9100          1.0144            1.0101         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_243.55       903.13    11_146.68       0.9023          1.0172            1.0125         6.32
IVFOPQ-m32-nl158-np12                                  9_198.49     1_044.82    10_243.31       0.9026          1.0169            1.0124         3.84
IVFOPQ-m64-nl158-np12                                 13_172.68     1_684.01    14_856.69       0.9102          1.0143            1.0100         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_243.55     1_152.33    11_395.88       0.9025          1.0169            1.0124         6.32
IVFOPQ-m32-nl223-np1                                   8_488.28       421.36     8_909.65       0.6955          1.1880            1.1139         3.96
IVFOPQ-m64-nl223-np1                                  12_570.51       473.67    13_044.18       0.6983          1.1856            1.1114         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_887.17       429.64    10_316.80       0.8331          1.0653            1.0319         6.45
IVFOPQ-m32-nl223-np2                                   8_488.28       472.40     8_960.68       0.8441          1.0543            1.0260         3.96
IVFOPQ-m64-nl223-np2                                  12_570.51       565.04    13_135.55       0.8506          1.0516            1.0224         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_887.17       489.49    10_376.66       0.8917          1.0253            1.0151         6.45
IVFOPQ-m32-nl223-np4                                   8_488.28       574.61     9_062.89       0.8994          1.0194            1.0126         3.96
IVFOPQ-m64-nl223-np4                                  12_570.51       762.11    13_332.62       0.9080          1.0165            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_887.17       608.46    10_495.63       0.9047          1.0178            1.0117         6.45
IVFOPQ-m32-nl223-np8                                   8_488.28       783.38     9_271.66       0.9071          1.0156            1.0110         3.96
IVFOPQ-m64-nl223-np8                                  12_570.51     1_153.95    13_724.46       0.9164          1.0125            1.0084         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_887.17       840.66    10_727.83       0.9071          1.0159            1.0111         6.45
IVFOPQ-m32-nl223-np11                                  8_488.28       940.73     9_429.01       0.9075          1.0154            1.0109         3.96
IVFOPQ-m64-nl223-np11                                 12_570.51     1_445.00    14_015.51       0.9168          1.0124            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_887.17     1_018.41    10_905.57       0.9075          1.0156            1.0109         6.45
IVFOPQ-m32-nl223-np14                                  8_488.28     1_095.83     9_584.12       0.9075          1.0154            1.0109         3.96
IVFOPQ-m64-nl223-np14                                 12_570.51     1_742.83    14_313.34       0.9169          1.0123            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_887.17     1_202.64    11_089.81       0.9075          1.0154            1.0109         6.45
IVFOPQ-m32-nl316-np1                                   8_646.29       429.57     9_075.86       0.6783          1.2033            1.1315         4.65
IVFOPQ-m64-nl316-np1                                  12_894.64       469.43    13_364.07       0.6798          1.2013            1.1297         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_920.99       432.15    10_353.14       0.8247          1.0715            1.0366         7.13
IVFOPQ-m32-nl316-np2                                   8_646.29       471.52     9_117.81       0.8377          1.0586            1.0283         4.65
IVFOPQ-m64-nl316-np2                                  12_894.64       560.18    13_454.83       0.8418          1.0567            1.0253         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_920.99       484.33    10_405.32       0.8939          1.0255            1.0140         7.13
IVFOPQ-m32-nl316-np4                                   8_646.29       566.85     9_213.14       0.9032          1.0184            1.0113         4.65
IVFOPQ-m64-nl316-np4                                  12_894.64       743.96    13_638.60       0.9089          1.0165            1.0095         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_920.99       592.45    10_513.44       0.9102          1.0163            1.0103         7.13
IVFOPQ-m32-nl316-np8                                   8_646.29       756.14     9_402.43       0.9137          1.0133            1.0093         4.65
IVFOPQ-m64-nl316-np8                                  12_894.64     1_106.79    14_001.44       0.9199          1.0113            1.0076         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_920.99       809.99    10_730.98       0.9136          1.0139            1.0094         7.13
IVFOPQ-m32-nl316-np15                                  8_646.29     1_094.78     9_741.07       0.9145          1.0130            1.0092         4.65
IVFOPQ-m64-nl316-np15                                 12_894.64     1_757.93    14_652.58       0.9208          1.0110            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_920.99     1_185.47    11_106.46       0.9144          1.0131            1.0092         7.13
IVFOPQ-m32-nl316-np17                                  8_646.29     1_195.40     9_841.69       0.9145          1.0130            1.0092         4.65
IVFOPQ-m64-nl316-np17                                 12_894.64     1_951.74    14_846.39       0.9208          1.0110            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_920.99     1_296.52    11_217.51       0.9145          1.0130            1.0092         7.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.51     1_294.21     1_362.72       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_322.37       438.94    10_761.32       0.8452          1.0573            1.0241         6.32
SOAROPQ-near-np2                                      10_322.37       519.13    10_841.50       0.8917          1.0237            1.0148         6.32
SOAROPQ-near-np4                                      10_322.37       645.51    10_967.89       0.9010          1.0181            1.0129         6.32
SOAROPQ-near-np7                                      10_322.37       840.01    11_162.39       0.9023          1.0171            1.0125         6.32
SOAROPQ-near-np8                                      10_322.37       902.68    11_225.05       0.9024          1.0170            1.0125         6.32
SOAROPQ-near-np12                                     10_322.37     1_152.14    11_474.51       0.9026          1.0169            1.0124         6.32
SOAROPQ-shift0.3-np1                                  10_475.39       445.80    10_921.19       0.8485          1.0533            1.0247         6.32
SOAROPQ-shift0.3-np2                                  10_475.39       508.79    10_984.18       0.8922          1.0234            1.0148         6.32
SOAROPQ-shift0.3-np4                                  10_475.39       645.51    11_120.90       0.9007          1.0183            1.0130         6.32
SOAROPQ-shift0.3-np7                                  10_475.39       846.71    11_322.10       0.9022          1.0172            1.0126         6.32
SOAROPQ-shift0.3-np8                                  10_475.39       928.27    11_403.66       0.9024          1.0171            1.0125         6.32
SOAROPQ-shift0.3-np12                                 10_475.39     1_151.28    11_626.67       0.9025          1.0169            1.0124         6.32
SOAROPQ-shift0.7-np1                                  10_490.75       454.81    10_945.56       0.8436          1.0574            1.0265         6.32
SOAROPQ-shift0.7-np2                                  10_490.75       509.36    11_000.11       0.8909          1.0247            1.0153         6.32
SOAROPQ-shift0.7-np4                                  10_490.75       678.23    11_168.97       0.9001          1.0189            1.0131         6.32
SOAROPQ-shift0.7-np7                                  10_490.75       870.26    11_361.01       0.9021          1.0174            1.0126         6.32
SOAROPQ-shift0.7-np8                                  10_490.75       904.94    11_395.69       0.9023          1.0172            1.0125         6.32
SOAROPQ-shift0.7-np12                                 10_490.75     1_152.96    11_643.71       0.9025          1.0169            1.0124         6.32
SOAROPQ-orth1-np1                                     10_392.75       442.55    10_835.30       0.8460          1.0560            1.0248         6.32
SOAROPQ-orth1-np2                                     10_392.75       509.44    10_902.18       0.8917          1.0239            1.0149         6.32
SOAROPQ-orth1-np4                                     10_392.75       644.07    11_036.82       0.9006          1.0184            1.0130         6.32
SOAROPQ-orth1-np7                                     10_392.75       842.79    11_235.54       0.9022          1.0172            1.0125         6.32
SOAROPQ-orth1-np8                                     10_392.75       904.06    11_296.80       0.9024          1.0171            1.0125         6.32
SOAROPQ-orth1-np12                                    10_392.75     1_161.02    11_553.77       0.9025          1.0169            1.0124         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.39     1_303.48     1_376.86       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   7_667.81       426.98     8_094.79       0.7779          1.1326            1.0517         3.49
IVFOPQ-m64-nl111-np1                                  11_662.84       482.28    12_145.12       0.7817          1.1296            1.0467         5.02
SOAROPQ-orth1-m32-nl111-np1                            9_175.47       446.88     9_622.36       0.8616          1.0578            1.0267         5.98
IVFOPQ-m32-nl111-np2                                   7_667.81       487.53     8_155.34       0.8790          1.0367            1.0199         3.49
IVFOPQ-m64-nl111-np2                                  11_662.84       596.87    12_259.72       0.8858          1.0334            1.0167         5.02
SOAROPQ-orth1-m32-nl111-np2                            9_175.47       516.73     9_692.20       0.8926          1.0300            1.0180         5.98
IVFOPQ-m32-nl111-np4                                   7_667.81       603.51     8_271.32       0.8990          1.0225            1.0156         3.49
IVFOPQ-m64-nl111-np4                                  11_662.84       833.11    12_495.96       0.9071          1.0192            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np4                            9_175.47       676.18     9_851.65       0.8988          1.0242            1.0160         5.98
IVFOPQ-m32-nl111-np5                                   7_667.81       661.73     8_329.54       0.9000          1.0220            1.0153         3.49
IVFOPQ-m64-nl111-np5                                  11_662.84       953.47    12_616.31       0.9082          1.0187            1.0125         5.02
SOAROPQ-orth1-m32-nl111-np5                            9_175.47       739.16     9_914.63       0.8995          1.0234            1.0157         5.98
IVFOPQ-m32-nl111-np8                                   7_667.81       875.82     8_543.63       0.9005          1.0218            1.0152         3.49
IVFOPQ-m64-nl111-np8                                  11_662.84     1_308.55    12_971.39       0.9088          1.0185            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np8                            9_175.47       942.20    10_117.68       0.9003          1.0222            1.0153         5.98
IVFOPQ-m32-nl111-np10                                  7_667.81       961.58     8_629.39       0.9006          1.0217            1.0152         3.49
IVFOPQ-m64-nl111-np10                                 11_662.84     1_536.76    13_199.61       0.9088          1.0184            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np10                           9_175.47     1_121.04    10_296.51       0.9005          1.0220            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   9_004.86       421.37     9_426.23       0.7594          1.1510            1.0684         3.84
IVFOPQ-m64-nl158-np1                                  13_075.37       470.47    13_545.84       0.7621          1.1486            1.0644         5.36
SOAROPQ-orth1-m32-nl158-np1                           10_224.43       430.98    10_655.42       0.8568          1.0634            1.0270         6.32
IVFOPQ-m32-nl158-np2                                   9_004.86       475.20     9_480.06       0.8773          1.0396            1.0193         3.84
IVFOPQ-m64-nl158-np2                                  13_075.37       573.56    13_648.92       0.8816          1.0375            1.0169         5.36
SOAROPQ-orth1-m32-nl158-np2                           10_224.43       500.22    10_724.65       0.8979          1.0286            1.0157         6.32
IVFOPQ-m32-nl158-np4                                   9_004.86       582.39     9_587.26       0.9070          1.0189            1.0131         3.84
IVFOPQ-m64-nl158-np4                                  13_075.37       790.14    13_865.51       0.9125          1.0166            1.0111         5.36
SOAROPQ-orth1-m32-nl158-np4                           10_224.43       629.36    10_853.79       0.9073          1.0207            1.0133         6.32
IVFOPQ-m32-nl158-np7                                   9_004.86       746.21     9_751.07       0.9093          1.0176            1.0126         3.84
IVFOPQ-m64-nl158-np7                                  13_075.37     1_095.76    14_171.13       0.9151          1.0153            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np7                           10_224.43       810.14    11_034.57       0.9090          1.0184            1.0127         6.32
IVFOPQ-m32-nl158-np8                                   9_004.86       799.52     9_804.38       0.9094          1.0176            1.0126         3.84
IVFOPQ-m64-nl158-np8                                  13_075.37     1_196.13    14_271.50       0.9153          1.0152            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np8                           10_224.43       873.12    11_097.55       0.9091          1.0182            1.0127         6.32
IVFOPQ-m32-nl158-np12                                  9_004.86     1_015.42    10_020.28       0.9095          1.0176            1.0125         3.84
IVFOPQ-m64-nl158-np12                                 13_075.37     1_630.60    14_705.96       0.9154          1.0152            1.0105         5.36
SOAROPQ-orth1-m32-nl158-np12                          10_224.43     1_115.51    11_339.95       0.9094          1.0177            1.0126         6.32
IVFOPQ-m32-nl223-np1                                   8_534.97       420.03     8_954.99       0.7330          1.1767            1.0999         3.96
IVFOPQ-m64-nl223-np1                                  12_616.69       464.12    13_080.80       0.7347          1.1749            1.0978         5.49
SOAROPQ-orth1-m32-nl223-np1                            9_732.79       427.07    10_159.86       0.8494          1.0687            1.0294         6.45
IVFOPQ-m32-nl223-np2                                   8_534.97       488.66     9_023.63       0.8709          1.0451            1.0207         3.96
IVFOPQ-m64-nl223-np2                                  12_616.69       562.03    13_178.72       0.8741          1.0434            1.0185         5.49
SOAROPQ-orth1-m32-nl223-np2                            9_732.79       485.31    10_218.10       0.9000          1.0280            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   8_534.97       562.88     9_097.84       0.9099          1.0179            1.0120         3.96
IVFOPQ-m64-nl223-np4                                  12_616.69       750.29    13_366.98       0.9149          1.0162            1.0103         5.49
SOAROPQ-orth1-m32-nl223-np4                            9_732.79       596.27    10_329.06       0.9108          1.0198            1.0121         6.45
IVFOPQ-m32-nl223-np8                                   8_534.97       761.80     9_296.76       0.9134          1.0162            1.0113         3.96
IVFOPQ-m64-nl223-np8                                  12_616.69     1_121.05    13_737.74       0.9185          1.0144            1.0095         5.49
SOAROPQ-orth1-m32-nl223-np8                            9_732.79       816.42    10_549.21       0.9132          1.0168            1.0114         6.45
IVFOPQ-m32-nl223-np11                                  8_534.97       903.14     9_438.11       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np11                                 12_616.69     1_414.94    14_031.63       0.9187          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np11                           9_732.79       982.56    10_715.35       0.9135          1.0163            1.0113         6.45
IVFOPQ-m32-nl223-np14                                  8_534.97     1_053.38     9_588.35       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np14                                 12_616.69     1_704.98    14_321.66       0.9188          1.0143            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np14                           9_732.79     1_152.22    10_885.02       0.9135          1.0162            1.0112         6.45
IVFOPQ-m32-nl316-np1                                   8_846.25       427.84     9_274.09       0.7080          1.2039            1.1272         4.65
IVFOPQ-m64-nl316-np1                                  13_053.97       475.76    13_529.74       0.7097          1.2018            1.1246         6.17
SOAROPQ-orth1-m32-nl316-np1                           10_149.24       426.67    10_575.92       0.8347          1.0799            1.0371         7.13
IVFOPQ-m32-nl316-np2                                   8_846.25       473.05     9_319.30       0.8590          1.0536            1.0252         4.65
IVFOPQ-m64-nl316-np2                                  13_053.97       555.92    13_609.90       0.8632          1.0515            1.0223         6.17
SOAROPQ-orth1-m32-nl316-np2                           10_149.24       480.79    10_630.04       0.8984          1.0293            1.0154         7.13
IVFOPQ-m32-nl316-np4                                   8_846.25       569.44     9_415.69       0.9114          1.0177            1.0112         4.65
IVFOPQ-m64-nl316-np4                                  13_053.97       737.50    13_791.47       0.9173          1.0157            1.0093         6.17
SOAROPQ-orth1-m32-nl316-np4                           10_149.24       587.02    10_736.26       0.9135          1.0191            1.0112         7.13
IVFOPQ-m32-nl316-np8                                   8_846.25       750.96     9_597.21       0.9170          1.0148            1.0101         4.65
IVFOPQ-m64-nl316-np8                                  13_053.97     1_100.52    14_154.49       0.9234          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np8                           10_149.24       835.02    10_984.26       0.9167          1.0159            1.0103         7.13
IVFOPQ-m32-nl316-np15                                  8_846.25     1_088.29     9_934.54       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np15                                 13_053.97     1_726.42    14_780.39       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np15                          10_149.24     1_155.85    11_305.10       0.9173          1.0148            1.0100         7.13
IVFOPQ-m32-nl316-np17                                  8_846.25     1_193.23    10_039.48       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np17                                 13_053.97     1_921.90    14_975.87       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np17                          10_149.24     1_353.85    11_503.10       0.9174          1.0147            1.0100         7.13
-----------------------------------------------------------------------------------------------------------------------------------------------------
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
=====================================================================================================================================================
Benchmark: Sweep B: rules at nlist=158, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        73.39     1_303.48     1_376.86       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                      10_280.75       440.96    10_721.71       0.8631          1.0549            1.0252         6.32
SOAROPQ-near-np2                                      10_280.75       496.05    10_776.80       0.8998          1.0253            1.0153         6.32
SOAROPQ-near-np4                                      10_280.75       629.79    10_910.53       0.9078          1.0194            1.0131         6.32
SOAROPQ-near-np7                                      10_280.75       810.91    11_091.66       0.9092          1.0179            1.0127         6.32
SOAROPQ-near-np8                                      10_280.75       871.31    11_152.06       0.9093          1.0178            1.0126         6.32
SOAROPQ-near-np12                                     10_280.75     1_114.21    11_394.96       0.9095          1.0176            1.0126         6.32
SOAROPQ-shift0.3-np1                                  10_182.30       434.56    10_616.86       0.8624          1.0567            1.0262         6.32
SOAROPQ-shift0.3-np2                                  10_182.30       494.86    10_677.16       0.8984          1.0272            1.0156         6.32
SOAROPQ-shift0.3-np4                                  10_182.30       624.11    10_806.41       0.9069          1.0204            1.0134         6.32
SOAROPQ-shift0.3-np7                                  10_182.30       811.36    10_993.66       0.9089          1.0183            1.0127         6.32
SOAROPQ-shift0.3-np8                                  10_182.30       870.94    11_053.24       0.9091          1.0181            1.0127         6.32
SOAROPQ-shift0.3-np12                                 10_182.30     1_110.45    11_292.75       0.9094          1.0177            1.0126         6.32
SOAROPQ-shift0.7-np1                                  10_200.06       439.87    10_639.94       0.8569          1.0638            1.0280         6.32
SOAROPQ-shift0.7-np2                                  10_200.06       496.55    10_696.62       0.8954          1.0311            1.0165         6.32
SOAROPQ-shift0.7-np4                                  10_200.06       625.79    10_825.86       0.9056          1.0223            1.0137         6.32
SOAROPQ-shift0.7-np7                                  10_200.06       813.40    11_013.47       0.9085          1.0190            1.0128         6.32
SOAROPQ-shift0.7-np8                                  10_200.06       870.00    11_070.07       0.9088          1.0187            1.0128         6.32
SOAROPQ-shift0.7-np12                                 10_200.06     1_108.94    11_309.00       0.9094          1.0178            1.0126         6.32
SOAROPQ-orth1-np1                                     10_206.68       439.71    10_646.39       0.8568          1.0634            1.0270         6.32
SOAROPQ-orth1-np2                                     10_206.68       495.19    10_701.87       0.8979          1.0286            1.0157         6.32
SOAROPQ-orth1-np4                                     10_206.68       623.09    10_829.77       0.9073          1.0207            1.0133         6.32
SOAROPQ-orth1-np7                                     10_206.68       812.67    11_019.35       0.9090          1.0184            1.0127         6.32
SOAROPQ-orth1-np8                                     10_206.68       874.60    11_081.28       0.9091          1.0182            1.0127         6.32
SOAROPQ-orth1-np12                                    10_206.68     1_109.07    11_315.75       0.9094          1.0177            1.0126         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
