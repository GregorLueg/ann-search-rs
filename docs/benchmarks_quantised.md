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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.14       659.97       671.10       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.14     6_511.18     6_522.31       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.33     1_169.82     1_182.16       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    12.33    12_202.16    12_214.49       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                               79.13        85.14       164.27       0.9793          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                               79.13        96.97       176.11       0.9827          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                               79.13       131.07       210.21       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                     79.13     1_311.22     1_390.36       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                               94.60        89.49       184.09       0.9807          1.0002            1.0000         9.20
IVF-BF16-nl387-np27 (query)                               94.60       113.65       208.26       0.9828          1.0001            1.0000         9.20
IVF-BF16-nl387 (self)                                     94.60     1_159.94     1_254.55       0.9798          1.0001            1.0000         9.20
IVF-BF16-nl547-np23 (query)                              142.86        79.57       222.43       0.9768          1.0005            1.0000         9.22
IVF-BF16-nl547-np27 (query)                              142.86        88.65       231.51       0.9812          1.0002            1.0000         9.22
IVF-BF16-nl547-np33 (query)                              142.86       103.61       246.46       0.9827          1.0001            1.0000         9.22
IVF-BF16-nl547 (self)                                    142.86     1_062.80     1_205.66       0.9798          1.0001            1.0000         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.85       703.96       715.81       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.85     6_877.61     6_889.46       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   13.95     1_201.07     1_215.03       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    13.95    12_366.14    12_380.10       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                               98.35        97.86       196.22       0.8841          1.0075            1.0021         9.48
IVF-BF16-nl273-np16 (query)                               98.35       111.81       210.17       0.8867          1.0072            1.0019         9.48
IVF-BF16-nl273-np23 (query)                               98.35       153.20       251.56       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                     98.35     1_531.02     1_629.38       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              108.68        96.35       205.03       0.8861          1.0073            1.0020         9.49
IVF-BF16-nl387-np27 (query)                              108.68       123.80       232.48       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    108.68     1_261.45     1_370.12       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              145.81        85.11       230.92       0.8841          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              145.81        96.85       242.65       0.8862          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              145.81       115.76       261.57       0.8871          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    145.81     1_133.35     1_279.16       0.8852          1.0073            1.0020         9.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.24       669.48       680.72       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.24     6_241.48     6_252.72       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.23     1_197.89     1_212.12       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    14.23    11_857.81    11_872.03       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                               79.49        81.31       160.81       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                               79.49        89.27       168.77       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                               79.49       114.16       193.65       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                     79.49     1_150.67     1_230.16       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                               96.12        84.33       180.45       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387-np27 (query)                               96.12       100.18       196.30       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387 (self)                                     96.12     1_067.22     1_163.34       0.9184          1.0030            1.0021         9.20
IVF-BF16-nl547-np23 (query)                              139.74        76.07       215.81       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np27 (query)                              139.74        83.61       223.35       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np33 (query)                              139.74        94.82       234.56       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547 (self)                                    139.74       946.11     1_085.84       0.9184          1.0030            1.0021         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.23       646.67       657.91       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.23     6_200.05     6_211.28       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.92     1_155.77     1_169.69       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    13.92    12_444.84    12_458.75       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                               86.90        80.78       167.68       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                               86.90        86.68       173.58       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                               86.90       113.53       200.43       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                     86.90     1_148.00     1_234.90       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                               94.72        80.83       175.55       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387-np27 (query)                               94.72       100.90       195.62       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387 (self)                                     94.72     1_002.31     1_097.03       0.9429          1.0017            1.0009         9.20
IVF-BF16-nl547-np23 (query)                              139.65        76.27       215.91       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np27 (query)                              139.65        81.59       221.23       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np33 (query)                              139.65        94.18       233.83       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547 (self)                                    139.65       942.15     1_081.80       0.9429          1.0017            1.0009         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.77     1_213.67     1_262.44       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.77    12_075.56    12_124.33       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   56.46     5_090.35     5_146.81       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    56.46    52_262.67    52_319.13       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              177.54       261.72       439.26       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              177.54       297.75       475.29       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              177.54       401.18       578.73       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    177.54     4_074.28     4_251.82       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              231.66       268.03       499.69       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              231.66       356.59       588.25       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    231.66     3_514.90     3_746.56       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                              333.06       259.18       592.24       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                              333.06       285.51       618.57       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                              333.06       328.85       661.90       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                    333.06     3_262.27     3_595.33       0.9679          1.0005            1.0000        36.89
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.23       664.81       676.04       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.23     6_164.09     6_175.32       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.12       966.37       982.49       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     16.12     9_717.56     9_733.68       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                                83.18        61.84       145.03       0.9227          1.0021            1.0010         5.76
IVF-SQ8-nl273-np16 (query)                                83.18        69.70       152.89       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273-np23 (query)                                83.18        91.22       174.41       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273 (self)                                      83.18       883.17       966.35       0.9249          1.0018            1.0009         5.76
IVF-SQ8-nl387-np19 (query)                               101.62        63.10       164.73       0.9243          1.0020            1.0009         5.77
IVF-SQ8-nl387-np27 (query)                               101.62        80.37       182.00       0.9257          1.0018            1.0009         5.77
IVF-SQ8-nl387 (self)                                     101.62       794.99       896.62       0.9251          1.0018            1.0009         5.77
IVF-SQ8-nl547-np23 (query)                               143.40        56.70       200.11       0.9210          1.0023            1.0011         5.79
IVF-SQ8-nl547-np27 (query)                               143.40        63.48       206.88       0.9239          1.0020            1.0010         5.79
IVF-SQ8-nl547-np33 (query)                               143.40        72.87       216.27       0.9249          1.0018            1.0009         5.79
IVF-SQ8-nl547 (self)                                     143.40       728.83       872.23       0.9250          1.0018            1.0009         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.92       719.06       730.98       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.92     6_980.66     6_992.58       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    19.31       912.72       932.03       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     19.31    10_225.18    10_244.48       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               116.80        62.27       179.07       0.7379          1.0360            1.0158         5.76
IVF-SQ8-nl273-np16 (query)                               116.80        71.78       188.58       0.7387          1.0359            1.0157         5.76
IVF-SQ8-nl273-np23 (query)                               116.80        94.64       211.45       0.7385          1.0360            1.0157         5.76
IVF-SQ8-nl273 (self)                                     116.80       983.42     1_100.22       0.7376          1.0361            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               125.30        68.28       193.58       0.7370          1.0360            1.0159         5.77
IVF-SQ8-nl387-np27 (query)                               125.30        82.17       207.47       0.7373          1.0359            1.0158         5.77
IVF-SQ8-nl387 (self)                                     125.30       841.63       966.93       0.7360          1.0363            1.0158         5.77
IVF-SQ8-nl547-np23 (query)                               154.70        54.39       209.08       0.7386          1.0358            1.0161         5.79
IVF-SQ8-nl547-np27 (query)                               154.70        60.37       215.07       0.7395          1.0356            1.0156         5.79
IVF-SQ8-nl547-np33 (query)                               154.70        71.28       225.98       0.7397          1.0356            1.0155         5.79
IVF-SQ8-nl547 (self)                                     154.70       755.46       910.16       0.7379          1.0361            1.0154         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.51       669.27       680.79       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.51     6_437.34     6_448.85       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.27     1_029.95     1_047.22       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     17.27     9_957.29     9_974.56       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                                92.08        59.42       151.51       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np16 (query)                                92.08        63.62       155.70       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np23 (query)                                92.08        79.50       171.58       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273 (self)                                      92.08       785.58       877.67       0.8118          1.0175            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               100.89        58.29       159.19       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387-np27 (query)                               100.89        70.20       171.09       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387 (self)                                     100.89       691.17       792.06       0.8119          1.0175            1.0155         5.77
IVF-SQ8-nl547-np23 (query)                               146.28        54.51       200.79       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np27 (query)                               146.28        59.00       205.28       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np33 (query)                               146.28        67.50       213.78       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547 (self)                                     146.28       660.15       806.43       0.8120          1.0175            1.0155         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.39       637.81       649.20       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.39     6_179.13     6_190.53       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.14     1_019.23     1_037.37       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     18.14     9_853.22     9_871.36       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                                86.74        58.04       144.78       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np16 (query)                                86.74        62.24       148.98       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np23 (query)                                86.74        79.55       166.29       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273 (self)                                      86.74       789.34       876.09       0.7901          1.0280            1.0256         5.76
IVF-SQ8-nl387-np19 (query)                                98.49        56.53       155.02       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387-np27 (query)                                98.49        70.54       169.03       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387 (self)                                      98.49       689.25       787.75       0.7901          1.0281            1.0256         5.77
IVF-SQ8-nl547-np23 (query)                               162.86        53.80       216.66       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np27 (query)                               162.86        57.90       220.76       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np33 (query)                               162.86        65.40       228.27       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547 (self)                                     162.86       649.75       812.61       0.7903          1.0280            1.0256         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### More dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.76     1_201.98     1_250.74       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.76    11_992.57    12_041.33       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    75.16     1_159.13     1_234.29       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     75.16    12_034.44    12_109.60       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               203.17        76.45       279.63       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np16 (query)                               203.17        85.44       288.61       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np23 (query)                               203.17       107.10       310.28       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273 (self)                                     203.17       851.41     1_054.58       0.8866          1.0073            1.0059        19.59
IVF-SQ8-nl387-np19 (query)                               256.76        82.64       339.40       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387-np27 (query)                               256.76       100.48       357.25       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387 (self)                                     256.76       784.86     1_041.62       0.8866          1.0073            1.0059        19.65
IVF-SQ8-nl547-np23 (query)                               362.52        81.31       443.83       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np27 (query)                               362.52        86.97       449.49       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np33 (query)                               362.52        97.31       459.84       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547 (self)                                     362.52       774.62     1_137.15       0.8867          1.0073            1.0059        19.73
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.97       632.35       643.32       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         10.97     6_235.06     6_246.02       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.08       968.68       984.75       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               759.64        45.33       804.97       0.9300          1.0188            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              759.64        81.59       841.23       0.9644          1.0072            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              759.64       150.39       910.03       0.9830          1.0030            1.0000        38.52
HNSW-M16-ef100 (self)                                    759.64       793.91     1_553.55       0.9649          1.0096            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_476.58        48.66     1_525.25       0.9594          1.0064            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_476.58        89.38     1_565.96       0.9829          1.0037            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_476.58       160.65     1_637.23       0.9922          1.0016            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_476.58       851.70     2_328.28       0.9834          1.0043            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_619.33        54.90     1_674.23       0.9696          1.0131            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_619.33       105.48     1_724.82       0.9882          1.0069            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_619.33       177.94     1_797.27       0.9956          1.0012            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_619.33       959.92     2_579.25       0.9886          1.0053            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_670.16        61.08     1_731.23       0.9704          1.0817            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_670.16       104.42     1_774.58       0.9892          1.0158            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_670.16       183.58     1_853.74       0.9960          1.0028            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_670.16     1_008.86     2_679.02       0.9896          1.0195            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          700.10        34.24       734.34       0.8735          1.0542            1.0032        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         700.10        63.33       763.43       0.9004          1.0168            1.0020        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         700.10       116.91       817.01       0.9139          1.0064            1.0014        26.32
HNSW-SQ8U-M16-ef100 (self)                               700.10       601.60     1_301.70       0.9002          1.0143            1.0020        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_337.69        35.34     1_373.03       0.8984          1.0297            1.0021        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_337.69        65.17     1_402.86       0.9142          1.0188            1.0014        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_337.69       133.07     1_470.77       0.9206          1.0058            1.0011        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_337.69       632.95     1_970.65       0.9142          1.0160            1.0014        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_443.83        41.47     1_485.30       0.9069          1.0074            1.0017        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_443.83        71.56     1_515.39       0.9187          1.0044            1.0012        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_443.83       133.80     1_577.63       0.9229          1.0034            1.0010        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_443.83       695.41     2_139.25       0.9181          1.0050            1.0012        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_513.30        41.20     1_554.50       0.9087          1.0112            1.0016        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_513.30        76.86     1_590.17       0.9193          1.0076            1.0012        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_513.30       138.68     1_651.98       0.9234          1.0026            1.0010        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_513.30       724.25     2_237.55       0.9191          1.0079            1.0012        44.63
HNSW-SQ8U-drop0 (query)                                1_338.16        69.53     1_407.69       0.8947          1.0124            1.0024        26.32
HNSW-SQ8U-drop0.001 (query)                            1_333.82        64.45     1_398.27       0.9147          1.0060            1.0014        26.32
HNSW-SQ8U-drop0.01 (query)                             1_340.73        64.15     1_404.88       0.8988          1.0071            1.0018        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.69       695.65       707.34       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.69     6_797.96     6_809.65       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    19.83       933.69       953.52       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               877.89        49.87       927.75       0.9334          1.0214            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              877.89        89.66       967.54       0.9689          1.0079            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              877.89       167.05     1_044.94       0.9877          1.0024            1.0000        39.09
HNSW-M16-ef100 (self)                                    877.89       878.19     1_756.07       0.9690          1.0071            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_542.47        51.28     1_593.75       0.9644          1.0084            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_542.47        93.45     1_635.92       0.9871          1.0029            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_542.47       166.94     1_709.41       0.9952          1.0012            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_542.47       899.40     2_441.87       0.9873          1.0037            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_712.28        59.65     1_771.93       0.9739          1.0053            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_712.28       108.91     1_821.19       0.9908          1.0029            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_712.28       192.61     1_904.88       0.9971          1.0018            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_712.28     1_044.46     2_756.74       0.9909          1.0021            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_895.91        76.38     1_972.30       0.9760          1.0141            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_895.91       117.63     2_013.55       0.9918          1.0040            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_895.91       209.59     2_105.50       0.9974          1.0006            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_895.91     1_125.75     3_021.66       0.9916          1.0098            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          713.51        36.43       749.94       0.6866          1.0522            1.0283        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         713.51        62.95       776.47       0.7104          1.0459            1.0233        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         713.51       117.27       830.78       0.7235          1.0417            1.0204        26.32
HNSW-SQ8U-M16-ef100 (self)                               713.51       594.83     1_308.34       0.7095          1.0482            1.0235        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_374.67        35.34     1_410.01       0.7071          1.0485            1.0232        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_374.67        73.99     1_448.66       0.7237          1.0430            1.0199        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_374.67       126.92     1_501.59       0.7313          1.0404            1.0185        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_374.67       631.34     2_006.01       0.7233          1.0449            1.0199        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_488.32        40.56     1_528.87       0.7180          1.0418            1.0208        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_488.32        73.54     1_561.86       0.7303          1.0377            1.0183        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_488.32       134.84     1_623.16       0.7353          1.0368            1.0173        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_488.32       711.72     2_200.04       0.7295          1.0389            1.0181        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_560.02        41.73     1_601.75       0.7216          1.0394            1.0197        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_560.02        75.82     1_635.84       0.7328          1.0376            1.0176        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_560.02       138.24     1_698.26       0.7368          1.0363            1.0168        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_560.02       758.64     2_318.65       0.7319          1.0376            1.0174        44.63
HNSW-SQ8U-drop0 (query)                                1_364.64        62.97     1_427.61       0.6628          1.0642            1.0306        26.32
HNSW-SQ8U-drop0.001 (query)                            1_373.20        64.38     1_437.58       0.7235          1.0423            1.0201        26.32
HNSW-SQ8U-drop0.01 (query)                             1_343.76        63.40     1_407.16       0.6900          1.0529            1.0259        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.12       652.50       663.62       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.12     6_469.02     6_480.14       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.82       989.80     1_009.62       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               800.59        49.10       849.69       0.9974          1.2963            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              800.59        94.47       895.05       0.9992          1.0447            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              800.59       162.52       963.10       0.9997          1.0001            1.0000        38.52
HNSW-M16-ef100 (self)                                    800.59       821.89     1_622.48       0.9993          1.0688            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_430.52        51.42     1_481.94       0.9967          3.1669            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_430.52        89.01     1_519.53       0.9997          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_430.52       153.71     1_584.23       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_430.52       829.04     2_259.55       0.9998          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_465.41        74.54     1_539.95       0.9990          1.0001            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_465.41        93.72     1_559.13       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_465.41       158.17     1_623.58       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_465.41       857.33     2_322.74       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_519.50        64.41     1_583.91       0.9993          1.0001            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_519.50       109.37     1_628.87       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_519.50       189.88     1_709.38       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_519.50     1_043.46     2_562.96       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          754.47        35.38       789.85       0.8114          1.0309            1.0150        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         754.47        70.71       825.18       0.8120          1.0228            1.0149        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         754.47       116.53       871.00       0.8121          1.0181            1.0149        26.32
HNSW-SQ8U-M16-ef100 (self)                               754.47       608.58     1_363.05       0.8094          1.0374            1.0157        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_308.61        64.33     1_372.95       0.8142          1.0166            1.0148        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_308.61        68.09     1_376.70       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_308.61       120.97     1_429.58       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_308.61       623.46     1_932.07       0.8119          1.0313            1.0155        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_364.14        40.65     1_404.78       0.8141          1.0636            1.0148        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_364.14        69.40     1_433.54       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_364.14       125.20     1_489.33       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_364.14       648.31     2_012.44       0.8119          1.0175            1.0155        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_462.85        40.48     1_503.33       0.8145          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_462.85        69.32     1_532.17       0.8145          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_462.85       129.02     1_591.86       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_462.85       654.80     2_117.64       0.8119          1.0175            1.0155        44.63
HNSW-SQ8U-drop0 (query)                                1_283.03        63.32     1_346.35       0.8081          1.0176            1.0157        26.32
HNSW-SQ8U-drop0.001 (query)                            1_283.64        64.04     1_347.68       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-drop0.01 (query)                             1_279.85        62.52     1_342.37       0.8049          1.0195            1.0163        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.12       650.81       661.92       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.12     6_504.28     6_515.40       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.05       992.56     1_010.62       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               871.64        55.23       926.87       0.9981          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              871.64        95.39       967.03       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              871.64       171.89     1_043.53       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    871.64       933.52     1_805.16       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_567.67        54.82     1_622.50       0.9986          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_567.67        95.56     1_663.23       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_567.67       174.21     1_741.89       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_567.67       942.55     2_510.22       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_651.27        60.25     1_711.51       0.9993          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_651.27       104.88     1_756.15       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_651.27       188.00     1_839.27       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_651.27     1_034.09     2_685.36       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_729.42        62.10     1_791.52       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_729.42       113.50     1_842.92       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_729.42       201.01     1_930.43       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_729.42     1_047.03     2_776.45       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          799.01        39.12       838.13       0.7888          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         799.01        73.85       872.86       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         799.01       133.21       932.22       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100 (self)                               799.01       680.88     1_479.89       0.7896          1.0282            1.0258        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_435.42        58.41     1_493.83       0.7890          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_435.42        72.80     1_508.22       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_435.42       133.75     1_569.18       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_435.42       685.32     2_120.75       0.7897          1.0281            1.0258        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_546.38        52.45     1_598.83       0.7891          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_546.38        82.61     1_628.99       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_546.38       147.23     1_693.61       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_546.38       760.99     2_307.37       0.7897          1.0281            1.0258        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_645.32        48.56     1_693.87       0.7892          1.0267            1.0244        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_645.32        82.79     1_728.11       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_645.32       147.82     1_793.13       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_645.32       802.17     2_447.48       0.7897          1.0281            1.0258        44.63
HNSW-SQ8U-drop0 (query)                                1_433.11        72.92     1_506.03       0.7860          1.0279            1.0254        26.32
HNSW-SQ8U-drop0.001 (query)                            1_412.39        76.95     1_489.34       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-drop0.01 (query)                             1_415.90        69.84     1_485.74       0.7830          1.0294            1.0262        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        51.72     1_258.66     1_310.38       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         51.72    12_648.26    12_699.98       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    80.34     1_229.65     1_309.99       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_360.97        80.50     1_441.47       0.9916          1.0402            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_360.97       137.87     1_498.84       0.9944          1.0246            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_360.97       246.49     1_607.46       0.9964          1.0121            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_360.97     1_349.80     2_710.77       0.9951          1.0209            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_472.50       102.86     2_575.36       0.9967          1.0185            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_472.50       153.75     2_626.26       0.9979          1.0109            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_472.50       269.50     2_742.01       0.9990          1.0047            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_472.50     1_470.99     3_943.49       0.9983          1.0089            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_630.44        99.67     2_730.11       0.9980          1.0103            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_630.44       162.74     2_793.19       0.9990          1.0048            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_630.44       274.60     2_905.04       0.9996          1.0011            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_630.44     1_522.13     4_152.57       0.9989          1.0061            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_708.45        95.13     2_803.58       0.9985          1.0084            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_708.45       155.09     2_863.54       0.9994          1.0028            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_708.45       270.75     2_979.20       0.9996          1.0016            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_708.45     1_514.86     4_223.31       0.9994          1.0028            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          843.46        58.22       901.68       0.9282          1.0348            1.0038        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         843.46        70.32       913.78       0.9306          1.0198            1.0038        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         843.46       136.37       979.83       0.9319          1.0133            1.0038        40.05
HNSW-SQ8U-M16-ef100 (self)                               843.46       676.00     1_519.46       0.9302          1.0225            1.0038        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_626.44        45.19     1_671.63       0.9305          1.0312            1.0037        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_626.44        74.37     1_700.80       0.9323          1.0197            1.0036        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_626.44       133.19     1_759.63       0.9333          1.0131            1.0036        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_626.44       699.05     2_325.49       0.9321          1.0195            1.0037        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_637.45        50.81     1_688.26       0.9318          1.0214            1.0036        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_637.45        77.49     1_714.94       0.9331          1.0133            1.0036        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_637.45       139.78     1_777.22       0.9337          1.0103            1.0036        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_637.45       740.78     2_378.23       0.9330          1.0125            1.0036        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_710.15        46.44     1_756.59       0.9330          1.0129            1.0036        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_710.15        83.06     1_793.21       0.9336          1.0098            1.0036        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_710.15       138.13     1_848.29       0.9339          1.0082            1.0036        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_710.15       760.62     2_470.78       0.9333          1.0106            1.0036        58.36
HNSW-SQ8U-drop0 (query)                                1_546.42        81.32     1_627.74       0.8638          1.0511            1.0221        40.05
HNSW-SQ8U-drop0.001 (query)                            1_591.74        83.49     1_675.23       0.9326          1.0158            1.0036        40.05
HNSW-SQ8U-drop0.01 (query)                             1_600.14        84.16     1_684.30       0.9314          1.0493            1.0020        40.05
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        53.04     1_315.09     1_368.14       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         53.04    13_237.83    13_290.87       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    90.98     1_230.35     1_321.33       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_279.23        71.63     1_350.86       0.9938          1.1228            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_279.23       130.84     1_410.07       0.9972          1.0359            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_279.23       239.17     1_518.39       0.9983          1.0181            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_279.23     1_160.06     2_439.29       0.9965          1.0410            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_368.04        82.12     2_450.16       0.9947          1.0890            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_368.04       153.35     2_521.39       0.9970          1.0533            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_368.04       290.23     2_658.27       0.9979          1.0336            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_368.04     1_505.56     3_873.59       0.9971          1.0519            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_405.06        72.64     2_477.70       0.9983          1.0253            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_405.06       124.19     2_529.25       0.9992          1.0102            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_405.06       220.53     2_625.60       0.9997          1.0045            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_405.06     1_212.21     3_617.28       0.9993          1.0086            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_409.89        72.31     2_482.20       0.9989          1.0198            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_409.89       124.43     2_534.32       0.9995          1.0070            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_409.89       220.24     2_630.13       0.9997          1.0027            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_409.89     1_280.68     3_690.57       0.9993          1.0125            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          780.55        37.02       817.57       0.6636          1.4135            1.1641        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         780.55        66.59       847.14       0.6653          1.3827            1.1629        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         780.55       115.06       895.62       0.6664          1.3639            1.1622        40.05
HNSW-SQ8U-M16-ef100 (self)                               780.55       601.43     1_381.99       0.6647          1.3856            1.1636        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_464.67        39.33     1_504.01       0.6630          1.4255            1.1644        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_464.67        66.52     1_531.19       0.6647          1.3886            1.1631        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_464.67       119.39     1_584.06       0.6664          1.3591            1.1620        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_464.67       628.15     2_092.82       0.6646          1.3973            1.1634        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_516.31        44.68     1_560.99       0.6666          1.3584            1.1619        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_516.31        67.45     1_583.77       0.6672          1.3511            1.1614        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_516.31       120.54     1_636.85       0.6674          1.3479            1.1614        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_516.31       639.60     2_155.91       0.6667          1.3574            1.1618        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_749.06        48.21     1_797.27       0.6665          1.3679            1.1618        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_749.06        85.06     1_834.12       0.6669          1.3598            1.1617        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_749.06       147.31     1_896.37       0.6673          1.3499            1.1614        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_749.06       756.68     2_505.74       0.6666          1.3645            1.1619        58.36
HNSW-SQ8U-drop0 (query)                                1_595.10        81.81     1_676.91       0.6196          1.5124            1.2358        40.05
HNSW-SQ8U-drop0.001 (query)                            1_548.63        73.86     1_622.49       0.6652          1.3936            1.1626        40.05
HNSW-SQ8U-drop0.01 (query)                             1_605.68        72.83     1_678.50       0.6789          1.3424            1.1503        40.05
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.71       707.97       740.68       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.71     2_258.49     2_291.20       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                312.28       501.71       813.99       0.2581          1.1827            1.1592         1.26
Exhaustive-PQ-m16 (self)                                 312.28     1_702.51     2_014.79       0.2365          1.1998            1.1748         1.26
Exhaustive-PQ-m32 (query)                                357.70       875.35     1_233.05       0.2961          1.1446            1.1423         2.03
Exhaustive-PQ-m32 (self)                                 357.70     2_870.26     3_227.96       0.2627          1.1633            1.1601         2.03
Exhaustive-PQ-m64 (query)                                447.39     1_861.46     2_308.84       0.3611          1.1111            1.1080         3.55
Exhaustive-PQ-m64 (self)                                 447.39     6_178.00     6_625.38       0.3106          1.1303            1.1270         3.55
IVF-PQ-nl158-m16-np7 (query)                             489.97        58.94       548.91       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16-np12 (query)                            489.97        72.14       562.11       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16-np17 (query)                            489.97        81.03       570.99       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16 (self)                                  489.97       228.81       718.78       0.3056          1.1267            1.1313         3.89
IVF-PQ-nl158-m32-np7 (query)                             548.86        84.63       633.49       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32-np12 (query)                            548.86       103.44       652.30       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32-np17 (query)                            548.86       127.03       675.89       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32 (self)                                  548.86       370.97       919.83       0.4073          1.0805            1.0788         7.12
IVF-PQ-nl158-m64-np7 (query)                             628.84       129.05       757.89       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64-np12 (query)                            628.84       170.59       799.43       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64-np17 (query)                            628.84       207.27       836.11       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64 (self)                                  628.84       646.30     1_275.13       0.6343          1.0278            1.0241        13.58
IVF-PQ-nl223-m16-np11 (query)                            443.71        56.76       500.47       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16-np14 (query)                            443.71        62.39       506.10       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16-np21 (query)                            443.71        75.39       519.10       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16 (self)                                  443.71       215.45       659.16       0.3120          1.1216            1.1261         4.97
IVF-PQ-nl223-m32-np11 (query)                            496.17        90.18       586.35       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32-np14 (query)                            496.17       106.01       602.19       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32-np21 (query)                            496.17       131.28       627.45       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32 (self)                                  496.17       392.15       888.33       0.4161          1.0769            1.0757         9.21
IVF-PQ-nl223-m64-np11 (query)                            572.79       153.25       726.03       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64-np14 (query)                            572.79       179.69       752.47       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64-np21 (query)                            572.79       225.07       797.85       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64 (self)                                  572.79       711.36     1_284.15       0.6428          1.0259            1.0230        17.71
IVF-PQ-nl316-m16-np15 (query)                            467.17        62.41       529.57       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16-np17 (query)                            467.17        64.28       531.45       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16-np25 (query)                            467.17        80.32       547.48       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16 (self)                                  467.17       223.70       690.86       0.3174          1.1174            1.1224         6.51
IVF-PQ-nl316-m32-np15 (query)                            514.84       100.28       615.12       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32-np17 (query)                            514.84       108.82       623.66       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32-np25 (query)                            514.84       136.44       651.28       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32 (self)                                  514.84       409.75       924.59       0.4252          1.0725            1.0727        12.21
IVF-PQ-nl316-m64-np15 (query)                            605.84       169.01       774.86       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64-np17 (query)                            605.84       183.27       789.11       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64-np25 (query)                            605.84       245.19       851.03       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64 (self)                                  605.84       769.19     1_375.03       0.6508          1.0241            1.0218        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.26     1_362.88     1_434.13       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         71.26     4_493.09     4_564.35       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                593.36       526.88     1_120.24       0.2443          1.1297            1.1195         1.76
Exhaustive-PQ-m16 (self)                                 593.36     1_708.85     2_302.21       0.2277          1.1396            1.1265         1.76
Exhaustive-PQ-m32 (query)                                636.08       903.97     1_540.04       0.2649          1.1130            1.1155         2.53
Exhaustive-PQ-m32 (self)                                 636.08     2_947.31     3_583.39       0.2433          1.1221            1.1232         2.53
Exhaustive-PQ-m64 (query)                                722.05     1_994.45     2_716.50       0.2958          1.0991            1.1029         4.05
Exhaustive-PQ-m64 (self)                                 722.05     6_419.64     7_141.69       0.2627          1.1103            1.1143         4.05
IVF-PQ-nl158-m16-np7 (query)                             895.93        73.61       969.55       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16-np12 (query)                            895.93        88.02       983.95       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16-np17 (query)                            895.93        96.65       992.58       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16 (self)                                  895.93       238.78     1_134.71       0.2643          1.1069            1.1139         4.54
IVF-PQ-nl158-m32-np7 (query)                             937.08        93.87     1_030.95       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32-np12 (query)                            937.08       114.25     1_051.33       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32-np17 (query)                            937.08       128.49     1_065.57       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32 (self)                                  937.08       355.90     1_292.98       0.2908          1.0912            1.0946         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_034.78       146.22     1_180.99       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_034.78       182.74     1_217.51       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_034.78       216.75     1_251.53       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64 (self)                                1_034.78       657.70     1_692.48       0.3928          1.0582            1.0562        14.24
IVF-PQ-nl223-m16-np11 (query)                            790.40        75.11       865.51       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16-np14 (query)                            790.40        79.83       870.24       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16-np21 (query)                            790.40        98.81       889.21       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16 (self)                                  790.40       230.55     1_020.95       0.2682          1.1028            1.1098         5.68
IVF-PQ-nl223-m32-np11 (query)                            843.42       104.36       947.78       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32-np14 (query)                            843.42       116.37       959.79       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32-np21 (query)                            843.42       143.18       986.60       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32 (self)                                  843.42       385.69     1_229.11       0.2958          1.0879            1.0916         9.93
IVF-PQ-nl223-m64-np11 (query)                            914.10       163.86     1_077.97       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64-np14 (query)                            914.10       185.54     1_099.64       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64-np21 (query)                            914.10       239.21     1_153.32       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64 (self)                                  914.10       702.71     1_616.82       0.4000          1.0559            1.0544        18.43
IVF-PQ-nl316-m16-np15 (query)                            835.29        80.67       915.96       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16-np17 (query)                            835.29        83.27       918.56       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16-np25 (query)                            835.29        96.82       932.10       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16 (self)                                  835.29       242.26     1_077.54       0.2716          1.0994            1.1064         7.32
IVF-PQ-nl316-m32-np15 (query)                            877.96       116.99       994.94       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32-np17 (query)                            877.96       122.37     1_000.33       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32-np25 (query)                            877.96       144.85     1_022.80       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32 (self)                                  877.96       400.10     1_278.06       0.3010          1.0845            1.0888        13.02
IVF-PQ-nl316-m64-np15 (query)                            966.96       185.63     1_152.60       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64-np17 (query)                            966.96       202.39     1_169.35       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64-np25 (query)                            966.96       255.82     1_222.78       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64 (self)                                  966.96       766.95     1_733.92       0.4089          1.0523            1.0526        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.90     1_929.90     2_033.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.90     6_221.40     6_325.29       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                875.23       557.46     1_432.70       0.2345          1.1095            1.1000         2.26
Exhaustive-PQ-m16 (self)                                 875.23     1_743.60     2_618.84       0.2206          1.1180            1.1048         2.26
Exhaustive-PQ-m32 (query)                                922.21       927.83     1_850.04       0.2567          1.0943            1.0974         3.03
Exhaustive-PQ-m32 (self)                                 922.21     3_025.26     3_947.47       0.2391          1.1012            1.1021         3.03
Exhaustive-PQ-m64 (query)                              1_019.09     1_926.90     2_945.99       0.2775          1.0855            1.0910         4.55
Exhaustive-PQ-m64 (self)                               1_019.09     6_327.33     7_346.42       0.2515          1.0934            1.0980         4.55
Exhaustive-PQ-m128 (query)                             1_173.81     3_646.99     4_820.80       0.3162          1.0712            1.0752         7.61
Exhaustive-PQ-m128 (self)                              1_173.81    12_094.30    13_268.10       0.2755          1.0817            1.0859         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_287.22        84.77     1_371.99       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_287.22        99.59     1_386.81       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_287.22       109.19     1_396.41       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16 (self)                                1_287.22       261.87     1_549.09       0.2524          1.0921            1.0990         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_306.82       112.28     1_419.11       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_306.82       131.88     1_438.71       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_306.82       148.49     1_455.32       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32 (self)                                1_306.82       395.51     1_702.33       0.2635          1.0846            1.0898         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_392.52       152.11     1_544.63       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_392.52       190.02     1_582.54       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_392.52       226.04     1_618.56       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64 (self)                                1_392.52       646.73     2_039.25       0.3087          1.0667            1.0672        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_572.92       264.97     1_837.89       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_572.92       341.54     1_914.46       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_572.92       422.63     1_995.55       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128 (self)                               1_572.92     1_298.14     2_871.06       0.4648          1.0341            1.0317        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_123.32        95.30     1_218.62       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_123.32        99.63     1_222.95       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_123.32       111.10     1_234.42       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16 (self)                                1_123.32       273.01     1_396.33       0.2560          1.0886            1.0956         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_174.18       122.54     1_296.72       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_174.18       133.32     1_307.50       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_174.18       158.72     1_332.91       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32 (self)                                1_174.18       432.92     1_607.10       0.2676          1.0816            1.0868        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_262.88       177.69     1_440.58       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_262.88       197.70     1_460.58       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_262.88       264.02     1_526.90       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64 (self)                                1_262.88       751.31     2_014.19       0.3162          1.0635            1.0652        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_507.63       340.41     1_848.04       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_507.63       384.65     1_892.27       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_507.63       499.50     2_007.13       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128 (self)                               1_507.63     1_477.64     2_985.27       0.4736          1.0318            1.0306        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_206.41       102.62     1_309.04       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_206.41       102.70     1_309.11       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_206.41       119.56     1_325.97       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16 (self)                                1_206.41       296.26     1_502.68       0.2586          1.0861            1.0927         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_256.67       137.03     1_393.69       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_256.67       143.31     1_399.98       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_256.67       168.58     1_425.24       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32 (self)                                1_256.67       454.63     1_711.30       0.2711          1.0790            1.0845        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_357.83       199.83     1_557.66       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_357.83       211.46     1_569.29       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_357.83       263.93     1_621.76       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64 (self)                                1_357.83       772.11     2_129.94       0.3237          1.0605            1.0632        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_524.47       367.26     1_891.73       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_524.47       400.08     1_924.55       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_524.47       552.46     2_076.93       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128 (self)                               1_524.47     1_704.78     3_229.25       0.4840          1.0301            1.0294        48.03
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Data where the structure resides on a lower-dimensional manifold.

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.52       727.20       759.72       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.52     2_400.26     2_432.78       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                326.89       512.46       839.34       0.2932          1.2577            1.2510         1.26
Exhaustive-PQ-m16 (self)                                 326.89     1_681.46     2_008.34       0.2301          1.3863            1.3798         1.26
Exhaustive-PQ-m32 (query)                                370.07       891.53     1_261.61       0.4008          1.1658            1.1600         2.03
Exhaustive-PQ-m32 (self)                                 370.07     2_949.86     3_319.93       0.3180          1.2686            1.2616         2.03
Exhaustive-PQ-m64 (query)                                452.59     1_891.23     2_343.82       0.5384          1.0881            1.0842         3.55
Exhaustive-PQ-m64 (self)                                 452.59     6_223.77     6_676.36       0.4587          1.1480            1.1426         3.55
IVF-PQ-nl158-m16-np7 (query)                             499.26        55.65       554.91       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16-np12 (query)                            499.26        73.32       572.58       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16-np17 (query)                            499.26        88.06       587.33       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16 (self)                                  499.26       256.34       755.60       0.4271          1.1656            1.1619         3.89
IVF-PQ-nl158-m32-np7 (query)                             538.48        81.98       620.46       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32-np12 (query)                            538.48       102.63       641.11       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32-np17 (query)                            538.48       137.60       676.08       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32 (self)                                  538.48       402.87       941.35       0.6045          1.0697            1.0649         7.12
IVF-PQ-nl158-m64-np7 (query)                             623.38       131.78       755.16       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64-np12 (query)                            623.38       181.02       804.41       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64-np17 (query)                            623.38       232.49       855.87       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64 (self)                                  623.38       741.68     1_365.06       0.7984          1.0163            1.0142        13.58
IVF-PQ-nl223-m16-np11 (query)                            485.26        68.42       553.68       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16-np14 (query)                            485.26        73.97       559.24       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16-np21 (query)                            485.26        99.99       585.25       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16 (self)                                  485.26       271.09       756.35       0.4228          1.1688            1.1643         4.97
IVF-PQ-nl223-m32-np11 (query)                            523.33       100.22       623.55       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32-np14 (query)                            523.33       111.65       634.98       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32-np21 (query)                            523.33       148.12       671.45       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32 (self)                                  523.33       444.36       967.69       0.6025          1.0704            1.0658         9.21
IVF-PQ-nl223-m64-np11 (query)                            589.27       147.52       736.79       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64-np14 (query)                            589.27       165.46       754.73       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64-np21 (query)                            589.27       239.39       828.66       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64 (self)                                  589.27       761.71     1_350.98       0.8009          1.0158            1.0139        17.71
IVF-PQ-nl316-m16-np15 (query)                            466.21        73.04       539.25       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16-np17 (query)                            466.21        77.85       544.07       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16-np25 (query)                            466.21       102.86       569.08       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16 (self)                                  466.21       281.73       747.94       0.4167          1.1731            1.1687         6.51
IVF-PQ-nl316-m32-np15 (query)                            573.16       109.30       682.47       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32-np17 (query)                            573.16       120.26       693.43       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32-np25 (query)                            573.16       149.22       722.38       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32 (self)                                  573.16       451.45     1_024.62       0.6014          1.0706            1.0662        12.21
IVF-PQ-nl316-m64-np15 (query)                            680.93       178.54       859.47       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64-np17 (query)                            680.93       188.01       868.94       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64-np25 (query)                            680.93       261.40       942.33       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64 (self)                                  680.93       823.24     1_504.18       0.8029          1.0154            1.0136        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.88     1_261.86     1_329.74       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.88     4_408.15     4_476.02       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                593.81       527.52     1_121.34       0.2128          1.2291            1.2259         1.76
Exhaustive-PQ-m16 (self)                                 593.81     1_697.96     2_291.77       0.1772          1.3099            1.3107         1.76
Exhaustive-PQ-m32 (query)                                641.11       910.62     1_551.73       0.2801          1.1736            1.1700         2.53
Exhaustive-PQ-m32 (self)                                 641.11     3_010.18     3_651.29       0.2228          1.2514            1.2496         2.53
Exhaustive-PQ-m64 (query)                                794.51     2_015.16     2_809.67       0.3752          1.1186            1.1154         4.05
Exhaustive-PQ-m64 (self)                                 794.51     6_299.94     7_094.45       0.2986          1.1838            1.1817         4.05
IVF-PQ-nl158-m16-np7 (query)                             910.63        74.47       985.10       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16-np12 (query)                            910.63        84.50       995.12       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16-np17 (query)                            910.63       100.93     1_011.56       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16 (self)                                  910.63       249.39     1_160.02       0.2728          1.2080            1.2113         4.54
IVF-PQ-nl158-m32-np7 (query)                             940.24        91.91     1_032.15       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32-np12 (query)                            940.24       112.41     1_052.65       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32-np17 (query)                            940.24       135.54     1_075.77       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32 (self)                                  940.24       352.97     1_293.21       0.3902          1.1266            1.1249         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_022.65       137.84     1_160.49       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_022.65       178.37     1_201.02       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_022.65       217.86     1_240.50       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64 (self)                                1_022.65       642.46     1_665.10       0.5735          1.0546            1.0504        14.24
IVF-PQ-nl223-m16-np11 (query)                            898.80        80.11       978.91       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16-np14 (query)                            898.80        84.73       983.53       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16-np21 (query)                            898.80        99.23       998.02       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16 (self)                                  898.80       245.97     1_144.77       0.2700          1.2105            1.2146         5.68
IVF-PQ-nl223-m32-np11 (query)                            939.65       111.24     1_050.89       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32-np14 (query)                            939.65       118.23     1_057.88       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32-np21 (query)                            939.65       144.97     1_084.62       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32 (self)                                  939.65       388.32     1_327.97       0.3857          1.1288            1.1269         9.93
IVF-PQ-nl223-m64-np11 (query)                            922.31       163.96     1_086.27       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64-np14 (query)                            922.31       183.70     1_106.02       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64-np21 (query)                            922.31       235.70     1_158.02       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64 (self)                                  922.31       732.99     1_655.31       0.5729          1.0547            1.0507        18.43
IVF-PQ-nl316-m16-np15 (query)                          1_124.46       127.58     1_252.03       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16-np17 (query)                          1_124.46       116.45     1_240.90       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16-np25 (query)                          1_124.46       111.60     1_236.06       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16 (self)                                1_124.46       273.16     1_397.62       0.2616          1.2171            1.2216         7.32
IVF-PQ-nl316-m32-np15 (query)                          1_164.05       128.94     1_292.99       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32-np17 (query)                          1_164.05       137.78     1_301.82       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32-np25 (query)                          1_164.05       184.64     1_348.69       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32 (self)                                1_164.05       430.31     1_594.36       0.3754          1.1341            1.1326        13.02
IVF-PQ-nl316-m64-np15 (query)                          1_097.14       190.77     1_287.91       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64-np17 (query)                          1_097.14       207.29     1_304.43       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64-np25 (query)                          1_097.14       269.17     1_366.31       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64 (self)                                1_097.14       825.45     1_922.59       0.5696          1.0555            1.0517        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.58     1_825.46     1_925.04       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.58     6_088.32     6_187.89       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                874.38       541.82     1_416.19       0.2070          1.2190            1.2147         2.26
Exhaustive-PQ-m16 (self)                                 874.38     1_735.45     2_609.83       0.1758          1.3086            1.3089         2.26
Exhaustive-PQ-m32 (query)                                905.27       920.01     1_825.28       0.2712          1.1686            1.1636         3.03
Exhaustive-PQ-m32 (self)                                 905.27     2_994.21     3_899.48       0.2191          1.2527            1.2506         3.03
Exhaustive-PQ-m64 (query)                              1_002.68     1_910.06     2_912.73       0.3546          1.1211            1.1168         4.55
Exhaustive-PQ-m64 (self)                               1_002.68     6_297.99     7_300.67       0.2870          1.1905            1.1878         4.55
Exhaustive-PQ-m128 (query)                             1_161.76     3_924.77     5_086.53       0.4597          1.0781            1.0752         7.61
Exhaustive-PQ-m128 (self)                              1_161.76    12_044.77    13_206.54       0.3908          1.1257            1.1234         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_303.41        83.99     1_387.40       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_303.41        96.12     1_399.53       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_303.41       109.03     1_412.44       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16 (self)                                1_303.41       274.80     1_578.21       0.2567          1.2219            1.2266         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_319.71       108.24     1_427.95       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_319.71       133.36     1_453.07       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_319.71       160.43     1_480.13       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32 (self)                                1_319.71       414.47     1_734.18       0.3618          1.1419            1.1414         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_404.88       148.31     1_553.19       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_404.88       187.91     1_592.78       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_404.88       233.94     1_638.82       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64 (self)                                1_404.88       656.89     2_061.76       0.5142          1.0726            1.0686        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_554.30       256.72     1_811.03       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_554.30       346.33     1_900.64       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_554.30       438.20     1_992.51       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128 (self)                               1_554.30     1_342.16     2_896.46       0.7142          1.0238            1.0195        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_133.02        91.01     1_224.03       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_133.02        99.23     1_232.25       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_133.02       114.15     1_247.17       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16 (self)                                1_133.02       283.55     1_416.57       0.2535          1.2252            1.2304         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_160.12       120.66     1_280.78       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_160.12       136.70     1_296.82       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_160.12       169.24     1_329.37       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32 (self)                                1_160.12       448.68     1_608.80       0.3520          1.1464            1.1458        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_403.28       186.59     1_589.87       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_403.28       209.64     1_612.92       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_403.28       267.86     1_671.14       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64 (self)                                1_403.28       769.18     2_172.46       0.5029          1.0763            1.0720        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_518.15       310.41     1_828.56       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_518.15       369.73     1_887.88       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_518.15       496.23     2_014.38       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128 (self)                               1_518.15     1_517.78     3_035.93       0.7120          1.0240            1.0200        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_201.03       103.63     1_304.67       0.3584          1.1192            1.1195         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_201.03       103.91     1_304.94       0.3584          1.1192            1.1195         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_201.03       119.93     1_320.96       0.3584          1.1192            1.1195         8.13
IVF-PQ-nl316-m16 (self)                                1_201.03       296.73     1_497.77       0.2455          1.2315            1.2371         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_256.20       134.39     1_390.60       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_256.20       141.18     1_397.39       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_256.20       169.76     1_425.96       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32 (self)                                1_256.20       455.86     1_712.06       0.3334          1.1569            1.1576        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_334.31       195.77     1_530.08       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_334.31       210.25     1_544.56       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_334.31       264.54     1_598.85       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64 (self)                                1_334.31       785.05     2_119.37       0.4877          1.0813            1.0776        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_508.79       363.12     1_871.92       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_508.79       409.53     1_918.32       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_508.79       557.14     2_065.93       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128 (self)                               1_508.79     1_759.57     3_268.37       0.7100          1.0240            1.0206        48.03
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.80       718.33       751.13       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.80     2_494.54     2_527.35       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                345.63       549.34       894.97       0.7118          1.1576            1.1395         1.26
Exhaustive-PQ-m16 (self)                                 345.63     1_751.20     2_096.83       0.6210          1.2885            1.2506         1.26
Exhaustive-PQ-m32 (query)                                370.31       895.04     1_265.35       0.7717          1.0965            1.0836         2.03
Exhaustive-PQ-m32 (self)                                 370.31     2_939.55     3_309.86       0.6993          1.1778            1.1516         2.03
Exhaustive-PQ-m64 (query)                                457.04     1_880.92     2_337.96       0.8251          1.0574            1.0468         3.55
Exhaustive-PQ-m64 (self)                                 457.04     6_496.64     6_953.68       0.7675          1.1055            1.0855         3.55
IVF-PQ-nl158-m16-np7 (query)                             495.49        67.56       563.05       0.8305          1.0508            1.0429         3.89
IVF-PQ-nl158-m16-np12 (query)                            495.49        99.80       595.29       0.8312          1.0503            1.0425         3.89
IVF-PQ-nl158-m16-np17 (query)                            495.49       136.73       632.22       0.8312          1.0503            1.0425         3.89
IVF-PQ-nl158-m16 (self)                                  495.49       395.20       890.69       0.7709          1.0956            1.0803         3.89
IVF-PQ-nl158-m32-np7 (query)                             576.83       102.15       678.98       0.8763          1.0258            1.0214         7.12
IVF-PQ-nl158-m32-np12 (query)                            576.83       158.25       735.08       0.8772          1.0252            1.0211         7.12
IVF-PQ-nl158-m32-np17 (query)                            576.83       216.44       793.27       0.8773          1.0252            1.0211         7.12
IVF-PQ-nl158-m32 (self)                                  576.83       670.72     1_247.55       0.8315          1.0494            1.0404         7.12
IVF-PQ-nl158-m64-np7 (query)                             683.57       162.10       845.67       0.9051          1.0152            1.0115        13.58
IVF-PQ-nl158-m64-np12 (query)                            683.57       253.45       937.02       0.9061          1.0146            1.0113        13.58
IVF-PQ-nl158-m64-np17 (query)                            683.57       356.67     1_040.24       0.9062          1.0146            1.0113        13.58
IVF-PQ-nl158-m64 (self)                                  683.57     1_116.80     1_800.37       0.8723          1.0278            1.0215        13.58
IVF-PQ-nl223-m16-np11 (query)                            439.94        67.37       507.31       0.8437          1.0424            1.0363         4.97
IVF-PQ-nl223-m16-np14 (query)                            439.94        79.21       519.15       0.8438          1.0424            1.0363         4.97
IVF-PQ-nl223-m16-np21 (query)                            439.94       110.80       550.73       0.8438          1.0424            1.0363         4.97
IVF-PQ-nl223-m16 (self)                                  439.94       322.06       761.99       0.7859          1.0824            1.0687         4.97
IVF-PQ-nl223-m32-np11 (query)                            475.42       112.30       587.71       0.8841          1.0224            1.0184         9.21
IVF-PQ-nl223-m32-np14 (query)                            475.42       134.67       610.09       0.8842          1.0224            1.0183         9.21
IVF-PQ-nl223-m32-np21 (query)                            475.42       185.07       660.49       0.8842          1.0224            1.0183         9.21
IVF-PQ-nl223-m32 (self)                                  475.42       583.11     1_058.53       0.8399          1.0442            1.0358         9.21
IVF-PQ-nl223-m64-np11 (query)                            573.56       175.89       749.46       0.9100          1.0133            1.0101        17.71
IVF-PQ-nl223-m64-np14 (query)                            573.56       219.05       792.61       0.9102          1.0132            1.0101        17.71
IVF-PQ-nl223-m64-np21 (query)                            573.56       313.10       886.66       0.9102          1.0132            1.0101        17.71
IVF-PQ-nl223-m64 (self)                                  573.56       984.41     1_557.97       0.8764          1.0258            1.0201        17.71
IVF-PQ-nl316-m16-np15 (query)                            461.78        70.08       531.86       0.8515          1.0384            1.0325         6.51
IVF-PQ-nl316-m16-np17 (query)                            461.78        76.90       538.68       0.8516          1.0384            1.0325         6.51
IVF-PQ-nl316-m16-np25 (query)                            461.78       102.57       564.36       0.8516          1.0384            1.0324         6.51
IVF-PQ-nl316-m16 (self)                                  461.78       299.18       760.97       0.7932          1.0774            1.0628         6.51
IVF-PQ-nl316-m32-np15 (query)                            510.30       119.59       629.88       0.8886          1.0207            1.0168        12.21
IVF-PQ-nl316-m32-np17 (query)                            510.30       129.63       639.92       0.8886          1.0207            1.0168        12.21
IVF-PQ-nl316-m32-np25 (query)                            510.30       177.66       687.95       0.8887          1.0207            1.0168        12.21
IVF-PQ-nl316-m32 (self)                                  510.30       547.89     1_058.19       0.8445          1.0420            1.0330        12.21
IVF-PQ-nl316-m64-np15 (query)                            600.68       192.33       793.01       0.9133          1.0124            1.0096        23.61
IVF-PQ-nl316-m64-np17 (query)                            600.68       215.69       816.37       0.9133          1.0124            1.0096        23.61
IVF-PQ-nl316-m64-np25 (query)                            600.68       304.46       905.13       0.9134          1.0124            1.0096        23.61
IVF-PQ-nl316-m64 (self)                                  600.68       960.08     1_560.76       0.8800          1.0242            1.0186        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.62     1_369.85     1_441.47       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         71.62     4_462.13     4_533.76       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                607.96       533.64     1_141.59       0.6791          1.1977            1.1746         1.76
Exhaustive-PQ-m16 (self)                                 607.96     1_712.01     2_319.97       0.5853          1.3494            1.3061         1.76
Exhaustive-PQ-m32 (query)                                644.09       912.49     1_556.57       0.7374          1.1283            1.1129         2.53
Exhaustive-PQ-m32 (self)                                 644.09     2_979.74     3_623.83       0.6552          1.2348            1.2026         2.53
Exhaustive-PQ-m64 (query)                                728.16     1_907.92     2_636.08       0.7805          1.0879            1.0755         4.05
Exhaustive-PQ-m64 (self)                                 728.16     6_539.83     7_267.99       0.7136          1.1583            1.1336         4.05
IVF-PQ-nl158-m16-np7 (query)                             953.25        86.64     1_039.88       0.8415          1.0469            1.0377         4.54
IVF-PQ-nl158-m16-np12 (query)                            953.25       125.26     1_078.51       0.8418          1.0467            1.0375         4.54
IVF-PQ-nl158-m16-np17 (query)                            953.25       155.30     1_108.55       0.8418          1.0467            1.0375         4.54
IVF-PQ-nl158-m16 (self)                                  953.25       419.18     1_372.43       0.7803          1.0947            1.0676         4.54
IVF-PQ-nl158-m32-np7 (query)                           1_018.41       121.12     1_139.52       0.8688          1.0313            1.0244         7.77
IVF-PQ-nl158-m32-np12 (query)                          1_018.41       174.72     1_193.12       0.8692          1.0311            1.0243         7.77
IVF-PQ-nl158-m32-np17 (query)                          1_018.41       226.09     1_244.49       0.8692          1.0311            1.0243         7.77
IVF-PQ-nl158-m32 (self)                                1_018.41       663.77     1_682.17       0.8174          1.0636            1.0455         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_028.46       167.18     1_195.64       0.8922          1.0204            1.0155        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_028.46       255.49     1_283.95       0.8927          1.0202            1.0154        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_028.46       343.98     1_372.44       0.8927          1.0202            1.0154        14.24
IVF-PQ-nl158-m64 (self)                                1_028.46     1_046.14     2_074.60       0.8484          1.0422            1.0299        14.24
IVF-PQ-nl223-m16-np11 (query)                            793.50        94.42       887.92       0.8603          1.0367            1.0290         5.68
IVF-PQ-nl223-m16-np14 (query)                            793.50       110.91       904.41       0.8603          1.0367            1.0289         5.68
IVF-PQ-nl223-m16-np21 (query)                            793.50       142.06       935.57       0.8603          1.0367            1.0289         5.68
IVF-PQ-nl223-m16 (self)                                  793.50       400.43     1_193.93       0.8010          1.0770            1.0539         5.68
IVF-PQ-nl223-m32-np11 (query)                            829.19       130.71       959.89       0.8835          1.0251            1.0189         9.93
IVF-PQ-nl223-m32-np14 (query)                            829.19       153.59       982.78       0.8836          1.0251            1.0188         9.93
IVF-PQ-nl223-m32-np21 (query)                            829.19       208.07     1_037.26       0.8836          1.0251            1.0188         9.93
IVF-PQ-nl223-m32 (self)                                  829.19       596.31     1_425.50       0.8325          1.0539            1.0361         9.93
IVF-PQ-nl223-m64-np11 (query)                            916.89       194.20     1_111.09       0.9025          1.0170            1.0127        18.43
IVF-PQ-nl223-m64-np14 (query)                            916.89       277.62     1_194.51       0.9026          1.0170            1.0126        18.43
IVF-PQ-nl223-m64-np21 (query)                            916.89       335.15     1_252.04       0.9026          1.0170            1.0126        18.43
IVF-PQ-nl223-m64 (self)                                  916.89     1_030.68     1_947.57       0.8592          1.0367            1.0244        18.43
IVF-PQ-nl316-m16-np15 (query)                            871.37       100.34       971.71       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16-np17 (query)                            871.37       152.44     1_023.82       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16-np25 (query)                            871.37       152.27     1_023.65       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16 (self)                                  871.37       369.20     1_240.58       0.8168          1.0640            1.0447         7.32
IVF-PQ-nl316-m32-np15 (query)                            923.07       133.02     1_056.09       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32-np17 (query)                            923.07       149.59     1_072.67       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32-np25 (query)                            923.07       201.76     1_124.83       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32 (self)                                  923.07       579.55     1_502.63       0.8456          1.0448            1.0302        13.02
IVF-PQ-nl316-m64-np15 (query)                          1_017.26       235.41     1_252.67       0.9077          1.0150            1.0109        24.42
IVF-PQ-nl316-m64-np17 (query)                          1_017.26       250.84     1_268.10       0.9077          1.0150            1.0109        24.42
IVF-PQ-nl316-m64-np25 (query)                          1_017.26       347.00     1_364.26       0.9077          1.0150            1.0108        24.42
IVF-PQ-nl316-m64 (self)                                1_017.26     1_063.73     2_080.99       0.8670          1.0322            1.0213        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.82     1_825.92     1_925.74       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.82     6_347.24     6_447.06       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                869.01       545.77     1_414.78       0.6502          1.2419            1.2113         2.26
Exhaustive-PQ-m16 (self)                                 869.01     1_742.17     2_611.18       0.5522          1.4109            1.3575         2.26
Exhaustive-PQ-m32 (query)                                906.05       921.28     1_827.33       0.7657          1.0989            1.0852         3.03
Exhaustive-PQ-m32 (self)                                 906.05     2_998.29     3_904.34       0.6925          1.1782            1.1510         3.03
Exhaustive-PQ-m64 (query)                                999.27     2_054.55     3_053.82       0.8202          1.0558            1.0466         4.55
Exhaustive-PQ-m64 (self)                                 999.27     6_272.83     7_272.11       0.7633          1.1010            1.0854         4.55
Exhaustive-PQ-m128 (query)                             1_160.52     3_663.22     4_823.74       0.8668          1.0289            1.0236         7.61
Exhaustive-PQ-m128 (self)                              1_160.52    12_165.34    13_325.85       0.8261          1.0515            1.0424         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_417.27       102.25     1_519.52       0.8560          1.0399            1.0308         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_417.27       135.24     1_552.51       0.8562          1.0399            1.0307         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_417.27       166.05     1_583.33       0.8561          1.0399            1.0307         5.20
IVF-PQ-nl158-m16 (self)                                1_417.27       434.17     1_851.45       0.7979          1.0768            1.0566         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_429.96       131.75     1_561.70       0.9037          1.0190            1.0117         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_429.96       179.60     1_609.55       0.9039          1.0190            1.0117         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_429.96       230.22     1_660.18       0.9039          1.0190            1.0117         8.43
IVF-PQ-nl158-m32 (self)                                1_429.96       643.95     2_073.91       0.8613          1.0392            1.0217         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_392.81       180.37     1_573.19       0.9233          1.0120            1.0064        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_392.81       270.65     1_663.46       0.9235          1.0119            1.0064        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_392.81       357.53     1_750.34       0.9235          1.0119            1.0064        14.89
IVF-PQ-nl158-m64 (self)                                1_392.81     1_042.52     2_435.33       0.8877          1.0258            1.0122        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_572.10       324.01     1_896.11       0.9405          1.0068            1.0029        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_572.10       485.87     2_057.97       0.9407          1.0067            1.0029        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_572.10       659.80     2_231.90       0.9407          1.0067            1.0029        27.82
IVF-PQ-nl158-m128 (self)                               1_572.10     2_052.85     3_624.95       0.9097          1.0162            1.0065        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_140.44       104.62     1_245.06       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_140.44       122.44     1_262.88       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_140.44       155.60     1_296.04       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16 (self)                                1_140.44       401.15     1_541.59       0.8071          1.0694            1.0515         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_160.26       138.28     1_298.53       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_160.26       160.33     1_320.59       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_160.26       216.78     1_377.04       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32 (self)                                1_160.26       614.03     1_774.28       0.8679          1.0350            1.0196        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_252.49       205.99     1_458.48       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_252.49       246.00     1_498.50       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_252.49       339.79     1_592.28       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64 (self)                                1_252.49     1_005.74     2_258.23       0.8928          1.0233            1.0110        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_421.07       372.51     1_793.58       0.9442          1.0059            1.0024        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_421.07       461.90     1_882.97       0.9443          1.0059            1.0024        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_421.07       658.33     2_079.40       0.9443          1.0059            1.0024        36.14
IVF-PQ-nl223-m128 (self)                               1_421.07     2_048.76     3_469.83       0.9138          1.0145            1.0057        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_221.19       114.09     1_335.28       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_221.19       120.62     1_341.81       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_221.19       151.48     1_372.67       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16 (self)                                1_221.19       395.97     1_617.16       0.8156          1.0629            1.0467         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_253.14       148.49     1_401.63       0.9116          1.0160            1.0097        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_253.14       163.42     1_416.56       0.9116          1.0160            1.0096        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_253.14       214.88     1_468.01       0.9116          1.0160            1.0096        13.83
IVF-PQ-nl316-m32 (self)                                1_253.14       593.41     1_846.55       0.8732          1.0323            1.0180        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_363.96       229.29     1_593.25       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_363.96       246.26     1_610.22       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_363.96       340.62     1_704.57       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64 (self)                                1_363.96     1_025.77     2_389.73       0.8978          1.0211            1.0099        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_519.14       423.64     1_942.78       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_519.14       464.49     1_983.63       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_519.14       689.55     2_208.69       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128 (self)                               1_519.14     2_125.78     3_644.92       0.9179          1.0132            1.0050        48.03
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.52       685.04       717.55       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.52     2_216.86     2_249.37       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_393.21       523.20     1_916.41       0.2858          1.1535            1.1327         1.51
Exhaustive-OPQ-m16 (self)                              1_393.21     1_651.58     3_044.79       0.2593          1.1721            1.1492         1.51
Exhaustive-OPQ-m32 (query)                             1_578.35       881.00     2_459.35       0.3254          1.1214            1.1169         2.28
Exhaustive-OPQ-m32 (self)                              1_578.35     2_878.22     4_456.56       0.2827          1.1442            1.1387         2.28
Exhaustive-OPQ-m64 (query)                             1_914.39     1_860.00     3_774.39       0.3791          1.0985            1.0952         3.80
Exhaustive-OPQ-m64 (self)                              1_914.39     6_399.50     8_313.89       0.3227          1.1203            1.1168         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_536.28        62.73     1_599.01       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_536.28        73.28     1_609.56       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_536.28        86.80     1_623.08       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16 (self)                               1_536.28       241.09     1_777.37       0.3177          1.1185            1.1226         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_742.60        83.68     1_826.28       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_742.60       110.74     1_853.34       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_742.60       131.44     1_874.04       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32 (self)                               1_742.60       372.29     2_114.89       0.4166          1.0763            1.0763         7.62
IVF-OPQ-nl158-m64-np7 (query)                          2_045.23       132.52     2_177.75       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64-np12 (query)                         2_045.23       170.46     2_215.69       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64-np17 (query)                         2_045.23       205.23     2_250.46       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64 (self)                               2_045.23       643.31     2_688.54       0.6384          1.0264            1.0236        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_526.05        68.84     1_594.89       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_526.05        73.61     1_599.66       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_526.05        88.74     1_614.79       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16 (self)                               1_526.05       250.42     1_776.47       0.3219          1.1154            1.1195         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_743.56        99.21     1_842.78       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_743.56       108.33     1_851.89       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_743.56       140.85     1_884.41       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32 (self)                               1_743.56       397.90     2_141.46       0.4258          1.0726            1.0728         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_050.27       157.32     2_207.59       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_050.27       177.47     2_227.74       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_050.27       228.40     2_278.68       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64 (self)                               2_050.27       701.49     2_751.76       0.6468          1.0247            1.0224        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_544.60        77.09     1_621.69       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_544.60        81.97     1_626.57       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_544.60        92.24     1_636.84       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16 (self)                               1_544.60       256.57     1_801.17       0.3273          1.1114            1.1168         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_747.94       102.63     1_850.58       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_747.94       113.02     1_860.97       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_747.94       140.61     1_888.55       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32 (self)                               1_747.94       416.63     2_164.57       0.4330          1.0696            1.0707        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_073.30       173.95     2_247.25       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_073.30       186.60     2_259.91       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_073.30       244.24     2_317.54       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64 (self)                               2_073.30       759.06     2_832.36       0.6548          1.0232            1.0214        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.78     1_340.69     1_408.47       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.78     4_490.17     4_557.95       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_922.78       575.08     3_497.86       0.2661          1.1132            1.1009         2.76
Exhaustive-OPQ-m16 (self)                              2_922.78     1_711.39     4_634.17       0.2457          1.1248            1.1094         2.76
Exhaustive-OPQ-m32 (query)                             3_040.04       947.80     3_987.84       0.2884          1.0947            1.0956         3.53
Exhaustive-OPQ-m32 (self)                              3_040.04     2_960.83     6_000.87       0.2623          1.1062            1.1056         3.53
Exhaustive-OPQ-m64 (query)                             3_403.91     1_920.54     5_324.45       0.3208          1.0819            1.0844         5.05
Exhaustive-OPQ-m64 (self)                              3_403.91     6_177.90     9_581.81       0.2785          1.0968            1.0989         5.05
Exhaustive-OPQ-m128 (query)                            4_008.77     3_596.36     7_605.13       0.3695          1.0682            1.0688         8.11
Exhaustive-OPQ-m128 (self)                             4_008.77    11_734.92    15_743.69       0.3153          1.0826            1.0834         8.11
IVF-OPQ-nl158-m16-np7 (query)                          2_928.61        87.11     3_015.72       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16-np12 (query)                         2_928.61       100.26     3_028.87       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16-np17 (query)                         2_928.61       109.14     3_037.75       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16 (self)                               2_928.61       239.38     3_168.00       0.2765          1.0975            1.1034         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_038.64       110.67     3_149.31       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_038.64       132.06     3_170.70       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_038.64       153.57     3_192.21       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32 (self)                               3_038.64       381.40     3_420.04       0.3017          1.0856            1.0895         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_390.51       155.41     3_545.93       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_390.51       191.29     3_581.81       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_390.51       235.75     3_626.26       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64 (self)                               3_390.51       630.13     4_020.64       0.4016          1.0548            1.0547        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_072.80       277.74     4_350.54       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_072.80       350.17     4_422.97       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_072.80       428.66     4_501.46       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128 (self)                              4_072.80     1_290.54     5_363.34       0.6263          1.0191            1.0170        28.67
IVF-OPQ-nl223-m16-np11 (query)                         2_938.68        96.28     3_034.97       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16-np14 (query)                         2_938.68       100.64     3_039.33       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16-np21 (query)                         2_938.68       119.82     3_058.50       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16 (self)                               2_938.68       260.60     3_199.28       0.2771          1.0966            1.1032         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_134.08       121.59     3_255.67       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_134.08       130.83     3_264.90       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_134.08       159.00     3_293.08       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32 (self)                               3_134.08       412.09     3_546.17       0.3046          1.0836            1.0876        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_482.56       178.39     3_660.96       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_482.56       198.73     3_681.29       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_482.56       255.26     3_737.83       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64 (self)                               3_482.56       699.49     4_182.05       0.4096          1.0526            1.0530        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_179.25       324.73     4_503.98       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_179.25       374.38     4_553.63       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_179.25       494.68     4_673.93       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128 (self)                              4_179.25     1_466.09     5_645.34       0.6353          1.0180            1.0161        36.92
IVF-OPQ-nl316-m16-np15 (query)                         3_021.64       102.41     3_124.05       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16-np17 (query)                         3_021.64       102.99     3_124.63       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16-np25 (query)                         3_021.64       118.68     3_140.32       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16 (self)                               3_021.64       273.44     3_295.08       0.2802          1.0943            1.1011         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_215.54       133.07     3_348.61       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_215.54       139.08     3_354.63       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_215.54       163.54     3_379.09       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32 (self)                               3_215.54       426.02     3_641.56       0.3099          1.0809            1.0857        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_595.95       197.89     3_793.84       0.5032          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_595.95       208.93     3_804.88       0.5032          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_595.95       269.88     3_865.83       0.5032          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64 (self)                               3_595.95       759.51     4_355.46       0.4170          1.0500            1.0512        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_303.37       368.30     4_671.68       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_303.37       404.66     4_708.03       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_303.37       560.57     4_863.94       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128 (self)                              4_303.37     1_718.63     6_022.00       0.6420          1.0168            1.0155        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.36     1_838.45     1_937.81       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.36     6_236.12     6_335.48       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_528.72       644.86     5_173.58       0.2572          1.0953            1.0838         4.51
Exhaustive-OPQ-m16 (self)                              4_528.72     1_756.99     6_285.70       0.2411          1.1050            1.0895         4.51
Exhaustive-OPQ-m32 (query)                             4_661.04     1_010.56     5_671.61       0.2801          1.0776            1.0800         5.28
Exhaustive-OPQ-m32 (self)                              4_661.04     2_985.50     7_646.54       0.2567          1.0868            1.0870         5.28
Exhaustive-OPQ-m64 (query)                             4_956.10     1_965.65     6_921.74       0.2974          1.0722            1.0753         6.80
Exhaustive-OPQ-m64 (self)                              4_956.10     6_173.49    11_129.59       0.2647          1.0826            1.0853         6.80
Exhaustive-OPQ-m128 (query)                            5_548.25     3_865.00     9_413.25       0.3296          1.0636            1.0665         9.86
Exhaustive-OPQ-m128 (self)                             5_548.25    12_115.52    17_663.76       0.2830          1.0757            1.0788         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_753.02       120.13     4_873.16       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_753.02       132.20     4_885.22       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_753.02       137.03     4_890.05       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16 (self)                               4_753.02       255.35     5_008.37       0.2654          1.0828            1.0887         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_530.68       136.82     4_667.51       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_530.68       156.57     4_687.26       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_530.68       173.10     4_703.79       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32 (self)                               4_530.68       381.75     4_912.44       0.2747          1.0782            1.0828        11.43
IVF-OPQ-nl158-m64-np7 (query)                          4_898.09       184.81     5_082.90       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64-np12 (query)                         4_898.09       219.71     5_117.80       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64-np17 (query)                         4_898.09       257.98     5_156.07       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64 (self)                               4_898.09       648.65     5_546.74       0.3189          1.0629            1.0647        17.89
IVF-OPQ-nl158-m128-np7 (query)                         5_939.69       300.69     6_240.39       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128-np12 (query)                        5_939.69       370.03     6_309.73       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128-np17 (query)                        5_939.69       483.95     6_423.65       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128 (self)                              5_939.69     1_354.91     7_294.61       0.4727          1.0316            1.0308        30.82
IVF-OPQ-nl223-m16-np11 (query)                         5_225.76       124.17     5_349.93       0.3097          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16-np14 (query)                         5_225.76       126.19     5_351.95       0.3097          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16-np21 (query)                         5_225.76       142.08     5_367.83       0.3097          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16 (self)                               5_225.76       276.86     5_502.62       0.2676          1.0815            1.0878         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_756.18       148.97     4_905.15       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_756.18       166.30     4_922.48       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_756.18       184.73     4_940.91       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32 (self)                               4_756.18       425.83     5_182.01       0.2769          1.0770            1.0821        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_154.43       203.82     5_358.25       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_154.43       230.74     5_385.16       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_154.43       272.14     5_426.56       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64 (self)                               5_154.43       709.96     5_864.38       0.3253          1.0604            1.0630        22.15
IVF-OPQ-nl223-m128-np11 (query)                        5_863.97       344.44     6_208.41       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128-np14 (query)                        5_863.97       394.16     6_258.13       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128-np21 (query)                        5_863.97       515.88     6_379.85       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128 (self)                              5_863.97     1_515.75     7_379.72       0.4809          1.0301            1.0299        39.14
IVF-OPQ-nl316-m16-np15 (query)                         4_799.19       128.01     4_927.21       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16-np17 (query)                         4_799.19       132.01     4_931.21       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16-np25 (query)                         4_799.19       146.76     4_945.95       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16 (self)                               4_799.19       295.78     5_094.97       0.2683          1.0802            1.0863        11.88
IVF-OPQ-nl316-m32-np15 (query)                         4_987.85       159.43     5_147.27       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32-np17 (query)                         4_987.85       168.11     5_155.95       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32-np25 (query)                         4_987.85       194.16     5_182.01       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32 (self)                               4_987.85       448.57     5_436.41       0.2785          1.0757            1.0812        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_427.40       229.05     5_656.45       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_427.40       240.28     5_667.68       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_427.40       294.52     5_721.92       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64 (self)                               5_427.40       799.93     6_227.33       0.3306          1.0583            1.0617        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_117.23       409.11     6_526.34       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_117.23       434.86     6_552.09       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_117.23       592.25     6_709.48       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128 (self)                              6_117.23     1_748.49     7_865.72       0.4890          1.0288            1.0288        51.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.59       686.96       719.55       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.59     2_297.36     2_329.96       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_423.41       516.26     1_939.67       0.3003          1.2501            1.2435         1.51
Exhaustive-OPQ-m16 (self)                              1_423.41     1_655.36     3_078.78       0.2369          1.3780            1.3719         1.51
Exhaustive-OPQ-m32 (query)                             1_616.37       887.15     2_503.52       0.4212          1.1523            1.1471         2.28
Exhaustive-OPQ-m32 (self)                              1_616.37     2_898.06     4_514.43       0.3382          1.2478            1.2414         2.28
Exhaustive-OPQ-m64 (query)                             1_946.68     1_863.60     3_810.28       0.5666          1.0762            1.0730         3.80
Exhaustive-OPQ-m64 (self)                              1_946.68     6_097.73     8_044.41       0.4882          1.1282            1.1236         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_512.63        60.65     1_573.28       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_512.63        74.30     1_586.93       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_512.63        90.49     1_603.12       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16 (self)                               1_512.63       249.82     1_762.45       0.6228          1.0616            1.0590         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_667.71        86.09     1_753.80       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_667.71       110.21     1_777.92       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_667.71       143.55     1_811.27       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32 (self)                               1_667.71       391.00     2_058.72       0.7523          1.0251            1.0231         7.62
IVF-OPQ-nl158-m64-np7 (query)                          2_039.60       125.82     2_165.42       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64-np12 (query)                         2_039.60       172.53     2_212.13       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64-np17 (query)                         2_039.60       226.71     2_266.31       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64 (self)                               2_039.60       676.63     2_716.24       0.8324          1.0110            1.0095        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_559.69        72.00     1_631.68       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_559.69        79.12     1_638.80       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_559.69       101.93     1_661.62       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16 (self)                               1_559.69       276.78     1_836.46       0.6294          1.0591            1.0566         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_791.97        98.52     1_890.49       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_791.97       109.74     1_901.71       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_791.97       147.79     1_939.76       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32 (self)                               1_791.97       432.09     2_224.06       0.7555          1.0241            1.0224         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_065.74       150.00     2_215.74       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_065.74       171.64     2_237.38       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_065.74       236.54     2_302.28       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64 (self)                               2_065.74       736.80     2_802.54       0.8351          1.0105            1.0093        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_522.25        71.23     1_593.48       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_522.25        79.94     1_602.19       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_522.25        99.31     1_621.56       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16 (self)                               1_522.25       280.41     1_802.66       0.6361          1.0569            1.0546         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_709.20       102.24     1_811.44       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_709.20       107.68     1_816.88       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_709.20       146.05     1_855.26       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32 (self)                               1_709.20       420.01     2_129.21       0.7593          1.0233            1.0217        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_068.06       168.91     2_236.98       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_068.06       182.57     2_250.63       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_068.06       247.60     2_315.66       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64 (self)                               2_068.06       758.03     2_826.09       0.8374          1.0102            1.0089        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.27     1_266.46     1_334.73       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.27     4_361.19     4_429.45       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             3_035.38       580.12     3_615.50       0.2324          1.2133            1.2098         2.76
Exhaustive-OPQ-m16 (self)                              3_035.38     1_739.60     4_774.97       0.1872          1.2980            1.2978         2.76
Exhaustive-OPQ-m32 (query)                             3_172.86       960.29     4_133.15       0.3204          1.1503            1.1475         3.53
Exhaustive-OPQ-m32 (self)                              3_172.86     3_003.03     6_175.89       0.2587          1.2172            1.2150         3.53
Exhaustive-OPQ-m64 (query)                             3_689.44     1_951.81     5_641.25       0.4335          1.0938            1.0916         5.05
Exhaustive-OPQ-m64 (self)                              3_689.44     6_282.54     9_971.98       0.3626          1.1418            1.1394         5.05
Exhaustive-OPQ-m128 (query)                            4_156.51     3_678.95     7_835.47       0.5701          1.0489            1.0475         8.11
Exhaustive-OPQ-m128 (self)                             4_156.51    12_346.94    16_503.46       0.5007          1.0772            1.0753         8.11
IVF-OPQ-nl158-m16-np7 (query)                          2_932.06        85.84     3_017.90       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16-np12 (query)                         2_932.06        95.38     3_027.44       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16-np17 (query)                         2_932.06       112.30     3_044.36       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16 (self)                               2_932.06       248.48     3_180.53       0.4384          1.1011            1.0990         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_033.56       107.64     3_141.20       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_033.56       131.64     3_165.20       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_033.56       149.69     3_183.25       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32 (self)                               3_033.56       377.70     3_411.26       0.6101          1.0436            1.0415         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_415.96       148.89     3_564.85       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_415.96       190.98     3_606.94       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_415.96       227.92     3_643.88       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64 (self)                               3_415.96       630.79     4_046.76       0.7314          1.0197            1.0178        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_094.39       263.59     4_357.98       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_094.39       339.81     4_434.20       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_094.39       425.55     4_519.93       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128 (self)                              4_094.39     1_300.71     5_395.10       0.8113          1.0097            1.0079        28.67
IVF-OPQ-nl223-m16-np11 (query)                         2_936.27        94.62     3_030.88       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16-np14 (query)                         2_936.27        98.87     3_035.14       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16-np21 (query)                         2_936.27       115.22     3_051.49       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16 (self)                               2_936.27       257.74     3_194.01       0.4469          1.0974            1.0955         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_086.55       119.48     3_206.03       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_086.55       129.92     3_216.47       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_086.55       158.54     3_245.09       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32 (self)                               3_086.55       399.02     3_485.58       0.6182          1.0416            1.0399        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_453.56       173.50     3_627.06       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_453.56       198.65     3_652.21       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_453.56       247.07     3_700.63       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64 (self)                               3_453.56       681.39     4_134.95       0.7362          1.0189            1.0173        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_197.91       307.25     4_505.16       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_197.91       357.59     4_555.51       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_197.91       486.99     4_684.91       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128 (self)                              4_197.91     1_475.66     5_673.57       0.8131          1.0093            1.0080        36.92
IVF-OPQ-nl316-m16-np15 (query)                         2_986.48       100.06     3_086.54       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16-np17 (query)                         2_986.48       104.57     3_091.05       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16-np25 (query)                         2_986.48       117.22     3_103.70       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16 (self)                               2_986.48       282.45     3_268.93       0.4548          1.0943            1.0926         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_188.43       131.77     3_320.20       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_188.43       135.19     3_323.61       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_188.43       167.56     3_355.99       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32 (self)                               3_188.43       415.24     3_603.67       0.6243          1.0401            1.0384        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_571.70       199.49     3_771.19       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_571.70       211.84     3_783.54       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_571.70       264.76     3_836.46       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64 (self)                               3_571.70       758.16     4_329.87       0.7395          1.0183            1.0168        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_283.19       371.57     4_654.76       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_283.19       400.33     4_683.52       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_283.19       560.95     4_844.15       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128 (self)                              4_283.19     1_732.53     6_015.72       0.8145          1.0092            1.0078        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.99     1_840.21     1_940.20       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.99     6_281.12     6_381.11       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_586.76       644.21     5_230.98       0.2278          1.2029            1.1998         4.51
Exhaustive-OPQ-m16 (self)                              4_586.76     1_756.44     6_343.20       0.1860          1.2978            1.2978         4.51
Exhaustive-OPQ-m32 (query)                             4_679.55     1_017.13     5_696.68       0.3126          1.1446            1.1410         5.28
Exhaustive-OPQ-m32 (self)                              4_679.55     2_990.43     7_669.98       0.2577          1.2169            1.2141         5.28
Exhaustive-OPQ-m64 (query)                             4_973.57     1_972.05     6_945.62       0.4083          1.0976            1.0948         6.80
Exhaustive-OPQ-m64 (self)                              4_973.57     6_181.78    11_155.35       0.3475          1.1498            1.1473         6.80
Exhaustive-OPQ-m128 (query)                            5_568.55     3_650.62     9_219.18       0.5288          1.0566            1.0549         9.86
Exhaustive-OPQ-m128 (self)                             5_568.55    11_726.87    17_295.42       0.4657          1.0908            1.0889         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_403.95       111.93     4_515.88       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_403.95       124.90     4_528.85       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_403.95       138.66     4_542.61       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16 (self)                               4_403.95       265.66     4_669.61       0.4336          1.1027            1.1010         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_532.85       135.14     4_667.99       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_532.85       155.80     4_688.65       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_532.85       176.40     4_709.24       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32 (self)                               4_532.85       416.27     4_949.12       0.6051          1.0446            1.0425        11.43
IVF-OPQ-nl158-m64-np7 (query)                          4_983.61       176.50     5_160.11       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64-np12 (query)                         4_983.61       216.75     5_200.36       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64-np17 (query)                         4_983.61       264.65     5_248.26       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64 (self)                               4_983.61       679.10     5_662.71       0.7271          1.0202            1.0182        17.89
IVF-OPQ-nl158-m128-np7 (query)                         5_916.70       282.39     6_199.09       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128-np12 (query)                        5_916.70       363.88     6_280.58       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128-np17 (query)                        5_916.70       460.94     6_377.64       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128 (self)                              5_916.70     1_331.44     7_248.14       0.8051          1.0103            1.0084        30.82
IVF-OPQ-nl223-m16-np11 (query)                         4_575.63       119.38     4_695.01       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16-np14 (query)                         4_575.63       125.13     4_700.76       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16-np21 (query)                         4_575.63       145.61     4_721.24       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16 (self)                               4_575.63       284.83     4_860.45       0.4416          1.0993            1.0976         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_719.92       147.14     4_867.06       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_719.92       161.39     4_881.32       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_719.92       187.49     4_907.41       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32 (self)                               4_719.92       432.99     5_152.91       0.6076          1.0439            1.0421        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_132.58       206.10     5_338.68       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_132.58       223.59     5_356.17       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_132.58       281.65     5_414.23       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64 (self)                               5_132.58       715.46     5_848.04       0.7310          1.0196            1.0179        22.15
IVF-OPQ-nl223-m128-np11 (query)                        5_859.69       336.88     6_196.57       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128-np14 (query)                        5_859.69       387.68     6_247.37       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128-np21 (query)                        5_859.69       526.19     6_385.88       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128 (self)                              5_859.69     1_579.77     7_439.46       0.8079          1.0099            1.0083        39.14
IVF-OPQ-nl316-m16-np15 (query)                         5_093.96       129.94     5_223.90       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16-np17 (query)                         5_093.96       134.33     5_228.29       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16-np25 (query)                         5_093.96       149.90     5_243.86       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16 (self)                               5_093.96       303.46     5_397.42       0.4474          1.0969            1.0954        11.88
IVF-OPQ-nl316-m32-np15 (query)                         4_991.22       162.32     5_153.54       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32-np17 (query)                         4_991.22       171.38     5_162.61       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32-np25 (query)                         4_991.22       203.44     5_194.66       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32 (self)                               4_991.22       461.74     5_452.96       0.6111          1.0430            1.0413        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_401.35       228.85     5_630.20       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_401.35       239.96     5_641.31       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_401.35       305.87     5_707.22       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64 (self)                               5_401.35       806.92     6_208.26       0.7344          1.0190            1.0175        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_095.28       407.07     6_502.36       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_095.28       436.13     6_531.42       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_095.28       602.82     6_698.10       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128 (self)                              6_095.28     1_788.50     7_883.78       0.8096          1.0096            1.0082        51.78
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.46       695.87       728.33       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.46     2_233.62     2_266.09       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_380.13       511.48     1_891.61       0.7915          1.0818            1.0683         1.51
Exhaustive-OPQ-m16 (self)                              1_380.13     1_643.04     3_023.17       0.7235          1.1500            1.1252         1.51
Exhaustive-OPQ-m32 (query)                             1_560.16       876.66     2_436.82       0.8299          1.0537            1.0423         2.28
Exhaustive-OPQ-m32 (self)                              1_560.16     2_861.80     4_421.95       0.7758          1.0978            1.0769         2.28
Exhaustive-OPQ-m64 (query)                             1_917.49     1_861.91     3_779.39       0.8560          1.0398            1.0294         3.80
Exhaustive-OPQ-m64 (self)                              1_917.49     6_676.86     8_594.34       0.8090          1.0724            1.0538         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_648.99        74.25     1_723.24       0.8911          1.0206            1.0158         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_648.99       106.23     1_755.22       0.8920          1.0201            1.0156         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_648.99       143.26     1_792.24       0.8920          1.0200            1.0156         4.39
IVF-OPQ-nl158-m16 (self)                               1_648.99       405.93     2_054.92       0.8499          1.0393            1.0311         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_816.58       102.81     1_919.39       0.9117          1.0132            1.0097         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_816.58       149.93     1_966.52       0.9128          1.0127            1.0094         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_816.58       203.15     2_019.73       0.9128          1.0126            1.0094         7.62
IVF-OPQ-nl158-m32 (self)                               1_816.58       658.10     2_474.69       0.8783          1.0253            1.0190         7.62
IVF-OPQ-nl158-m64-np7 (query)                          2_030.67       160.85     2_191.53       0.9248          1.0100            1.0064        14.08
IVF-OPQ-nl158-m64-np12 (query)                         2_030.67       255.81     2_286.48       0.9259          1.0095            1.0062        14.08
IVF-OPQ-nl158-m64-np17 (query)                         2_030.67       356.90     2_387.58       0.9259          1.0094            1.0062        14.08
IVF-OPQ-nl158-m64 (self)                               2_030.67     1_113.88     3_144.56       0.8977          1.0182            1.0127        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_635.88        87.44     1_723.32       0.8969          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_635.88        96.42     1_732.30       0.8970          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_635.88       133.38     1_769.26       0.8970          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16 (self)                               1_635.88       383.67     2_019.55       0.8574          1.0356            1.0273         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_858.44       112.27     1_970.71       0.9154          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_858.44       137.61     1_996.05       0.9155          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_858.44       187.15     2_045.59       0.9156          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32 (self)                               1_858.44       597.18     2_455.62       0.8825          1.0234            1.0176         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_246.28       193.74     2_440.02       0.9266          1.0091            1.0060        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_246.28       232.82     2_479.10       0.9268          1.0090            1.0060        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_246.28       338.21     2_584.49       0.9268          1.0090            1.0060        18.21
IVF-OPQ-nl223-m64 (self)                               2_246.28     1_052.71     3_298.99       0.8988          1.0176            1.0123        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_559.50        79.66     1_639.17       0.9020          1.0163            1.0125         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_559.50        89.32     1_648.83       0.9021          1.0163            1.0124         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_559.50       118.22     1_677.72       0.9021          1.0163            1.0124         7.26
IVF-OPQ-nl316-m16 (self)                               1_559.50       348.65     1_908.16       0.8638          1.0327            1.0243         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_711.65       113.69     1_825.34       0.9182          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_711.65       133.58     1_845.23       0.9182          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_711.65       174.40     1_886.06       0.9183          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32 (self)                               1_711.65       558.28     2_269.94       0.8852          1.0224            1.0166        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_302.56       208.61     2_511.17       0.9288          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_302.56       233.88     2_536.44       0.9289          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_302.56       336.92     2_639.47       0.9289          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64 (self)                               2_302.56       991.91     3_294.47       0.9011          1.0166            1.0116        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.38     1_283.01     1_351.38       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.38     4_254.26     4_322.64       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_868.42       571.21     3_439.63       0.7548          1.1139            1.0983         2.76
Exhaustive-OPQ-m16 (self)                              2_868.42     1_701.79     4_570.21       0.6783          1.2043            1.1743         2.76
Exhaustive-OPQ-m32 (query)                             2_999.18       942.65     3_941.83       0.8059          1.0693            1.0573         3.53
Exhaustive-OPQ-m32 (self)                              2_999.18     2_948.33     5_947.51       0.7451          1.1246            1.1019         3.53
Exhaustive-OPQ-m64 (query)                             3_336.02     1_916.29     5_252.31       0.8410          1.0458            1.0366         5.05
Exhaustive-OPQ-m64 (self)                              3_336.02     6_166.69     9_502.71       0.7916          1.0819            1.0657         5.05
Exhaustive-OPQ-m128 (query)                            4_026.51     3_575.71     7_602.22       0.9200          1.0107            1.0070         8.11
Exhaustive-OPQ-m128 (self)                             4_026.51    11_664.00    15_690.52       0.8931          1.0194            1.0139         8.11
IVF-OPQ-nl158-m16-np7 (query)                          2_905.17        94.10     2_999.27       0.8896          1.0221            1.0167         6.04
IVF-OPQ-nl158-m16-np12 (query)                         2_905.17       126.85     3_032.02       0.8900          1.0219            1.0166         6.04
IVF-OPQ-nl158-m16-np17 (query)                         2_905.17       161.91     3_067.08       0.8900          1.0219            1.0166         6.04
IVF-OPQ-nl158-m16 (self)                               2_905.17       411.55     3_316.72       0.8455          1.0452            1.0316         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_034.73       124.48     3_159.21       0.9010          1.0174            1.0128         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_034.73       172.46     3_207.18       0.9015          1.0172            1.0126         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_034.73       227.03     3_261.75       0.9015          1.0171            1.0126         9.27
IVF-OPQ-nl158-m32 (self)                               3_034.73       631.51     3_666.23       0.8625          1.0350            1.0243         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_370.22       177.48     3_547.69       0.9110          1.0140            1.0098        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_370.22       268.88     3_639.10       0.9115          1.0137            1.0097        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_370.22       351.18     3_721.40       0.9115          1.0137            1.0097        15.74
IVF-OPQ-nl158-m64 (self)                               3_370.22     1_019.77     4_389.98       0.8769          1.0274            1.0192        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_051.39       315.86     4_367.26       0.9609          1.0028            1.0001        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_051.39       491.01     4_542.41       0.9616          1.0026            1.0000        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_051.39       666.93     4_718.33       0.9616          1.0026            1.0000        28.67
IVF-OPQ-nl158-m128 (self)                              4_051.39     2_036.68     6_088.07       0.9449          1.0056            1.0015        28.67
IVF-OPQ-nl223-m16-np11 (query)                         2_920.13       107.72     3_027.86       0.9001          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16-np14 (query)                         2_920.13       117.51     3_037.65       0.9002          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16-np21 (query)                         2_920.13       157.70     3_077.84       0.9002          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16 (self)                               2_920.13       386.68     3_306.81       0.8566          1.0385            1.0263         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_084.36       142.36     3_226.72       0.9080          1.0153            1.0109        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_084.36       166.17     3_250.53       0.9081          1.0152            1.0109        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_084.36       222.72     3_307.08       0.9081          1.0152            1.0109        11.43
IVF-OPQ-nl223-m32 (self)                               3_084.36       616.22     3_700.58       0.8697          1.0314            1.0212        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_534.13       206.06     3_740.18       0.9179          1.0117            1.0080        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_534.13       249.12     3_783.25       0.9180          1.0116            1.0080        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_534.13       340.63     3_874.75       0.9180          1.0116            1.0080        19.93
IVF-OPQ-nl223-m64 (self)                               3_534.13       984.32     4_518.44       0.8844          1.0239            1.0161        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_158.32       377.77     4_536.09       0.9664          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_158.32       458.35     4_616.67       0.9666          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_158.32       666.29     4_824.61       0.9666          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128 (self)                              4_158.32     2_024.57     6_182.89       0.9487          1.0052            1.0008        36.92
IVF-OPQ-nl316-m16-np15 (query)                         3_022.01       110.95     3_132.96       0.9059          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16-np17 (query)                         3_022.01       121.78     3_143.79       0.9060          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16-np25 (query)                         3_022.01       152.53     3_174.53       0.9060          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16 (self)                               3_022.01       371.82     3_393.83       0.8666          1.0330            1.0223         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_204.09       155.05     3_359.14       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_204.09       165.01     3_369.10       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_204.09       211.69     3_415.79       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32 (self)                               3_204.09       584.95     3_789.05       0.8798          1.0262            1.0176        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_577.34       228.65     3_805.99       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_577.34       245.50     3_822.85       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_577.34       340.83     3_918.18       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64 (self)                               3_577.34       988.78     4_566.13       0.8895          1.0216            1.0145        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_620.24       522.10     5_142.34       0.9684          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_620.24       595.39     5_215.64       0.9685          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_620.24       773.77     5_394.01       0.9685          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128 (self)                              4_620.24     2_604.15     7_224.39       0.9511          1.0049            1.0005        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.50     1_866.40     1_965.90       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.50     6_245.95     6_345.45       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_539.96       650.00     5_189.96       0.7383          1.1310            1.1123         4.51
Exhaustive-OPQ-m16 (self)                              4_539.96     1_754.11     6_294.07       0.6589          1.2299            1.1973         4.51
Exhaustive-OPQ-m32 (query)                             4_693.44     1_015.20     5_708.63       0.8490          1.0412            1.0319         5.28
Exhaustive-OPQ-m32 (self)                              4_693.44     2_987.09     7_680.53       0.8007          1.0713            1.0579         5.28
Exhaustive-OPQ-m64 (query)                             5_014.46     1_973.00     6_987.46       0.8797          1.0255            1.0187         6.80
Exhaustive-OPQ-m64 (self)                              5_014.46     6_222.48    11_236.94       0.8412          1.0441            1.0343         6.80
Exhaustive-OPQ-m128 (query)                            5_657.96     3_654.06     9_312.02       0.9050          1.0147            1.0105         9.86
Exhaustive-OPQ-m128 (self)                             5_657.96    12_288.55    17_946.51       0.8745          1.0261            1.0199         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_405.23       125.96     4_531.19       0.8940          1.0213            1.0153         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_405.23       150.89     4_556.12       0.8941          1.0212            1.0153         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_405.23       184.43     4_589.66       0.8941          1.0212            1.0153         8.20
IVF-OPQ-nl158-m16 (self)                               4_405.23       427.12     4_832.36       0.8504          1.0416            1.0289         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_507.96       158.11     4_666.07       0.9385          1.0080            1.0032        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_507.96       204.70     4_712.66       0.9387          1.0079            1.0032        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_507.96       255.56     4_763.52       0.9387          1.0079            1.0032        11.43
IVF-OPQ-nl158-m32 (self)                               4_507.96       639.37     5_147.33       0.9109          1.0168            1.0067        11.43
IVF-OPQ-nl158-m64-np7 (query)                          5_097.98       246.79     5_344.77       0.9532          1.0047            1.0009        17.89
IVF-OPQ-nl158-m64-np12 (query)                         5_097.98       312.91     5_410.89       0.9534          1.0046            1.0009        17.89
IVF-OPQ-nl158-m64-np17 (query)                         5_097.98       387.74     5_485.71       0.9534          1.0046            1.0009        17.89
IVF-OPQ-nl158-m64 (self)                               5_097.98     1_062.07     6_160.05       0.9305          1.0107            1.0029        17.89
IVF-OPQ-nl158-m128-np7 (query)                         6_127.50       351.64     6_479.14       0.9617          1.0032            1.0000        30.82
IVF-OPQ-nl158-m128-np12 (query)                        6_127.50       547.03     6_674.53       0.9620          1.0031            1.0000        30.82
IVF-OPQ-nl158-m128-np17 (query)                        6_127.50       709.30     6_836.80       0.9620          1.0031            1.0000        30.82
IVF-OPQ-nl158-m128 (self)                              6_127.50     2_072.83     8_200.34       0.9405          1.0077            1.0015        30.82
IVF-OPQ-nl223-m16-np11 (query)                         4_881.60       137.12     5_018.72       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16-np14 (query)                         4_881.60       150.76     5_032.36       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16-np21 (query)                         4_881.60       185.21     5_066.81       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16 (self)                               4_881.60       414.50     5_296.10       0.8589          1.0368            1.0255         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_858.35       169.20     5_027.55       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_858.35       196.46     5_054.81       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_858.35       250.02     5_108.36       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32 (self)                               4_858.35       629.09     5_487.43       0.9147          1.0151            1.0061        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_288.61       239.48     5_528.09       0.9547          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_288.61       274.28     5_562.90       0.9547          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_288.61       373.49     5_662.11       0.9548          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64 (self)                               5_288.61     1_044.24     6_332.86       0.9334          1.0097            1.0026        22.15
IVF-OPQ-nl223-m128-np11 (query)                        6_039.43       409.28     6_448.70       0.9641          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128-np14 (query)                        6_039.43       501.51     6_540.94       0.9642          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128-np21 (query)                        6_039.43       721.00     6_760.43       0.9642          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128 (self)                              6_039.43     2_137.21     8_176.63       0.9436          1.0066            1.0011        39.14
IVF-OPQ-nl316-m16-np15 (query)                         4_979.26       139.93     5_119.18       0.9043          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16-np17 (query)                         4_979.26       147.04     5_126.30       0.9044          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16-np25 (query)                         4_979.26       181.02     5_160.28       0.9044          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16 (self)                               4_979.26       400.94     5_380.20       0.8654          1.0330            1.0227        11.88
IVF-OPQ-nl316-m32-np15 (query)                         5_239.59       179.78     5_419.37       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32-np17 (query)                         5_239.59       193.01     5_432.59       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32-np25 (query)                         5_239.59       248.92     5_488.50       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32 (self)                               5_239.59       608.75     5_848.34       0.9189          1.0138            1.0053        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_779.04       279.06     6_058.10       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_779.04       309.29     6_088.33       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_779.04       384.01     6_163.05       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64 (self)                               5_779.04     1_068.86     6_847.90       0.9358          1.0090            1.0023        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_861.34       475.66     7_337.00       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_861.34       666.64     7_527.98       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_861.34       757.66     7_619.01       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128 (self)                              6_861.34     2_235.78     9_097.12       0.9466          1.0060            1.0008        51.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### SOAR-PQ and SOAR-OPQ

[SOAR](benchmarks_standard.md#soar) spilling on top of IVF-PQ and IVF-OPQ. On
exact full-vector search spilling loses, because probing one more cell there
costs nothing fixed. PQ changes the arithmetic a little: codes are residuals
against a cell centroid, so the ADC table differs per cell. The query part of
that table is built once per query (by GEMM over a block of queries in
`query_batch`), and the per-cell part is precomputed at build time (up to a
256 MiB budget, see `set_cell_terms_budget`), so each probed cell still pays `m * 256` additions before it scores a candidate. At
50k points and a few hundred cells that is the same order as scanning the cell
itself. Pulling the spilled candidates out of cells that are probed anyway is
the bet; whether it pays against simply probing more cells is what the tables
answer.

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
  every probed cell carries the fixed `m * 256` table cost on top of its scan.
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
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.89     1_285.59     1_354.48       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      848.40        70.72       919.12       0.3450          1.0763            1.0750         6.21
IVFPQ-m64-nl111-np1                                      902.92        96.79       999.72       0.4472          1.0523            1.0446        11.21
SOARPQ-shift0.5-m32-nl111-np1                            950.00        75.79     1_025.78       0.3331          1.1249            1.0779         8.70
IVFPQ-m32-nl111-np2                                      848.40        77.29       925.69       0.3476          1.0754            1.0744         6.21
IVFPQ-m64-nl111-np2                                      902.92       109.69     1_012.61       0.4530          1.0507            1.0437        11.21
SOARPQ-shift0.5-m32-nl111-np2                            950.00        90.65     1_040.65       0.3464          1.0763            1.0748         8.70
IVFPQ-m32-nl111-np4                                      848.40        92.50       940.90       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np4                                      902.92       130.44     1_033.36       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np4                            950.00        96.81     1_046.81       0.3477          1.0754            1.0743         8.70
IVFPQ-m32-nl111-np5                                      848.40        96.06       944.45       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np5                                      902.92       140.95     1_043.87       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np5                            950.00       103.97     1_053.96       0.3477          1.0754            1.0743         8.70
IVFPQ-m32-nl111-np8                                      848.40       110.21       958.61       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np8                                      902.92       167.61     1_070.53       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np8                            950.00       125.39     1_075.39       0.3477          1.0754            1.0743         8.70
IVFPQ-m32-nl111-np10                                     848.40       124.49       972.89       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np10                                     902.92       179.24     1_082.17       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np10                           950.00       135.11     1_085.11       0.3477          1.0754            1.0743         8.70
IVFPQ-m32-nl158-np1                                      953.94        61.88     1_015.83       0.3513          1.0716            1.0718         7.77
IVFPQ-m64-nl158-np1                                    1_065.59        79.19     1_144.78       0.4507          1.0491            1.0433        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_111.57        74.93     1_186.50       0.3257          1.1404            1.0777        10.26
IVFPQ-m32-nl158-np2                                      953.94        74.11     1_028.06       0.3582          1.0692            1.0703         7.77
IVFPQ-m64-nl158-np2                                    1_065.59       100.31     1_165.90       0.4651          1.0456            1.0411        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_111.57        81.78     1_193.34       0.3562          1.0723            1.0709        10.26
IVFPQ-m32-nl158-np4                                      953.94        89.09     1_043.03       0.3585          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np4                                    1_065.59       124.08     1_189.67       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_111.57        89.09     1_200.65       0.3582          1.0695            1.0702        10.26
IVFPQ-m32-nl158-np7                                      953.94       105.09     1_059.03       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np7                                    1_065.59       143.11     1_208.70       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_111.57       105.98     1_217.54       0.3586          1.0691            1.0701        10.26
IVFPQ-m32-nl158-np8                                      953.94       104.92     1_058.87       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np8                                    1_065.59       151.43     1_217.03       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_111.57       110.22     1_221.79       0.3586          1.0691            1.0701        10.26
IVFPQ-m32-nl158-np12                                     953.94       123.32     1_077.26       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np12                                   1_065.59       180.63     1_246.22       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_111.57       126.87     1_238.43       0.3586          1.0691            1.0701        10.26
IVFPQ-m32-nl223-np1                                      876.47        54.73       931.20       0.3569          1.0686            1.0684         9.93
IVFPQ-m64-nl223-np1                                    1_078.93        63.40     1_142.33       0.4444          1.0490            1.0434        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_356.77        63.38     1_420.15       0.3265          1.1330            1.0759        12.42
IVFPQ-m32-nl223-np2                                      876.47        66.10       942.57       0.3710          1.0638            1.0651         9.93
IVFPQ-m64-nl223-np2                                    1_078.93        82.80     1_161.73       0.4763          1.0424            1.0384        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_356.77        79.33     1_436.10       0.3662          1.0680            1.0675        12.42
IVFPQ-m32-nl223-np4                                      876.47        82.44       958.91       0.3732          1.0633            1.0645         9.93
IVFPQ-m64-nl223-np4                                    1_078.93       109.06     1_188.00       0.4831          1.0413            1.0373        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_356.77        84.99     1_441.76       0.3720          1.0645            1.0650        12.42
IVFPQ-m32-nl223-np8                                      876.47       102.69       979.16       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np8                                    1_078.93       146.07     1_225.01       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_356.77       101.57     1_458.34       0.3733          1.0632            1.0645        12.42
IVFPQ-m32-nl223-np11                                     876.47       115.14       991.61       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np11                                   1_078.93       171.35     1_250.29       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_356.77       112.02     1_468.79       0.3733          1.0632            1.0645        12.42
IVFPQ-m32-nl223-np14                                     876.47       122.17       998.64       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np14                                   1_078.93       192.29     1_271.23       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_356.77       127.50     1_484.27       0.3733          1.0632            1.0645        12.42
IVFPQ-m32-nl316-np1                                      942.40        50.56       992.95       0.3578          1.0669            1.0670        13.02
IVFPQ-m64-nl316-np1                                    1_157.66        60.19     1_217.85       0.4395          1.0477            1.0440        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_329.45        57.84     1_387.29       0.3237          1.1327            1.0722        15.52
IVFPQ-m32-nl316-np2                                      942.40        59.09     1_001.49       0.3780          1.0605            1.0626        13.02
IVFPQ-m64-nl316-np2                                    1_157.66        74.65     1_232.32       0.4801          1.0392            1.0376        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_329.45        67.19     1_396.65       0.3720          1.0646            1.0653        15.52
IVFPQ-m32-nl316-np4                                      942.40        74.42     1_016.82       0.3827          1.0595            1.0615        13.02
IVFPQ-m64-nl316-np4                                    1_157.66        98.07     1_255.73       0.4932          1.0373            1.0358        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_329.45        81.28     1_410.73       0.3789          1.0620            1.0627        15.52
IVFPQ-m32-nl316-np8                                      942.40        98.26     1_040.66       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np8                                    1_157.66       137.18     1_294.85       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_329.45        98.63     1_428.09       0.3830          1.0597            1.0615        15.52
IVFPQ-m32-nl316-np15                                     942.40       123.93     1_066.33       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np15                                   1_157.66       188.50     1_346.17       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_329.45       123.42     1_452.87       0.3833          1.0594            1.0614        15.52
IVFPQ-m32-nl316-np17                                     942.40       130.55     1_072.95       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np17                                   1_157.66       200.09     1_357.76       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_329.45       134.82     1_464.27       0.3833          1.0594            1.0614        15.52
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
Exhaustive (query)                                        68.89     1_285.59     1_354.48       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_088.86        74.68     1_163.54       0.3269          1.1388            1.0776        10.26
SOARPQ-near-np2                                        1_088.86        79.27     1_168.13       0.3565          1.0721            1.0708        10.26
SOARPQ-near-np4                                        1_088.86        89.98     1_178.85       0.3583          1.0694            1.0702        10.26
SOARPQ-near-np7                                        1_088.86       105.95     1_194.82       0.3586          1.0691            1.0701        10.26
SOARPQ-near-np8                                        1_088.86       108.28     1_197.14       0.3586          1.0691            1.0701        10.26
SOARPQ-near-np12                                       1_088.86       126.13     1_214.99       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.3-np1                                    1_111.83        70.43     1_182.26       0.3262          1.1396            1.0777        10.26
SOARPQ-shift0.3-np2                                    1_111.83        77.60     1_189.43       0.3563          1.0722            1.0709        10.26
SOARPQ-shift0.3-np4                                    1_111.83        89.01     1_200.84       0.3582          1.0694            1.0702        10.26
SOARPQ-shift0.3-np7                                    1_111.83       106.16     1_217.99       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.3-np8                                    1_111.83       108.96     1_220.79       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.3-np12                                   1_111.83       127.16     1_239.00       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.7-np1                                    1_087.17        72.18     1_159.35       0.3255          1.1411            1.0777        10.26
SOARPQ-shift0.7-np2                                    1_087.17        78.41     1_165.58       0.3562          1.0723            1.0709        10.26
SOARPQ-shift0.7-np4                                    1_087.17        90.00     1_177.17       0.3582          1.0695            1.0702        10.26
SOARPQ-shift0.7-np7                                    1_087.17       103.93     1_191.10       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.7-np8                                    1_087.17       108.35     1_195.52       0.3586          1.0691            1.0701        10.26
SOARPQ-shift0.7-np12                                   1_087.17       127.26     1_214.43       0.3586          1.0691            1.0701        10.26
SOARPQ-orth1-np1                                       1_125.44        72.88     1_198.32       0.3266          1.1399            1.0776        10.26
SOARPQ-orth1-np2                                       1_125.44        80.93     1_206.37       0.3565          1.0721            1.0708        10.26
SOARPQ-orth1-np4                                       1_125.44        89.27     1_214.71       0.3583          1.0694            1.0702        10.26
SOARPQ-orth1-np7                                       1_125.44       107.46     1_232.90       0.3586          1.0691            1.0701        10.26
SOARPQ-orth1-np8                                       1_125.44       107.15     1_232.59       0.3586          1.0691            1.0701        10.26
SOARPQ-orth1-np12                                      1_125.44       126.51     1_251.95       0.3586          1.0691            1.0701        10.26
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.74     1_318.52     1_389.26       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      914.08        73.86       987.94       0.4796          1.0770            1.0730         6.21
IVFPQ-m64-nl111-np1                                      978.06        95.13     1_073.19       0.6136          1.0402            1.0354        11.21
SOARPQ-shift0.5-m32-nl111-np1                          1_058.80        75.60     1_134.41       0.4717          1.0889            1.0754         8.70
IVFPQ-m32-nl111-np2                                      914.08        82.20       996.27       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np2                                      978.06       114.73     1_092.80       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np2                          1_058.80        83.82     1_142.62       0.4866          1.0752            1.0720         8.70
IVFPQ-m32-nl111-np4                                      914.08        89.30     1_003.37       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np4                                      978.06       125.95     1_104.01       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np4                          1_058.80        97.71     1_156.51       0.4876          1.0738            1.0717         8.70
IVFPQ-m32-nl111-np5                                      914.08        94.51     1_008.59       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np5                                      978.06       133.65     1_111.71       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np5                          1_058.80        98.77     1_157.57       0.4876          1.0738            1.0717         8.70
IVFPQ-m32-nl111-np8                                      914.08       115.34     1_029.41       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np8                                      978.06       162.91     1_140.97       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np8                          1_058.80       127.66     1_186.46       0.4876          1.0738            1.0717         8.70
IVFPQ-m32-nl111-np10                                     914.08       120.73     1_034.81       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np10                                     978.06       176.18     1_154.25       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np10                         1_058.80       149.55     1_208.35       0.4876          1.0738            1.0717         8.70
IVFPQ-m32-nl158-np1                                    1_041.77        65.02     1_106.79       0.4506          1.0868            1.0811         7.77
IVFPQ-m64-nl158-np1                                    1_101.68        81.21     1_182.89       0.5626          1.0523            1.0430        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_180.20        73.26     1_253.46       0.4773          1.0794            1.0759        10.26
IVFPQ-m32-nl158-np2                                    1_041.77        72.76     1_114.53       0.4848          1.0742            1.0725         7.77
IVFPQ-m64-nl158-np2                                    1_101.68        98.63     1_200.31       0.6213          1.0369            1.0349        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_180.20        80.76     1_260.96       0.4887          1.0735            1.0718        10.26
IVFPQ-m32-nl158-np4                                    1_041.77        86.87     1_128.64       0.4900          1.0727            1.0709         7.77
IVFPQ-m64-nl158-np4                                    1_101.68       128.79     1_230.47       0.6307          1.0350            1.0335        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_180.20        93.35     1_273.55       0.4902          1.0727            1.0709        10.26
IVFPQ-m32-nl158-np7                                    1_041.77       109.46     1_151.22       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np7                                    1_101.68       167.64     1_269.32       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_180.20       113.20     1_293.40       0.4902          1.0727            1.0708        10.26
IVFPQ-m32-nl158-np8                                    1_041.77       116.48     1_158.25       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np8                                    1_101.68       150.00     1_251.68       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_180.20       106.14     1_286.34       0.4902          1.0727            1.0708        10.26
IVFPQ-m32-nl158-np12                                   1_041.77       124.66     1_166.43       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np12                                   1_101.68       187.64     1_289.32       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_180.20       132.14     1_312.34       0.4902          1.0727            1.0708        10.26
IVFPQ-m32-nl223-np1                                      936.97        58.63       995.60       0.4063          1.1015            1.0960         9.93
IVFPQ-m64-nl223-np1                                      963.86        69.80     1_033.67       0.4889          1.0698            1.0619        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_049.13        65.98     1_115.11       0.4647          1.0813            1.0790        12.42
IVFPQ-m32-nl223-np2                                      936.97        69.07     1_006.04       0.4684          1.0788            1.0771         9.93
IVFPQ-m64-nl223-np2                                      963.86        89.50     1_053.36       0.5916          1.0430            1.0398        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_049.13        73.02     1_122.15       0.4845          1.0744            1.0728        12.42
IVFPQ-m32-nl223-np4                                      936.97        81.62     1_018.59       0.4891          1.0725            1.0711         9.93
IVFPQ-m64-nl223-np4                                      963.86       110.86     1_074.73       0.6300          1.0350            1.0334        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_049.13        87.27     1_136.41       0.4906          1.0722            1.0705        12.42
IVFPQ-m32-nl223-np8                                      936.97        98.52     1_035.49       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np8                                      963.86       143.01     1_106.87       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_049.13       100.73     1_149.87       0.4908          1.0720            1.0705        12.42
IVFPQ-m32-nl223-np11                                     936.97       112.56     1_049.52       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np11                                     963.86       164.74     1_128.61       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_049.13       115.14     1_164.28       0.4908          1.0720            1.0705        12.42
IVFPQ-m32-nl223-np14                                     936.97       124.10     1_061.07       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np14                                     963.86       184.97     1_148.83       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_049.13       124.81     1_173.94       0.4908          1.0720            1.0705        12.42
IVFPQ-m32-nl316-np1                                      938.50        54.58       993.08       0.3503          1.1213            1.1184        13.02
IVFPQ-m64-nl316-np1                                    1_070.89        64.86     1_135.75       0.3990          1.0925            1.0871        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_209.95        60.43     1_270.39       0.4261          1.0933            1.0924        15.52
IVFPQ-m32-nl316-np2                                      938.50        60.58       999.08       0.4261          1.0917            1.0908        13.02
IVFPQ-m64-nl316-np2                                    1_070.89        79.00     1_149.89       0.5212          1.0577            1.0551        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_209.95        69.70     1_279.66       0.4658          1.0802            1.0791        15.52
IVFPQ-m32-nl316-np4                                      938.50        77.20     1_015.70       0.4735          1.0772            1.0760        13.02
IVFPQ-m64-nl316-np4                                    1_070.89       104.63     1_175.52       0.6051          1.0398            1.0378        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_209.95        86.21     1_296.17       0.4837          1.0745            1.0734        15.52
IVFPQ-m32-nl316-np8                                      938.50        95.12     1_033.63       0.4869          1.0734            1.0722        13.02
IVFPQ-m64-nl316-np8                                    1_070.89       141.19     1_212.08       0.6333          1.0342            1.0327        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_209.95       104.23     1_314.18       0.4871          1.0733            1.0721        15.52
IVFPQ-m32-nl316-np15                                     938.50       125.19     1_063.69       0.4874          1.0732            1.0720        13.02
IVFPQ-m64-nl316-np15                                   1_070.89       186.63     1_257.52       0.6347          1.0339            1.0324        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_209.95       125.68     1_335.63       0.4874          1.0732            1.0720        15.52
IVFPQ-m32-nl316-np17                                     938.50       131.48     1_069.98       0.4874          1.0732            1.0720        13.02
IVFPQ-m64-nl316-np17                                   1_070.89       200.03     1_270.92       0.6347          1.0339            1.0324        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_209.95       130.20     1_340.15       0.4874          1.0732            1.0720        15.52
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
Exhaustive (query)                                        70.74     1_318.52     1_389.26       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_130.05        73.08     1_203.12       0.4766          1.0798            1.0760        10.26
SOARPQ-near-np2                                        1_130.05        77.01     1_207.05       0.4888          1.0735            1.0718        10.26
SOARPQ-near-np4                                        1_130.05        85.67     1_215.71       0.4902          1.0727            1.0709        10.26
SOARPQ-near-np7                                        1_130.05        99.87     1_229.91       0.4902          1.0727            1.0708        10.26
SOARPQ-near-np8                                        1_130.05       105.10     1_235.14       0.4902          1.0727            1.0708        10.26
SOARPQ-near-np12                                       1_130.05       132.57     1_262.62       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.3-np1                                    1_137.86        72.96     1_210.81       0.4772          1.0795            1.0759        10.26
SOARPQ-shift0.3-np2                                    1_137.86        83.58     1_221.44       0.4888          1.0735            1.0718        10.26
SOARPQ-shift0.3-np4                                    1_137.86        88.40     1_226.26       0.4902          1.0727            1.0709        10.26
SOARPQ-shift0.3-np7                                    1_137.86       104.60     1_242.45       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.3-np8                                    1_137.86       108.52     1_246.38       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.3-np12                                   1_137.86       134.32     1_272.18       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.7-np1                                    1_218.14        78.00     1_296.14       0.4772          1.0794            1.0759        10.26
SOARPQ-shift0.7-np2                                    1_218.14        82.10     1_300.24       0.4887          1.0735            1.0718        10.26
SOARPQ-shift0.7-np4                                    1_218.14        90.92     1_309.06       0.4902          1.0727            1.0709        10.26
SOARPQ-shift0.7-np7                                    1_218.14       103.93     1_322.07       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.7-np8                                    1_218.14       112.17     1_330.31       0.4902          1.0727            1.0708        10.26
SOARPQ-shift0.7-np12                                   1_218.14       135.24     1_353.38       0.4902          1.0727            1.0708        10.26
SOARPQ-orth1-np1                                       1_166.64        72.77     1_239.41       0.4770          1.0795            1.0760        10.26
SOARPQ-orth1-np2                                       1_166.64        77.13     1_243.76       0.4887          1.0735            1.0718        10.26
SOARPQ-orth1-np4                                       1_166.64        86.71     1_253.35       0.4902          1.0727            1.0709        10.26
SOARPQ-orth1-np7                                       1_166.64       104.69     1_271.33       0.4902          1.0727            1.0708        10.26
SOARPQ-orth1-np8                                       1_166.64       104.92     1_271.55       0.4902          1.0727            1.0708        10.26
SOARPQ-orth1-np12                                      1_166.64       126.36     1_292.99       0.4902          1.0727            1.0708        10.26
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.38     1_327.26     1_395.64       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      822.56        52.23       874.79       0.6957          1.1946            1.1094         6.21
IVFPQ-m64-nl111-np1                                      906.08        66.29       972.38       0.7084          1.1834            1.0985        11.21
SOARPQ-shift0.5-m32-nl111-np1                            944.18        64.52     1_008.70       0.8094          1.0790            1.0499         8.71
IVFPQ-m32-nl111-np2                                      822.56        66.73       889.29       0.8166          1.0678            1.0432         6.21
IVFPQ-m64-nl111-np2                                      906.08        93.93     1_000.02       0.8398          1.0553            1.0292        11.21
SOARPQ-shift0.5-m32-nl111-np2                            944.18        87.29     1_031.47       0.8469          1.0463            1.0348         8.71
IVFPQ-m32-nl111-np4                                      822.56        94.95       917.51       0.8524          1.0396            1.0315         6.21
IVFPQ-m64-nl111-np4                                      906.08       140.95     1_047.03       0.8810          1.0262            1.0192        11.21
SOARPQ-shift0.5-m32-nl111-np4                            944.18       128.44     1_072.62       0.8546          1.0397            1.0314         8.71
IVFPQ-m32-nl111-np5                                      822.56       112.11       934.68       0.8548          1.0382            1.0308         6.21
IVFPQ-m64-nl111-np5                                      906.08       165.83     1_071.91       0.8834          1.0248            1.0185        11.21
SOARPQ-shift0.5-m32-nl111-np5                            944.18       148.28     1_092.46       0.8552          1.0389            1.0310         8.71
IVFPQ-m32-nl111-np8                                      822.56       153.51       976.08       0.8559          1.0375            1.0303         6.21
IVFPQ-m64-nl111-np8                                      906.08       236.66     1_142.75       0.8847          1.0241            1.0182        11.21
SOARPQ-shift0.5-m32-nl111-np8                            944.18       198.83     1_143.01       0.8559          1.0378            1.0304         8.71
IVFPQ-m32-nl111-np10                                     822.56       183.94     1_006.50       0.8560          1.0375            1.0303         6.21
IVFPQ-m64-nl111-np10                                     906.08       284.86     1_190.95       0.8848          1.0240            1.0182        11.21
SOARPQ-shift0.5-m32-nl111-np10                           944.18       220.01     1_164.18       0.8559          1.0376            1.0303         8.71
IVFPQ-m32-nl158-np1                                      995.75        47.98     1_043.73       0.6948          1.1931            1.1102         7.77
IVFPQ-m64-nl158-np1                                    1_263.00        59.02     1_322.02       0.7041          1.1842            1.1019        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_242.78        58.16     1_300.94       0.8148          1.0760            1.0451        10.29
IVFPQ-m32-nl158-np2                                      995.75        58.94     1_054.69       0.8250          1.0626            1.0379         7.77
IVFPQ-m64-nl158-np2                                    1_263.00        79.43     1_342.43       0.8439          1.0526            1.0271        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_242.78        74.86     1_317.64       0.8575          1.0419            1.0289        10.29
IVFPQ-m32-nl158-np4                                      995.75        81.21     1_076.97       0.8648          1.0337            1.0256         7.77
IVFPQ-m64-nl158-np4                                    1_263.00       115.35     1_378.35       0.8875          1.0231            1.0166        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_242.78       105.10     1_347.88       0.8669          1.0346            1.0257        10.29
IVFPQ-m32-nl158-np7                                      995.75       112.95     1_108.71       0.8688          1.0313            1.0244         7.77
IVFPQ-m64-nl158-np7                                    1_263.00       169.60     1_432.60       0.8922          1.0204            1.0155        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_242.78       147.73     1_390.51       0.8685          1.0323            1.0248        10.29
IVFPQ-m32-nl158-np8                                      995.75       123.19     1_118.95       0.8690          1.0312            1.0243         7.77
IVFPQ-m64-nl158-np8                                    1_263.00       190.05     1_453.05       0.8925          1.0203            1.0155        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_242.78       160.26     1_403.04       0.8688          1.0319            1.0245        10.29
IVFPQ-m32-nl158-np12                                     995.75       159.26     1_155.02       0.8692          1.0311            1.0243         7.77
IVFPQ-m64-nl158-np12                                   1_263.00       255.92     1_518.92       0.8927          1.0202            1.0154        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_242.78       201.32     1_444.10       0.8691          1.0313            1.0243        10.29
IVFPQ-m32-nl223-np1                                      872.91        48.07       920.98       0.6921          1.1905            1.1182         9.93
IVFPQ-m64-nl223-np1                                      976.59        57.91     1_034.49       0.6987          1.1833            1.1111        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_025.73        53.97     1_079.70       0.8163          1.0795            1.0457        12.44
IVFPQ-m32-nl223-np2                                      872.91        55.42       928.33       0.8302          1.0608            1.0354         9.93
IVFPQ-m64-nl223-np2                                      976.59        71.61     1_048.19       0.8444          1.0531            1.0266        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_025.73        66.35     1_092.08       0.8673          1.0408            1.0255        12.44
IVFPQ-m32-nl223-np4                                      872.91        71.70       944.61       0.8770          1.0286            1.0206         9.93
IVFPQ-m64-nl223-np4                                      976.59       107.86     1_084.45       0.8951          1.0206            1.0142        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_025.73        90.62     1_116.35       0.8788          1.0317            1.0210        12.44
IVFPQ-m32-nl223-np8                                      872.91       107.30       980.21       0.8831          1.0253            1.0190         9.93
IVFPQ-m64-nl223-np8                                      976.59       152.08     1_128.66       0.9021          1.0172            1.0128        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_025.73       133.35     1_159.08       0.8825          1.0270            1.0194        12.44
IVFPQ-m32-nl223-np11                                     872.91       134.06     1_006.97       0.8835          1.0251            1.0189         9.93
IVFPQ-m64-nl223-np11                                     976.59       196.72     1_173.30       0.9025          1.0170            1.0127        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_025.73       166.33     1_192.06       0.8832          1.0259            1.0191        12.44
IVFPQ-m32-nl223-np14                                     872.91       152.02     1_024.93       0.8836          1.0251            1.0188         9.93
IVFPQ-m64-nl223-np14                                     976.59       237.61     1_214.20       0.9026          1.0170            1.0126        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_025.73       195.15     1_220.88       0.8835          1.0254            1.0189        12.44
IVFPQ-m32-nl316-np1                                      930.81        49.20       980.01       0.6743          1.2059            1.1402        13.02
IVFPQ-m64-nl316-np1                                    1_041.26        57.24     1_098.50       0.6787          1.2003            1.1353        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_127.40        53.38     1_180.78       0.8095          1.0848            1.0492        15.56
IVFPQ-m32-nl316-np2                                      930.81        54.25       985.06       0.8249          1.0652            1.0373        13.02
IVFPQ-m64-nl316-np2                                    1_041.26        68.60     1_109.86       0.8350          1.0594            1.0302        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_127.40        62.73     1_190.12       0.8720          1.0397            1.0237        15.56
IVFPQ-m32-nl316-np4                                      930.81        67.02       997.83       0.8833          1.0261            1.0182        13.02
IVFPQ-m64-nl316-np4                                    1_041.26        89.91     1_131.16       0.8979          1.0200            1.0129        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_127.40        80.26     1_207.65       0.8862          1.0293            1.0184        15.56
IVFPQ-m32-nl316-np8                                      930.81        91.39     1_022.20       0.8918          1.0216            1.0159        13.02
IVFPQ-m64-nl316-np8                                    1_041.26       135.71     1_176.97       0.9071          1.0153            1.0110        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_127.40       116.21     1_243.60       0.8908          1.0239            1.0166        15.56
IVFPQ-m32-nl316-np15                                     930.81       135.38     1_066.19       0.8924          1.0213            1.0158        13.02
IVFPQ-m64-nl316-np15                                   1_041.26       215.68     1_256.94       0.9077          1.0150            1.0109        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_127.40       177.29     1_304.69       0.8921          1.0217            1.0159        15.56
IVFPQ-m32-nl316-np17                                     930.81       152.06     1_082.87       0.8924          1.0213            1.0158        13.02
IVFPQ-m64-nl316-np17                                   1_041.26       242.51     1_283.77       0.9077          1.0150            1.0109        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_127.40       188.18     1_315.57       0.8922          1.0215            1.0158        15.56
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
Exhaustive (query)                                        68.38     1_327.26     1_395.64       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_182.93        58.49     1_241.42       0.8136          1.0778            1.0433        10.29
SOARPQ-near-np2                                        1_182.93        76.73     1_259.66       0.8589          1.0401            1.0281        10.29
SOARPQ-near-np4                                        1_182.93       107.92     1_290.85       0.8676          1.0332            1.0251        10.29
SOARPQ-near-np7                                        1_182.93       146.58     1_329.51       0.8688          1.0317            1.0245        10.29
SOARPQ-near-np8                                        1_182.93       160.88     1_343.81       0.8690          1.0315            1.0244        10.29
SOARPQ-near-np12                                       1_182.93       207.10     1_390.03       0.8692          1.0312            1.0243        10.29
SOARPQ-shift0.3-np1                                    1_161.06        57.43     1_218.49       0.8171          1.0739            1.0436        10.29
SOARPQ-shift0.3-np2                                    1_161.06        74.21     1_235.26       0.8588          1.0406            1.0283        10.29
SOARPQ-shift0.3-np4                                    1_161.06       105.56     1_266.62       0.8672          1.0340            1.0255        10.29
SOARPQ-shift0.3-np7                                    1_161.06       147.97     1_309.03       0.8686          1.0321            1.0247        10.29
SOARPQ-shift0.3-np8                                    1_161.06       160.76     1_321.82       0.8689          1.0317            1.0244        10.29
SOARPQ-shift0.3-np12                                   1_161.06       206.90     1_367.95       0.8691          1.0312            1.0243        10.29
SOARPQ-shift0.7-np1                                    1_158.09        58.35     1_216.44       0.8120          1.0789            1.0467        10.29
SOARPQ-shift0.7-np2                                    1_158.09        76.62     1_234.71       0.8562          1.0433            1.0295        10.29
SOARPQ-shift0.7-np4                                    1_158.09       106.16     1_264.25       0.8666          1.0352            1.0259        10.29
SOARPQ-shift0.7-np7                                    1_158.09       147.81     1_305.90       0.8684          1.0326            1.0249        10.29
SOARPQ-shift0.7-np8                                    1_158.09       161.37     1_319.46       0.8688          1.0321            1.0246        10.29
SOARPQ-shift0.7-np12                                   1_158.09       207.29     1_365.38       0.8691          1.0313            1.0243        10.29
SOARPQ-orth1-np1                                       1_192.52        56.31     1_248.83       0.8140          1.0778            1.0441        10.29
SOARPQ-orth1-np2                                       1_192.52        73.99     1_266.52       0.8584          1.0413            1.0285        10.29
SOARPQ-orth1-np4                                       1_192.52       106.67     1_299.20       0.8672          1.0342            1.0254        10.29
SOARPQ-orth1-np7                                       1_192.52       145.92     1_338.45       0.8686          1.0322            1.0247        10.29
SOARPQ-orth1-np8                                       1_192.52       158.87     1_351.39       0.8689          1.0318            1.0244        10.29
SOARPQ-orth1-np12                                      1_192.52       206.29     1_398.81       0.8691          1.0312            1.0243        10.29
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        72.00     1_351.99     1_423.99       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                      864.70        51.32       916.02       0.7656          1.1464            1.0636         6.21
IVFPQ-m64-nl111-np1                                      940.56        63.21     1_003.77       0.7732          1.1390            1.0537        11.21
SOARPQ-orth1-m32-nl111-np1                             1_003.76        60.84     1_064.60       0.8458          1.0693            1.0366         8.71
IVFPQ-m32-nl111-np2                                      864.70        62.79       927.49       0.8632          1.0458            1.0276         6.21
IVFPQ-m64-nl111-np2                                      940.56        83.77     1_024.33       0.8766          1.0377            1.0203        11.21
SOARPQ-orth1-m32-nl111-np2                             1_003.76        78.64     1_082.40       0.8754          1.0407            1.0261         8.71
IVFPQ-m32-nl111-np4                                      864.70        88.37       953.07       0.8824          1.0311            1.0231         6.21
IVFPQ-m64-nl111-np4                                      940.56       126.44     1_067.00       0.8976          1.0229            1.0164        11.21
SOARPQ-orth1-m32-nl111-np4                             1_003.76       112.42     1_116.18       0.8822          1.0330            1.0235         8.71
IVFPQ-m32-nl111-np5                                      864.70       108.66       973.35       0.8833          1.0306            1.0228         6.21
IVFPQ-m64-nl111-np5                                      940.56       144.22     1_084.78       0.8985          1.0224            1.0161        11.21
SOARPQ-orth1-m32-nl111-np5                             1_003.76       134.65     1_138.41       0.8829          1.0320            1.0232         8.71
IVFPQ-m32-nl111-np8                                      864.70       139.71     1_004.41       0.8837          1.0304            1.0227         6.21
IVFPQ-m64-nl111-np8                                      940.56       208.59     1_149.15       0.8991          1.0221            1.0160        11.21
SOARPQ-orth1-m32-nl111-np8                             1_003.76       179.19     1_182.94       0.8836          1.0308            1.0228         8.71
IVFPQ-m32-nl111-np10                                     864.70       166.59     1_031.29       0.8838          1.0304            1.0227         6.21
IVFPQ-m64-nl111-np10                                     940.56       256.16     1_196.72       0.8992          1.0221            1.0160        11.21
SOARPQ-orth1-m32-nl111-np10                            1_003.76       209.55     1_213.30       0.8838          1.0306            1.0227         8.71
IVFPQ-m32-nl158-np1                                    1_006.03        49.13     1_055.15       0.7548          1.1545            1.0732         7.77
IVFPQ-m64-nl158-np1                                    1_092.85        58.77     1_151.62       0.7599          1.1495            1.0669        14.24
SOARPQ-orth1-m32-nl158-np1                             1_171.67        56.70     1_228.37       0.8459          1.0754            1.0343        10.29
IVFPQ-m32-nl158-np2                                    1_006.03        58.16     1_064.19       0.8655          1.0455            1.0254         7.77
IVFPQ-m64-nl158-np2                                    1_092.85        77.00     1_169.84       0.8747          1.0401            1.0200        14.24
SOARPQ-orth1-m32-nl158-np2                             1_171.67        71.65     1_243.32       0.8830          1.0403            1.0229        10.29
IVFPQ-m32-nl158-np4                                    1_006.03        80.34     1_086.37       0.8931          1.0254            1.0186         7.77
IVFPQ-m64-nl158-np4                                    1_092.85       111.88     1_204.73       0.9032          1.0200            1.0142        14.24
SOARPQ-orth1-m32-nl158-np4                             1_171.67       102.88     1_274.55       0.8929          1.0294            1.0192        10.29
IVFPQ-m32-nl158-np7                                    1_006.03       112.98     1_119.01       0.8955          1.0241            1.0180         7.77
IVFPQ-m64-nl158-np7                                    1_092.85       165.87     1_258.72       0.9057          1.0188            1.0135        14.24
SOARPQ-orth1-m32-nl158-np7                             1_171.67       141.69     1_313.36       0.8951          1.0255            1.0182        10.29
IVFPQ-m32-nl158-np8                                    1_006.03       123.84     1_129.87       0.8956          1.0241            1.0180         7.77
IVFPQ-m64-nl158-np8                                    1_092.85       188.37     1_281.21       0.9059          1.0187            1.0134        14.24
SOARPQ-orth1-m32-nl158-np8                             1_171.67       163.07     1_334.74       0.8953          1.0250            1.0181        10.29
IVFPQ-m32-nl158-np12                                   1_006.03       166.48     1_172.50       0.8957          1.0241            1.0179         7.77
IVFPQ-m64-nl158-np12                                   1_092.85       262.10     1_354.95       0.9059          1.0187            1.0134        14.24
SOARPQ-orth1-m32-nl158-np12                            1_171.67       204.32     1_375.99       0.8956          1.0243            1.0180        10.29
IVFPQ-m32-nl223-np1                                      914.73        49.83       964.55       0.7294          1.1793            1.1017         9.93
IVFPQ-m64-nl223-np1                                      994.78        55.48     1_050.27       0.7345          1.1745            1.0950        18.43
SOARPQ-orth1-m32-nl223-np1                             1_116.76        52.90     1_169.66       0.8375          1.0792            1.0390        12.44
IVFPQ-m32-nl223-np2                                      914.73        54.92       969.64       0.8598          1.0505            1.0273         9.93
IVFPQ-m64-nl223-np2                                      994.78        70.71     1_065.49       0.8698          1.0453            1.0215        18.43
SOARPQ-orth1-m32-nl223-np2                             1_116.76        64.81     1_181.57       0.8844          1.0391            1.0216        12.44
IVFPQ-m32-nl223-np4                                      914.73        67.36       982.09       0.8955          1.0240            1.0173         9.93
IVFPQ-m64-nl223-np4                                      994.78        93.90     1_088.68       0.9083          1.0185            1.0126        18.43
SOARPQ-orth1-m32-nl223-np4                             1_116.76        83.55     1_200.32       0.8948          1.0289            1.0180        12.44
IVFPQ-m32-nl223-np8                                      914.73        99.54     1_014.27       0.8984          1.0224            1.0166         9.93
IVFPQ-m64-nl223-np8                                      994.78       140.71     1_135.50       0.9116          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np8                             1_116.76       118.51     1_235.27       0.8980          1.0237            1.0169        12.44
IVFPQ-m32-nl223-np11                                     914.73       116.46     1_031.19       0.8986          1.0223            1.0166         9.93
IVFPQ-m64-nl223-np11                                     994.78       177.59     1_172.37       0.9118          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np11                            1_116.76       149.12     1_265.88       0.8984          1.0228            1.0167        12.44
IVFPQ-m32-nl223-np14                                     914.73       140.17     1_054.89       0.8986          1.0223            1.0166         9.93
IVFPQ-m64-nl223-np14                                     994.78       218.01     1_212.80       0.9118          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np14                            1_116.76       178.15     1_294.91       0.8986          1.0225            1.0166        12.44
IVFPQ-m32-nl316-np1                                      949.43        47.80       997.22       0.7028          1.2100            1.1294        13.02
IVFPQ-m64-nl316-np1                                    1_052.63        56.43     1_109.06       0.7064          1.2057            1.1253        24.42
SOARPQ-orth1-m32-nl316-np1                             1_201.02        51.53     1_252.55       0.8276          1.0861            1.0428        15.56
IVFPQ-m32-nl316-np2                                      949.43        54.87     1_004.30       0.8497          1.0588            1.0299        13.02
IVFPQ-m64-nl316-np2                                    1_052.63        66.11     1_118.74       0.8576          1.0543            1.0245        24.42
SOARPQ-orth1-m32-nl316-np2                             1_201.02        60.83     1_261.86       0.8875          1.0362            1.0203        15.56
IVFPQ-m32-nl316-np4                                      949.43        68.77     1_018.20       0.8998          1.0219            1.0153        13.02
IVFPQ-m64-nl316-np4                                    1_052.63        86.39     1_139.02       0.9104          1.0175            1.0115        24.42
SOARPQ-orth1-m32-nl316-np4                             1_201.02        76.53     1_277.55       0.9009          1.0258            1.0159        15.56
IVFPQ-m32-nl316-np8                                      949.43        89.62     1_039.05       0.9045          1.0193            1.0143        13.02
IVFPQ-m64-nl316-np8                                    1_052.63       127.41     1_180.04       0.9154          1.0150            1.0105        24.42
SOARPQ-orth1-m32-nl316-np8                             1_201.02       104.07     1_305.10       0.9040          1.0211            1.0146        15.56
IVFPQ-m32-nl316-np15                                     949.43       129.00     1_078.43       0.9047          1.0192            1.0143        13.02
IVFPQ-m64-nl316-np15                                   1_052.63       211.72     1_264.35       0.9156          1.0149            1.0105        24.42
SOARPQ-orth1-m32-nl316-np15                            1_201.02       158.36     1_359.38       0.9046          1.0195            1.0143        15.56
IVFPQ-m32-nl316-np17                                     949.43       158.00     1_107.43       0.9047          1.0192            1.0143        13.02
IVFPQ-m64-nl316-np17                                   1_052.63       223.05     1_275.68       0.9156          1.0149            1.0105        24.42
SOARPQ-orth1-m32-nl316-np17                            1_201.02       174.03     1_375.05       0.9047          1.0193            1.0143        15.56
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
Exhaustive (query)                                        72.00     1_351.99     1_423.99       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        1_157.59        57.04     1_214.63       0.8522          1.0645            1.0324        10.27
SOARPQ-near-np2                                        1_157.59        71.41     1_228.99       0.8856          1.0347            1.0219        10.27
SOARPQ-near-np4                                        1_157.59       101.26     1_258.85       0.8939          1.0269            1.0187        10.27
SOARPQ-near-np7                                        1_157.59       142.90     1_300.49       0.8955          1.0245            1.0181        10.27
SOARPQ-near-np8                                        1_157.59       156.28     1_313.87       0.8955          1.0243            1.0180        10.27
SOARPQ-near-np12                                       1_157.59       201.14     1_358.73       0.8957          1.0241            1.0179        10.27
SOARPQ-shift0.3-np1                                    1_145.03        58.26     1_203.29       0.8504          1.0685            1.0339        10.29
SOARPQ-shift0.3-np2                                    1_145.03        74.33     1_219.36       0.8839          1.0381            1.0227        10.29
SOARPQ-shift0.3-np4                                    1_145.03       101.45     1_246.48       0.8928          1.0287            1.0192        10.29
SOARPQ-shift0.3-np7                                    1_145.03       146.64     1_291.67       0.8951          1.0251            1.0182        10.29
SOARPQ-shift0.3-np8                                    1_145.03       156.24     1_301.28       0.8953          1.0247            1.0181        10.29
SOARPQ-shift0.3-np12                                   1_145.03       207.08     1_352.11       0.8956          1.0242            1.0180        10.29
SOARPQ-shift0.7-np1                                    1_122.18        57.56     1_179.74       0.8449          1.0770            1.0360        10.29
SOARPQ-shift0.7-np2                                    1_122.18        73.94     1_196.12       0.8806          1.0438            1.0238        10.29
SOARPQ-shift0.7-np4                                    1_122.18       102.39     1_224.57       0.8912          1.0318            1.0198        10.29
SOARPQ-shift0.7-np7                                    1_122.18       144.38     1_266.56       0.8946          1.0263            1.0184        10.29
SOARPQ-shift0.7-np8                                    1_122.18       152.92     1_275.10       0.8949          1.0256            1.0183        10.29
SOARPQ-shift0.7-np12                                   1_122.18       201.93     1_324.11       0.8956          1.0244            1.0180        10.29
SOARPQ-orth1-np1                                       1_166.33        58.89     1_225.22       0.8459          1.0754            1.0343        10.29
SOARPQ-orth1-np2                                       1_166.33        74.20     1_240.53       0.8830          1.0403            1.0229        10.29
SOARPQ-orth1-np4                                       1_166.33       102.75     1_269.08       0.8929          1.0294            1.0192        10.29
SOARPQ-orth1-np7                                       1_166.33       138.84     1_305.17       0.8951          1.0255            1.0182        10.29
SOARPQ-orth1-np8                                       1_166.33       157.24     1_323.57       0.8953          1.0250            1.0181        10.29
SOARPQ-orth1-np12                                      1_166.33       203.75     1_370.08       0.8956          1.0243            1.0180        10.29
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.50     1_323.41     1_393.91       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_993.01        80.67     3_073.68       0.3599          1.0700            1.0689         7.46
IVFOPQ-m64-nl111-np1                                   3_337.16       106.98     3_444.15       0.4579          1.0469            1.0429        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_160.57        87.60     3_248.17       0.3494          1.1073            1.0710         9.95
IVFOPQ-m32-nl111-np2                                   2_993.01        90.26     3_083.27       0.3628          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np2                                   3_337.16       124.18     3_461.35       0.4645          1.0449            1.0419        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_160.57        95.94     3_256.50       0.3618          1.0698            1.0685         9.95
IVFOPQ-m32-nl111-np4                                   2_993.01       102.55     3_095.56       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np4                                   3_337.16       140.21     3_477.37       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_160.57       107.97     3_268.54       0.3630          1.0687            1.0682         9.95
IVFOPQ-m32-nl111-np5                                   2_993.01       106.35     3_099.36       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np5                                   3_337.16       155.73     3_492.89       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_160.57       111.88     3_272.44       0.3630          1.0687            1.0682         9.95
IVFOPQ-m32-nl111-np8                                   2_993.01       119.40     3_112.41       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np8                                   3_337.16       179.25     3_516.42       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_160.57       128.97     3_289.54       0.3630          1.0687            1.0682         9.95
IVFOPQ-m32-nl111-np10                                  2_993.01       133.04     3_126.05       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np10                                  3_337.16       200.12     3_537.28       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_160.57       143.39     3_303.96       0.3630          1.0687            1.0682         9.95
IVFOPQ-m32-nl158-np1                                   3_434.92        78.71     3_513.63       0.3622          1.0670            1.0676         9.27
IVFOPQ-m64-nl158-np1                                   3_825.65        97.70     3_923.34       0.4613          1.0452            1.0420        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_606.57        88.14     3_694.71       0.3388          1.1233            1.0713        11.76
IVFOPQ-m32-nl158-np2                                   3_434.92        89.99     3_524.91       0.3690          1.0645            1.0659         9.27
IVFOPQ-m64-nl158-np2                                   3_825.65       119.56     3_945.20       0.4755          1.0416            1.0398        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_606.57        94.42     3_700.99       0.3675          1.0672            1.0664        11.76
IVFOPQ-m32-nl158-np4                                   3_434.92        99.68     3_534.60       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np4                                   3_825.65       139.46     3_965.10       0.4767          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_606.57       100.99     3_707.55       0.3694          1.0645            1.0659        11.76
IVFOPQ-m32-nl158-np7                                   3_434.92       106.01     3_540.94       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np7                                   3_825.65       160.30     3_985.95       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_606.57       116.45     3_723.02       0.3695          1.0643            1.0658        11.76
IVFOPQ-m32-nl158-np8                                   3_434.92       118.65     3_553.58       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np8                                   3_825.65       171.60     3_997.25       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_606.57       121.31     3_727.88       0.3695          1.0643            1.0658        11.76
IVFOPQ-m32-nl158-np12                                  3_434.92       131.98     3_566.90       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np12                                  3_825.65       195.80     4_021.45       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_606.57       139.81     3_746.37       0.3695          1.0643            1.0658        11.76
IVFOPQ-m32-nl223-np1                                   3_536.45        64.73     3_601.18       0.3636          1.0657            1.0659        11.43
IVFOPQ-m64-nl223-np1                                   3_898.78        75.99     3_974.77       0.4527          1.0457            1.0425        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_617.02        72.64     3_689.67       0.3388          1.1083            1.0710        13.92
IVFOPQ-m32-nl223-np2                                   3_536.45        74.91     3_611.36       0.3792          1.0607            1.0625        11.43
IVFOPQ-m64-nl223-np2                                   3_898.78        96.74     3_995.52       0.4857          1.0388            1.0376        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_617.02        81.15     3_698.17       0.3752          1.0635            1.0642        13.92
IVFOPQ-m32-nl223-np4                                   3_536.45        91.20     3_627.64       0.3813          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np4                                   3_898.78       122.75     4_021.53       0.4923          1.0378            1.0366        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_617.02        94.33     3_711.35       0.3804          1.0609            1.0622        13.92
IVFOPQ-m32-nl223-np8                                   3_536.45       104.17     3_640.62       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np8                                   3_898.78       158.45     4_057.23       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_617.02       109.27     3_726.30       0.3815          1.0601            1.0619        13.92
IVFOPQ-m32-nl223-np11                                  3_536.45       116.40     3_652.84       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np11                                  3_898.78       183.22     4_082.01       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_617.02       121.13     3_738.16       0.3815          1.0601            1.0619        13.92
IVFOPQ-m32-nl223-np14                                  3_536.45       129.02     3_665.47       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np14                                  3_898.78       201.33     4_100.11       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_617.02       134.46     3_751.49       0.3815          1.0601            1.0619        13.92
IVFOPQ-m32-nl316-np1                                   3_618.27        63.55     3_681.82       0.3641          1.0647            1.0652        15.02
IVFOPQ-m64-nl316-np1                                   3_861.31        72.44     3_933.75       0.4454          1.0455            1.0429        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_699.21        68.32     3_767.53       0.3369          1.1127            1.0691        17.52
IVFOPQ-m32-nl316-np2                                   3_618.27        69.89     3_688.16       0.3854          1.0581            1.0608        15.02
IVFOPQ-m64-nl316-np2                                   3_861.31        85.12     3_946.43       0.4876          1.0369            1.0368        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_699.21        77.14     3_776.35       0.3812          1.0609            1.0626        17.52
IVFOPQ-m32-nl316-np4                                   3_618.27        82.39     3_700.67       0.3905          1.0570            1.0595        15.02
IVFOPQ-m64-nl316-np4                                   3_861.31       112.24     3_973.55       0.5014          1.0349            1.0349        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_699.21        90.24     3_789.45       0.3874          1.0588            1.0605        17.52
IVFOPQ-m32-nl316-np8                                   3_618.27       106.13     3_724.41       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np8                                   3_861.31       151.02     4_012.34       0.5032          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_699.21       105.52     3_804.73       0.3908          1.0571            1.0595        17.52
IVFOPQ-m32-nl316-np15                                  3_618.27       128.44     3_746.72       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np15                                  3_861.31       201.59     4_062.91       0.5032          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_699.21       129.20     3_828.41       0.3910          1.0568            1.0594        17.52
IVFOPQ-m32-nl316-np17                                  3_618.27       131.75     3_750.03       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np17                                  3_861.31       216.46     4_077.77       0.5032          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_699.21       134.42     3_833.63       0.3910          1.0568            1.0594        17.52
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
Exhaustive (query)                                        70.50     1_323.41     1_393.91       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_504.19        85.15     3_589.34       0.3401          1.1216            1.0712        11.76
SOAROPQ-near-np2                                       3_504.19        87.67     3_591.86       0.3679          1.0668            1.0663        11.76
SOAROPQ-near-np4                                       3_504.19       101.22     3_605.40       0.3694          1.0645            1.0659        11.76
SOAROPQ-near-np7                                       3_504.19       112.99     3_617.18       0.3695          1.0643            1.0658        11.76
SOAROPQ-near-np8                                       3_504.19       118.08     3_622.26       0.3695          1.0643            1.0658        11.76
SOAROPQ-near-np12                                      3_504.19       135.03     3_639.22       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.3-np1                                   3_498.91        85.10     3_584.01       0.3393          1.1227            1.0712        11.76
SOAROPQ-shift0.3-np2                                   3_498.91        87.14     3_586.05       0.3676          1.0671            1.0664        11.76
SOAROPQ-shift0.3-np4                                   3_498.91        97.35     3_596.26       0.3694          1.0645            1.0659        11.76
SOAROPQ-shift0.3-np7                                   3_498.91       110.47     3_609.38       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.3-np8                                   3_498.91       116.69     3_615.60       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.3-np12                                  3_498.91       134.01     3_632.91       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.7-np1                                   3_510.38        87.00     3_597.38       0.3386          1.1236            1.0714        11.76
SOAROPQ-shift0.7-np2                                   3_510.38        89.13     3_599.51       0.3674          1.0672            1.0664        11.76
SOAROPQ-shift0.7-np4                                   3_510.38        97.89     3_608.27       0.3694          1.0645            1.0659        11.76
SOAROPQ-shift0.7-np7                                   3_510.38       114.10     3_624.48       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.7-np8                                   3_510.38       118.60     3_628.98       0.3695          1.0643            1.0658        11.76
SOAROPQ-shift0.7-np12                                  3_510.38       132.51     3_642.89       0.3695          1.0643            1.0658        11.76
SOAROPQ-orth1-np1                                      3_562.86        84.90     3_647.76       0.3397          1.1228            1.0712        11.76
SOAROPQ-orth1-np2                                      3_562.86        88.69     3_651.55       0.3677          1.0670            1.0663        11.76
SOAROPQ-orth1-np4                                      3_562.86        98.69     3_661.55       0.3694          1.0645            1.0659        11.76
SOAROPQ-orth1-np7                                      3_562.86       111.43     3_674.29       0.3695          1.0643            1.0658        11.76
SOAROPQ-orth1-np8                                      3_562.86       117.36     3_680.22       0.3695          1.0643            1.0658        11.76
SOAROPQ-orth1-np12                                     3_562.86       139.31     3_702.18       0.3695          1.0643            1.0658        11.76
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.20     1_341.92     1_410.12       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_975.73        81.79     3_057.52       0.6633          1.0306            1.0248         7.46
IVFOPQ-m64-nl111-np1                                   3_357.62       104.56     3_462.18       0.7519          1.0179            1.0115        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_172.47        86.28     3_258.75       0.6701          1.0307            1.0246         9.95
IVFOPQ-m32-nl111-np2                                   2_975.73        88.19     3_063.92       0.6816          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np2                                   3_357.62       121.04     3_478.66       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_172.47        94.06     3_266.52       0.6813          1.0253            1.0238         9.95
IVFOPQ-m32-nl111-np4                                   2_975.73        98.18     3_073.91       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np4                                   3_357.62       136.15     3_493.77       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_172.47       101.71     3_274.18       0.6817          1.0251            1.0237         9.95
IVFOPQ-m32-nl111-np5                                   2_975.73       103.46     3_079.19       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np5                                   3_357.62       141.33     3_498.95       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_172.47       111.98     3_284.45       0.6817          1.0251            1.0237         9.95
IVFOPQ-m32-nl111-np8                                   2_975.73       118.21     3_093.94       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np8                                   3_357.62       169.14     3_526.76       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_172.47       135.85     3_308.32       0.6817          1.0251            1.0237         9.95
IVFOPQ-m32-nl111-np10                                  2_975.73       130.48     3_106.21       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np10                                  3_357.62       187.78     3_545.40       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_172.47       155.98     3_328.44       0.6817          1.0251            1.0237         9.95
IVFOPQ-m32-nl158-np1                                   3_333.14        76.75     3_409.89       0.6009          1.0434            1.0325         9.27
IVFOPQ-m64-nl158-np1                                   3_737.04        93.67     3_830.71       0.6720          1.0313            1.0180        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_550.20        82.90     3_633.10       0.6626          1.0298            1.0258        11.76
IVFOPQ-m32-nl158-np2                                   3_333.14        84.19     3_417.33       0.6732          1.0265            1.0245         9.27
IVFOPQ-m64-nl158-np2                                   3_737.04       109.39     3_846.42       0.7628          1.0137            1.0113        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_550.20        90.41     3_640.61       0.6827          1.0249            1.0236        11.76
IVFOPQ-m32-nl158-np4                                   3_333.14        95.49     3_428.63       0.6846          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np4                                   3_737.04       129.10     3_866.14       0.7779          1.0116            1.0105        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_550.20        98.16     3_648.36       0.6851          1.0244            1.0233        11.76
IVFOPQ-m32-nl158-np7                                   3_333.14       108.24     3_441.38       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np7                                   3_737.04       151.08     3_888.12       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_550.20       111.68     3_661.88       0.6850          1.0244            1.0233        11.76
IVFOPQ-m32-nl158-np8                                   3_333.14       111.29     3_444.43       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np8                                   3_737.04       158.58     3_895.62       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_550.20       115.87     3_666.07       0.6850          1.0244            1.0233        11.76
IVFOPQ-m32-nl158-np12                                  3_333.14       131.63     3_464.77       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np12                                  3_737.04       190.37     3_927.40       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_550.20       137.23     3_687.43       0.6850          1.0244            1.0233        11.76
IVFOPQ-m32-nl223-np1                                   3_440.34        73.92     3_514.26       0.5148          1.0613            1.0521        11.43
IVFOPQ-m64-nl223-np1                                   3_786.54        82.58     3_869.12       0.5596          1.0502            1.0411        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_632.94        80.61     3_713.55       0.6279          1.0356            1.0312        13.92
IVFOPQ-m32-nl223-np2                                   3_440.34        82.35     3_522.69       0.6390          1.0326            1.0291        11.43
IVFOPQ-m64-nl223-np2                                   3_786.54       101.90     3_888.44       0.7154          1.0207            1.0158        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_632.94        91.73     3_724.67       0.6773          1.0259            1.0245        13.92
IVFOPQ-m32-nl223-np4                                   3_440.34        95.11     3_535.45       0.6861          1.0242            1.0230        11.43
IVFOPQ-m64-nl223-np4                                   3_786.54       123.50     3_910.04       0.7776          1.0119            1.0108        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_632.94        98.01     3_730.95       0.6903          1.0235            1.0224        13.92
IVFOPQ-m32-nl223-np8                                   3_440.34       110.03     3_550.37       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np8                                   3_786.54       153.54     3_940.08       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_632.94       113.01     3_745.95       0.6906          1.0234            1.0224        13.92
IVFOPQ-m32-nl223-np11                                  3_440.34       121.27     3_561.61       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np11                                  3_786.54       179.09     3_965.63       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_632.94       126.94     3_759.88       0.6906          1.0234            1.0224        13.92
IVFOPQ-m32-nl223-np14                                  3_440.34       135.00     3_575.33       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np14                                  3_786.54       199.41     3_985.95       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_632.94       137.04     3_769.97       0.6906          1.0234            1.0224        13.92
IVFOPQ-m32-nl316-np1                                   3_452.47        63.54     3_516.01       0.4112          1.0845            1.0784        15.02
IVFOPQ-m64-nl316-np1                                   3_876.11        70.79     3_946.90       0.4292          1.0746            1.0679        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_750.95        69.02     3_819.97       0.5461          1.0506            1.0474        17.52
IVFOPQ-m32-nl316-np2                                   3_452.47        70.53     3_523.01       0.5566          1.0475            1.0444        15.02
IVFOPQ-m64-nl316-np2                                   3_876.11        84.54     3_960.65       0.6052          1.0365            1.0329        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_750.95        79.43     3_830.38       0.6412          1.0320            1.0298        17.52
IVFOPQ-m32-nl316-np4                                   3_452.47        86.14     3_538.61       0.6579          1.0290            1.0271        15.02
IVFOPQ-m64-nl316-np4                                   3_876.11       114.00     3_990.11       0.7360          1.0174            1.0147        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_750.95        91.84     3_842.79       0.6857          1.0243            1.0229        17.52
IVFOPQ-m32-nl316-np8                                   3_452.47       105.45     3_557.92       0.6929          1.0231            1.0217        15.02
IVFOPQ-m64-nl316-np8                                   3_876.11       148.80     4_024.91       0.7821          1.0112            1.0102        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_750.95       108.43     3_859.38       0.6945          1.0228            1.0215        17.52
IVFOPQ-m32-nl316-np15                                  3_452.47       129.76     3_582.23       0.6949          1.0228            1.0215        15.02
IVFOPQ-m64-nl316-np15                                  3_876.11       198.29     4_074.40       0.7844          1.0109            1.0100        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_750.95       133.10     3_884.05       0.6949          1.0228            1.0215        17.52
IVFOPQ-m32-nl316-np17                                  3_452.47       139.60     3_592.08       0.6949          1.0228            1.0215        15.02
IVFOPQ-m64-nl316-np17                                  3_876.11       220.24     4_096.35       0.7844          1.0109            1.0100        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_750.95       150.46     3_901.41       0.6949          1.0228            1.0215        17.52
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
Exhaustive (query)                                        68.20     1_341.92     1_410.12       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_474.00        82.40     3_556.39       0.6614          1.0303            1.0259        11.76
SOAROPQ-near-np2                                       3_474.00        88.25     3_562.25       0.6828          1.0249            1.0236        11.76
SOAROPQ-near-np4                                       3_474.00        96.89     3_570.89       0.6851          1.0244            1.0233        11.76
SOAROPQ-near-np7                                       3_474.00       110.63     3_584.63       0.6850          1.0244            1.0233        11.76
SOAROPQ-near-np8                                       3_474.00       116.63     3_590.62       0.6850          1.0244            1.0233        11.76
SOAROPQ-near-np12                                      3_474.00       137.79     3_611.79       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.3-np1                                   3_517.37        85.78     3_603.15       0.6624          1.0299            1.0258        11.76
SOAROPQ-shift0.3-np2                                   3_517.37        89.88     3_607.25       0.6827          1.0249            1.0236        11.76
SOAROPQ-shift0.3-np4                                   3_517.37        96.57     3_613.94       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.3-np7                                   3_517.37       110.43     3_627.80       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.3-np8                                   3_517.37       117.53     3_634.90       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.3-np12                                  3_517.37       137.95     3_655.32       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.7-np1                                   3_966.02        88.04     4_054.06       0.6625          1.0298            1.0259        11.76
SOAROPQ-shift0.7-np2                                   3_966.02        94.33     4_060.35       0.6827          1.0249            1.0236        11.76
SOAROPQ-shift0.7-np4                                   3_966.02       103.66     4_069.69       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.7-np7                                   3_966.02       114.24     4_080.26       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.7-np8                                   3_966.02       119.13     4_085.15       0.6850          1.0244            1.0233        11.76
SOAROPQ-shift0.7-np12                                  3_966.02       145.93     4_111.95       0.6850          1.0244            1.0233        11.76
SOAROPQ-orth1-np1                                      3_957.65        84.34     4_041.99       0.6621          1.0299            1.0260        11.76
SOAROPQ-orth1-np2                                      3_957.65        87.74     4_045.38       0.6826          1.0249            1.0236        11.76
SOAROPQ-orth1-np4                                      3_957.65        96.94     4_054.59       0.6850          1.0244            1.0233        11.76
SOAROPQ-orth1-np7                                      3_957.65       109.23     4_066.88       0.6850          1.0244            1.0233        11.76
SOAROPQ-orth1-np8                                      3_957.65       114.14     4_071.79       0.6850          1.0244            1.0233        11.76
SOAROPQ-orth1-np12                                     3_957.65       143.18     4_100.82       0.6850          1.0244            1.0233        11.76
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.25     1_365.56     1_433.81       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   3_027.92        65.22     3_093.14       0.7108          1.1808            1.0958         7.46
IVFOPQ-m64-nl111-np1                                   3_397.46        77.78     3_475.24       0.7147          1.1780            1.0933        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_184.80        76.00     3_260.80       0.8389          1.0602            1.0300         9.96
IVFOPQ-m32-nl111-np2                                   3_027.92        77.45     3_105.37       0.8467          1.0518            1.0256         7.46
IVFOPQ-m64-nl111-np2                                   3_397.46       107.25     3_504.71       0.8548          1.0486            1.0219        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_184.80        99.36     3_284.16       0.8841          1.0270            1.0175         9.96
IVFOPQ-m32-nl111-np4                                   3_027.92       105.29     3_133.21       0.8894          1.0225            1.0159         7.46
IVFOPQ-m64-nl111-np4                                   3_397.46       152.62     3_550.08       0.9004          1.0190            1.0125        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_184.80       133.65     3_318.45       0.8922          1.0214            1.0154         9.96
IVFOPQ-m32-nl111-np5                                   3_027.92       116.43     3_144.35       0.8921          1.0210            1.0154         7.46
IVFOPQ-m64-nl111-np5                                   3_397.46       174.82     3_572.28       0.9033          1.0175            1.0120        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_184.80       157.10     3_341.90       0.8928          1.0208            1.0153         9.96
IVFOPQ-m32-nl111-np8                                   3_027.92       157.33     3_185.25       0.8934          1.0202            1.0151         7.46
IVFOPQ-m64-nl111-np8                                   3_397.46       249.32     3_646.78       0.9049          1.0167            1.0117        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_184.80       197.89     3_382.69       0.8934          1.0203            1.0151         9.96
IVFOPQ-m32-nl111-np10                                  3_027.92       182.06     3_209.99       0.8935          1.0202            1.0150         7.46
IVFOPQ-m64-nl111-np10                                  3_397.46       306.69     3_704.14       0.9050          1.0167            1.0116        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_184.80       219.85     3_404.65       0.8935          1.0202            1.0150         9.96
IVFOPQ-m32-nl158-np1                                   3_336.37        62.99     3_399.36       0.7063          1.1823            1.1002         9.27
IVFOPQ-m64-nl158-np1                                   3_735.38        72.78     3_808.16       0.7095          1.1793            1.0982        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_556.08        68.92     3_625.00       0.8383          1.0603            1.0294        11.79
IVFOPQ-m32-nl158-np2                                   3_336.37        71.06     3_407.43       0.8499          1.0500            1.0244         9.27
IVFOPQ-m64-nl158-np2                                   3_735.38        91.63     3_827.01       0.8570          1.0470            1.0204        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_556.08        85.31     3_641.38       0.8896          1.0251            1.0157        11.79
IVFOPQ-m32-nl158-np4                                   3_336.37        89.49     3_425.86       0.8960          1.0201            1.0139         9.27
IVFOPQ-m64-nl158-np4                                   3_735.38       131.06     3_866.44       0.9055          1.0167            1.0109        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_556.08       119.86     3_675.93       0.8998          1.0188            1.0132        11.79
IVFOPQ-m32-nl158-np7                                   3_336.37       121.73     3_458.10       0.9010          1.0174            1.0128         9.27
IVFOPQ-m64-nl158-np7                                   3_735.38       183.22     3_918.60       0.9110          1.0140            1.0098        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_556.08       151.32     3_707.39       0.9011          1.0177            1.0128        11.79
IVFOPQ-m32-nl158-np8                                   3_336.37       129.06     3_465.43       0.9013          1.0173            1.0127         9.27
IVFOPQ-m64-nl158-np8                                   3_735.38       200.65     3_936.03       0.9112          1.0138            1.0098        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_556.08       168.27     3_724.34       0.9013          1.0175            1.0128        11.79
IVFOPQ-m32-nl158-np12                                  3_336.37       166.62     3_502.99       0.9015          1.0172            1.0126         9.27
IVFOPQ-m64-nl158-np12                                  3_735.38       274.51     4_009.89       0.9115          1.0137            1.0097        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_556.08       202.24     3_758.31       0.9015          1.0172            1.0127        11.79
IVFOPQ-m32-nl223-np1                                   3_426.41        61.55     3_487.96       0.6999          1.1824            1.1101        11.43
IVFOPQ-m64-nl223-np1                                   3_855.07        69.80     3_924.87       0.7028          1.1792            1.1075        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_640.95        65.40     3_706.35       0.8355          1.0643            1.0322        13.94
IVFOPQ-m32-nl223-np2                                   3_426.41        67.76     3_494.18       0.8479          1.0517            1.0252        11.43
IVFOPQ-m64-nl223-np2                                   3_855.07        84.12     3_939.19       0.8547          1.0483            1.0209        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_640.95        77.37     3_718.32       0.8928          1.0254            1.0149        13.94
IVFOPQ-m32-nl223-np4                                   3_426.41        83.12     3_509.54       0.9003          1.0190            1.0125        11.43
IVFOPQ-m64-nl223-np4                                   3_855.07       110.06     3_965.13       0.9096          1.0155            1.0096        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_640.95       100.35     3_741.30       0.9049          1.0181            1.0119        13.94
IVFOPQ-m32-nl223-np8                                   3_426.41       109.81     3_536.23       0.9076          1.0155            1.0110        11.43
IVFOPQ-m64-nl223-np8                                   3_855.07       168.07     4_023.14       0.9174          1.0119            1.0081        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_640.95       138.66     3_779.61       0.9075          1.0160            1.0111        13.94
IVFOPQ-m32-nl223-np11                                  3_426.41       132.32     3_558.73       0.9080          1.0153            1.0109        11.43
IVFOPQ-m64-nl223-np11                                  3_855.07       212.75     4_067.82       0.9179          1.0117            1.0080        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_640.95       165.42     3_806.37       0.9079          1.0156            1.0110        13.94
IVFOPQ-m32-nl223-np14                                  3_426.41       155.55     3_581.96       0.9081          1.0152            1.0109        11.43
IVFOPQ-m64-nl223-np14                                  3_855.07       251.82     4_106.89       0.9180          1.0116            1.0080        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_640.95       189.41     3_830.36       0.9081          1.0154            1.0109        13.94
IVFOPQ-m32-nl316-np1                                   3_498.19        62.25     3_560.44       0.6802          1.1986            1.1335        15.02
IVFOPQ-m64-nl316-np1                                   3_870.37        69.03     3_939.41       0.6816          1.1968            1.1316        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_691.45        64.09     3_755.55       0.8253          1.0714            1.0362        17.57
IVFOPQ-m32-nl316-np2                                   3_498.19        66.15     3_564.33       0.8395          1.0575            1.0274        15.02
IVFOPQ-m64-nl316-np2                                   3_870.37        79.84     3_950.22       0.8437          1.0555            1.0255        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_691.45        74.69     3_766.14       0.8947          1.0260            1.0140        17.57
IVFOPQ-m32-nl316-np4                                   3_498.19        76.72     3_574.91       0.9044          1.0178            1.0110        15.02
IVFOPQ-m64-nl316-np4                                   3_870.37       102.32     3_972.70       0.9109          1.0157            1.0090        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_691.45        89.19     3_780.64       0.9104          1.0168            1.0101        17.57
IVFOPQ-m32-nl316-np8                                   3_498.19        98.77     3_596.95       0.9143          1.0131            1.0090        15.02
IVFOPQ-m64-nl316-np8                                   3_870.37       147.68     4_018.06       0.9211          1.0108            1.0073        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_691.45       124.56     3_816.01       0.9140          1.0139            1.0092        17.57
IVFOPQ-m32-nl316-np15                                  3_498.19       137.96     3_636.15       0.9150          1.0128            1.0089        15.02
IVFOPQ-m64-nl316-np15                                  3_870.37       226.17     4_096.54       0.9219          1.0105            1.0071        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_691.45       172.93     3_864.38       0.9149          1.0129            1.0089        17.57
IVFOPQ-m32-nl316-np17                                  3_498.19       148.93     3_647.12       0.9150          1.0128            1.0089        15.02
IVFOPQ-m64-nl316-np17                                  3_870.37       250.32     4_120.69       0.9219          1.0105            1.0071        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_691.45       185.57     3_877.02       0.9150          1.0129            1.0089        17.57
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
Exhaustive (query)                                        68.25     1_365.56     1_433.81       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_529.02        70.39     3_599.41       0.8364          1.0630            1.0279        11.79
SOAROPQ-near-np2                                       3_529.02        84.58     3_613.59       0.8905          1.0243            1.0153        11.79
SOAROPQ-near-np4                                       3_529.02       116.24     3_645.26       0.9004          1.0182            1.0131        11.79
SOAROPQ-near-np7                                       3_529.02       152.50     3_681.51       0.9013          1.0174            1.0128        11.79
SOAROPQ-near-np8                                       3_529.02       162.19     3_691.21       0.9014          1.0173            1.0127        11.79
SOAROPQ-near-np12                                      3_529.02       209.54     3_738.56       0.9015          1.0172            1.0127        11.79
SOAROPQ-shift0.3-np1                                   3_505.23        70.23     3_575.47       0.8406          1.0586            1.0282        11.79
SOAROPQ-shift0.3-np2                                   3_505.23        84.71     3_589.94       0.8906          1.0243            1.0154        11.79
SOAROPQ-shift0.3-np4                                   3_505.23       114.53     3_619.77       0.9000          1.0185            1.0132        11.79
SOAROPQ-shift0.3-np7                                   3_505.23       154.59     3_659.83       0.9012          1.0175            1.0128        11.79
SOAROPQ-shift0.3-np8                                   3_505.23       161.94     3_667.17       0.9013          1.0174            1.0127        11.79
SOAROPQ-shift0.3-np12                                  3_505.23       207.34     3_712.57       0.9015          1.0172            1.0127        11.79
SOAROPQ-shift0.7-np1                                   3_490.57        69.96     3_560.54       0.8354          1.0629            1.0307        11.79
SOAROPQ-shift0.7-np2                                   3_490.57        86.31     3_576.88       0.8886          1.0260            1.0160        11.79
SOAROPQ-shift0.7-np4                                   3_490.57       113.08     3_603.65       0.8995          1.0191            1.0133        11.79
SOAROPQ-shift0.7-np7                                   3_490.57       154.23     3_644.80       0.9010          1.0178            1.0129        11.79
SOAROPQ-shift0.7-np8                                   3_490.57       169.15     3_659.73       0.9012          1.0176            1.0128        11.79
SOAROPQ-shift0.7-np12                                  3_490.57       211.13     3_701.71       0.9015          1.0172            1.0127        11.79
SOAROPQ-orth1-np1                                      3_556.10        72.72     3_628.81       0.8374          1.0621            1.0285        11.79
SOAROPQ-orth1-np2                                      3_556.10        85.16     3_641.26       0.8900          1.0248            1.0154        11.79
SOAROPQ-orth1-np4                                      3_556.10       114.90     3_671.00       0.9000          1.0186            1.0131        11.79
SOAROPQ-orth1-np7                                      3_556.10       157.51     3_713.61       0.9012          1.0176            1.0128        11.79
SOAROPQ-orth1-np8                                      3_556.10       167.38     3_723.48       0.9013          1.0175            1.0128        11.79
SOAROPQ-orth1-np12                                     3_556.10       206.83     3_762.92       0.9015          1.0172            1.0127        11.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-49-gbb5be1f), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        72.24     1_358.50     1_430.74       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   3_040.20        63.05     3_103.25       0.7747          1.1390            1.0539         7.46
IVFOPQ-m64-nl111-np1                                   3_403.29        76.78     3_480.07       0.7778          1.1357            1.0498        12.46
SOAROPQ-orth1-m32-nl111-np1                            3_223.12        73.76     3_296.88       0.8622          1.0564            1.0265         9.96
IVFOPQ-m32-nl111-np2                                   3_040.20        73.71     3_113.90       0.8792          1.0372            1.0196         7.46
IVFOPQ-m64-nl111-np2                                   3_403.29        96.33     3_499.62       0.8848          1.0339            1.0168        12.46
SOAROPQ-orth1-m32-nl111-np2                            3_223.12        89.40     3_312.52       0.8943          1.0287            1.0173         9.96
IVFOPQ-m32-nl111-np4                                   3_040.20        95.19     3_135.39       0.9006          1.0223            1.0155         7.46
IVFOPQ-m64-nl111-np4                                   3_403.29       137.44     3_540.73       0.9071          1.0189            1.0130        12.46
SOAROPQ-orth1-m32-nl111-np4                            3_223.12       120.86     3_343.98       0.9008          1.0232            1.0156         9.96
IVFOPQ-m32-nl111-np5                                   3_040.20       108.97     3_149.16       0.9015          1.0218            1.0153         7.46
IVFOPQ-m64-nl111-np5                                   3_403.29       158.18     3_561.47       0.9081          1.0184            1.0127        12.46
SOAROPQ-orth1-m32-nl111-np5                            3_223.12       137.78     3_360.90       0.9014          1.0226            1.0154         9.96
IVFOPQ-m32-nl111-np8                                   3_040.20       150.81     3_191.00       0.9020          1.0216            1.0151         7.46
IVFOPQ-m64-nl111-np8                                   3_403.29       229.93     3_633.22       0.9088          1.0181            1.0125        12.46
SOAROPQ-orth1-m32-nl111-np8                            3_223.12       184.10     3_407.22       0.9019          1.0218            1.0151         9.96
IVFOPQ-m32-nl111-np10                                  3_040.20       167.38     3_207.58       0.9021          1.0216            1.0151         7.46
IVFOPQ-m64-nl111-np10                                  3_403.29       272.30     3_675.59       0.9089          1.0181            1.0125        12.46
SOAROPQ-orth1-m32-nl111-np10                           3_223.12       212.98     3_436.09       0.9020          1.0217            1.0151         9.96
IVFOPQ-m32-nl158-np1                                   3_360.55        60.64     3_421.19       0.7615          1.1488            1.0665         9.27
IVFOPQ-m64-nl158-np1                                   3_870.88        70.99     3_941.87       0.7638          1.1465            1.0637        15.74
SOAROPQ-orth1-m32-nl158-np1                            3_699.49        68.38     3_767.87       0.8579          1.0630            1.0259        11.79
IVFOPQ-m32-nl158-np2                                   3_360.55        74.30     3_434.85       0.8775          1.0392            1.0188         9.27
IVFOPQ-m64-nl158-np2                                   3_870.88        91.54     3_962.42       0.8819          1.0371            1.0168        15.74
SOAROPQ-orth1-m32-nl158-np2                            3_699.49        85.00     3_784.50       0.8982          1.0286            1.0155        11.79
IVFOPQ-m32-nl158-np4                                   3_360.55        90.88     3_451.43       0.9071          1.0189            1.0130         9.27
IVFOPQ-m64-nl158-np4                                   3_870.88       127.97     3_998.85       0.9119          1.0171            1.0115        15.74
SOAROPQ-orth1-m32-nl158-np4                            3_699.49       109.80     3_809.29       0.9077          1.0205            1.0131        11.79
IVFOPQ-m32-nl158-np7                                   3_360.55       116.25     3_476.80       0.9096          1.0176            1.0124         9.27
IVFOPQ-m64-nl158-np7                                   3_870.88       179.50     4_050.38       0.9145          1.0158            1.0109        15.74
SOAROPQ-orth1-m32-nl158-np7                            3_699.49       150.10     3_849.59       0.9094          1.0183            1.0125        11.79
IVFOPQ-m32-nl158-np8                                   3_360.55       125.21     3_485.76       0.9098          1.0175            1.0124         9.27
IVFOPQ-m64-nl158-np8                                   3_870.88       195.61     4_066.49       0.9147          1.0157            1.0108        15.74
SOAROPQ-orth1-m32-nl158-np8                            3_699.49       165.02     3_864.51       0.9095          1.0181            1.0125        11.79
IVFOPQ-m32-nl158-np12                                  3_360.55       184.21     3_544.77       0.9098          1.0175            1.0124         9.27
IVFOPQ-m64-nl158-np12                                  3_870.88       266.68     4_137.56       0.9147          1.0157            1.0108        15.74
SOAROPQ-orth1-m32-nl158-np12                           3_699.49       200.37     3_899.86       0.9098          1.0176            1.0124        11.79
IVFOPQ-m32-nl223-np1                                   3_434.32        59.87     3_494.19       0.7350          1.1743            1.0954        11.43
IVFOPQ-m64-nl223-np1                                   3_819.31        68.50     3_887.81       0.7371          1.1717            1.0929        19.93
SOAROPQ-orth1-m32-nl223-np1                            3_707.05        64.09     3_771.14       0.8488          1.0698            1.0308        13.94
IVFOPQ-m32-nl223-np2                                   3_434.32        65.72     3_500.04       0.8710          1.0451            1.0209        11.43
IVFOPQ-m64-nl223-np2                                   3_819.31        81.30     3_900.60       0.8752          1.0426            1.0184        19.93
SOAROPQ-orth1-m32-nl223-np2                            3_707.05        73.64     3_780.69       0.8987          1.0296            1.0151        13.94
IVFOPQ-m32-nl223-np4                                   3_434.32        78.20     3_512.53       0.9096          1.0183            1.0120        11.43
IVFOPQ-m64-nl223-np4                                   3_819.31       105.51     3_924.82       0.9156          1.0157            1.0102        19.93
SOAROPQ-orth1-m32-nl223-np4                            3_707.05        91.39     3_798.44       0.9099          1.0206            1.0123        13.94
IVFOPQ-m32-nl223-np8                                   3_434.32       103.98     3_538.31       0.9129          1.0166            1.0113        11.43
IVFOPQ-m64-nl223-np8                                   3_819.31       155.65     3_974.95       0.9193          1.0140            1.0094        19.93
SOAROPQ-orth1-m32-nl223-np8                            3_707.05       130.05     3_837.10       0.9126          1.0174            1.0115        13.94
IVFOPQ-m32-nl223-np11                                  3_434.32       123.83     3_558.15       0.9131          1.0165            1.0113        11.43
IVFOPQ-m64-nl223-np11                                  3_819.31       191.93     4_011.24       0.9194          1.0140            1.0093        19.93
SOAROPQ-orth1-m32-nl223-np11                           3_707.05       157.82     3_864.87       0.9130          1.0168            1.0113        13.94
IVFOPQ-m32-nl223-np14                                  3_434.32       142.51     3_576.83       0.9131          1.0165            1.0113        11.43
IVFOPQ-m64-nl223-np14                                  3_819.31       232.81     4_052.11       0.9194          1.0140            1.0093        19.93
SOAROPQ-orth1-m32-nl223-np14                           3_707.05       182.18     3_889.23       0.9131          1.0166            1.0113        13.94
IVFOPQ-m32-nl316-np1                                   3_512.15        61.15     3_573.30       0.7069          1.2052            1.1249        15.02
IVFOPQ-m64-nl316-np1                                   3_992.23        67.96     4_060.20       0.7085          1.2033            1.1226        26.42
SOAROPQ-orth1-m32-nl316-np1                            3_864.56        63.86     3_928.42       0.8365          1.0789            1.0355        17.57
IVFOPQ-m32-nl316-np2                                   3_512.15        65.28     3_577.44       0.8592          1.0540            1.0239        15.02
IVFOPQ-m64-nl316-np2                                   3_992.23        80.27     4_072.51       0.8628          1.0520            1.0215        26.42
SOAROPQ-orth1-m32-nl316-np2                            3_864.56        72.71     3_937.26       0.9007          1.0281            1.0144        17.57
IVFOPQ-m32-nl316-np4                                   3_512.15        75.30     3_587.45       0.9127          1.0170            1.0107        15.02
IVFOPQ-m64-nl316-np4                                   3_992.23        98.82     4_091.05       0.9183          1.0150            1.0089        26.42
SOAROPQ-orth1-m32-nl316-np4                            3_864.56        85.38     3_949.93       0.9147          1.0182            1.0106        17.57
IVFOPQ-m32-nl316-np8                                   3_512.15        95.94     3_608.10       0.9177          1.0143            1.0097        15.02
IVFOPQ-m64-nl316-np8                                   3_992.23       140.33     4_132.56       0.9240          1.0122            1.0078        26.42
SOAROPQ-orth1-m32-nl316-np8                            3_864.56       110.93     3_975.49       0.9175          1.0151            1.0099        17.57
IVFOPQ-m32-nl316-np15                                  3_512.15       132.34     3_644.49       0.9181          1.0142            1.0096        15.02
IVFOPQ-m64-nl316-np15                                  3_992.23       213.59     4_205.82       0.9244          1.0121            1.0077        26.42
SOAROPQ-orth1-m32-nl316-np15                           3_864.56       164.62     4_029.17       0.9180          1.0142            1.0097        17.57
IVFOPQ-m32-nl316-np17                                  3_512.15       139.14     3_651.29       0.9181          1.0141            1.0096        15.02
IVFOPQ-m64-nl316-np17                                  3_992.23       238.68     4_230.91       0.9244          1.0121            1.0077        26.42
SOAROPQ-orth1-m32-nl316-np17                           3_864.56       171.38     4_035.94       0.9180          1.0142            1.0096        17.57
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
Exhaustive (query)                                        72.24     1_358.50     1_430.74       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                       3_553.14        68.34     3_621.49       0.8640          1.0547            1.0244        11.77
SOAROPQ-near-np2                                       3_553.14        82.02     3_635.17       0.9004          1.0253            1.0151        11.77
SOAROPQ-near-np4                                       3_553.14       107.89     3_661.03       0.9084          1.0190            1.0128        11.77
SOAROPQ-near-np7                                       3_553.14       145.21     3_698.35       0.9096          1.0177            1.0124        11.77
SOAROPQ-near-np8                                       3_553.14       160.39     3_713.53       0.9097          1.0176            1.0124        11.77
SOAROPQ-near-np12                                      3_553.14       203.01     3_756.15       0.9098          1.0175            1.0124        11.77
SOAROPQ-shift0.3-np1                                   3_533.34        70.30     3_603.64       0.8627          1.0570            1.0257        11.79
SOAROPQ-shift0.3-np2                                   3_533.34        83.65     3_616.99       0.8989          1.0272            1.0155        11.79
SOAROPQ-shift0.3-np4                                   3_533.34       107.78     3_641.13       0.9075          1.0202            1.0131        11.79
SOAROPQ-shift0.3-np7                                   3_533.34       149.99     3_683.33       0.9094          1.0181            1.0125        11.79
SOAROPQ-shift0.3-np8                                   3_533.34       160.14     3_693.49       0.9095          1.0179            1.0125        11.79
SOAROPQ-shift0.3-np12                                  3_533.34       201.45     3_734.79       0.9098          1.0176            1.0124        11.79
SOAROPQ-shift0.7-np1                                   3_532.02        70.41     3_602.43       0.8573          1.0641            1.0271        11.79
SOAROPQ-shift0.7-np2                                   3_532.02        84.28     3_616.30       0.8958          1.0315            1.0162        11.79
SOAROPQ-shift0.7-np4                                   3_532.02       111.08     3_643.10       0.9060          1.0224            1.0135        11.79
SOAROPQ-shift0.7-np7                                   3_532.02       152.37     3_684.39       0.9089          1.0192            1.0127        11.79
SOAROPQ-shift0.7-np8                                   3_532.02       162.30     3_694.32       0.9092          1.0187            1.0126        11.79
SOAROPQ-shift0.7-np12                                  3_532.02       204.79     3_736.81       0.9097          1.0177            1.0124        11.79
SOAROPQ-orth1-np1                                      3_984.95        72.90     4_057.85       0.8579          1.0630            1.0259        11.79
SOAROPQ-orth1-np2                                      3_984.95        84.30     4_069.25       0.8982          1.0286            1.0155        11.79
SOAROPQ-orth1-np4                                      3_984.95       108.18     4_093.13       0.9077          1.0205            1.0131        11.79
SOAROPQ-orth1-np7                                      3_984.95       145.26     4_130.21       0.9094          1.0183            1.0125        11.79
SOAROPQ-orth1-np8                                      3_984.95       157.77     4_142.72       0.9095          1.0181            1.0125        11.79
SOAROPQ-orth1-np12                                     3_984.95       205.14     4_190.09       0.9098          1.0176            1.0124        11.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
