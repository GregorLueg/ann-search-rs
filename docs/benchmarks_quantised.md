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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.17       647.68       658.84       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.17     6_239.02     6_250.19       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.31     1_205.50     1_217.81       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    12.31    11_733.97    11_746.27       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                               79.10        84.47       163.57       0.9793          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                               79.10        97.17       176.27       0.9827          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                               79.10       128.77       207.87       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                     79.10     1_312.57     1_391.67       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                               93.22        89.42       182.63       0.9807          1.0002            1.0000         9.20
IVF-BF16-nl387-np27 (query)                               93.22       114.02       207.24       0.9828          1.0001            1.0000         9.20
IVF-BF16-nl387 (self)                                     93.22     1_158.98     1_252.20       0.9798          1.0001            1.0000         9.20
IVF-BF16-nl547-np23 (query)                              134.73        80.14       214.87       0.9768          1.0005            1.0000         9.22
IVF-BF16-nl547-np27 (query)                              134.73        88.53       223.25       0.9812          1.0002            1.0000         9.22
IVF-BF16-nl547-np33 (query)                              134.73       103.89       238.62       0.9827          1.0001            1.0000         9.22
IVF-BF16-nl547 (self)                                    134.73     1_048.89     1_183.62       0.9798          1.0001            1.0000         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.74       706.03       717.77       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.74     6_876.79     6_888.53       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   12.84     1_205.33     1_218.16       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    12.84    12_473.24    12_486.08       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              105.62        92.97       198.59       0.8841          1.0075            1.0021         9.48
IVF-BF16-nl273-np16 (query)                              105.62       107.30       212.91       0.8867          1.0072            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              105.62       147.32       252.94       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    105.62     1_497.84     1_603.46       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              120.56        94.53       215.09       0.8861          1.0073            1.0020         9.49
IVF-BF16-nl387-np27 (query)                              120.56       125.83       246.40       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    120.56     1_273.18     1_393.74       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              176.37        87.32       263.69       0.8841          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              176.37        95.94       272.31       0.8862          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              176.37       113.72       290.10       0.8871          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    176.37     1_149.34     1_325.72       0.8852          1.0073            1.0020         9.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.23       642.59       653.82       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.23     6_394.68     6_405.91       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.22     1_198.20     1_211.43       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    13.22    11_845.24    11_858.46       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                               94.17        82.81       176.98       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                               94.17        94.08       188.25       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                               94.17       116.11       210.27       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                     94.17     1_156.04     1_250.21       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              104.19        84.52       188.71       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387-np27 (query)                              104.19       102.87       207.06       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387 (self)                                    104.19     1_037.67     1_141.86       0.9184          1.0030            1.0021         9.20
IVF-BF16-nl547-np23 (query)                              156.56        78.27       234.83       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np27 (query)                              156.56        84.92       241.47       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np33 (query)                              156.56        96.19       252.75       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547 (self)                                    156.56       974.93     1_131.48       0.9184          1.0030            1.0021         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.50       663.05       674.56       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.50     6_186.68     6_198.19       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.67     1_178.39     1_193.06       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    14.67    11_800.88    11_815.55       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              185.35        78.63       263.97       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              185.35        88.71       274.06       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              185.35       115.17       300.52       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    185.35     1_167.38     1_352.73       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              117.18        80.82       198.00       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387-np27 (query)                              117.18       102.10       219.28       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387 (self)                                    117.18     1_012.26     1_129.44       0.9429          1.0017            1.0009         9.20
IVF-BF16-nl547-np23 (query)                              306.08        85.75       391.83       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np27 (query)                              306.08        84.20       390.28       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np33 (query)                              306.08        96.64       402.72       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547 (self)                                    306.08       965.51     1_271.60       0.9429          1.0017            1.0009         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        50.11     1_219.49     1_269.60       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.11    11_924.80    11_974.91       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   59.30     5_165.97     5_225.27       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    59.30    52_592.32    52_651.62       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              186.42       262.17       448.59       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              186.42       299.70       486.12       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              186.42       404.84       591.26       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    186.42     4_140.74     4_327.16       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              270.31       314.19       584.50       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              270.31       575.11       845.42       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    270.31     3_750.01     4_020.32       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                              385.81       255.37       641.18       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                              385.81       282.67       668.48       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                              385.81       325.22       711.03       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                    385.81     3_240.34     3_626.16       0.9679          1.0005            1.0000        36.89
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.12       646.80       657.91       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.12     6_217.47     6_228.59       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.80       966.42       982.22       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     15.80     9_787.85     9_803.65       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                                92.07        63.92       155.99       0.9227          1.0021            1.0010         5.76
IVF-SQ8-nl273-np16 (query)                                92.07        68.46       160.53       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273-np23 (query)                                92.07        89.47       181.54       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273 (self)                                      92.07       880.32       972.39       0.9249          1.0018            1.0009         5.76
IVF-SQ8-nl387-np19 (query)                               102.54        62.62       165.15       0.9243          1.0020            1.0009         5.77
IVF-SQ8-nl387-np27 (query)                               102.54        79.51       182.04       0.9257          1.0018            1.0009         5.77
IVF-SQ8-nl387 (self)                                     102.54       786.84       889.37       0.9251          1.0018            1.0009         5.77
IVF-SQ8-nl547-np23 (query)                               146.00        56.14       202.14       0.9210          1.0023            1.0011         5.79
IVF-SQ8-nl547-np27 (query)                               146.00        62.86       208.86       0.9239          1.0020            1.0010         5.79
IVF-SQ8-nl547-np33 (query)                               146.00        73.54       219.54       0.9249          1.0018            1.0009         5.79
IVF-SQ8-nl547 (self)                                     146.00       755.91       901.91       0.9250          1.0018            1.0009         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.79       703.97       715.75       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.79     7_035.90     7_047.68       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    20.10       901.53       921.63       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     20.10    10_033.81    10_053.92       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                                94.14        60.23       154.37       0.7379          1.0360            1.0158         5.76
IVF-SQ8-nl273-np16 (query)                                94.14        67.45       161.59       0.7387          1.0359            1.0157         5.76
IVF-SQ8-nl273-np23 (query)                                94.14        91.19       185.34       0.7385          1.0360            1.0157         5.76
IVF-SQ8-nl273 (self)                                      94.14       943.64     1_037.78       0.7376          1.0361            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               122.48        59.38       181.86       0.7370          1.0360            1.0159         5.77
IVF-SQ8-nl387-np27 (query)                               122.48        76.47       198.95       0.7373          1.0359            1.0158         5.77
IVF-SQ8-nl387 (self)                                     122.48       828.47       950.95       0.7360          1.0363            1.0158         5.77
IVF-SQ8-nl547-np23 (query)                               158.50        54.07       212.57       0.7386          1.0358            1.0161         5.79
IVF-SQ8-nl547-np27 (query)                               158.50        61.26       219.76       0.7395          1.0356            1.0156         5.79
IVF-SQ8-nl547-np33 (query)                               158.50        71.18       229.68       0.7397          1.0356            1.0155         5.79
IVF-SQ8-nl547 (self)                                     158.50       751.06       909.56       0.7379          1.0361            1.0154         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.39       623.79       635.18       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.39     6_242.16     6_253.55       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.80       964.19       980.99       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     16.80     9_732.83     9_749.63       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                                85.90        59.02       144.92       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np16 (query)                                85.90        64.11       150.01       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np23 (query)                                85.90        79.72       165.62       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273 (self)                                      85.90       783.39       869.29       0.8118          1.0175            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               100.90        58.45       159.36       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387-np27 (query)                               100.90        70.39       171.29       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387 (self)                                     100.90       688.76       789.66       0.8119          1.0175            1.0155         5.77
IVF-SQ8-nl547-np23 (query)                               142.27        54.69       196.96       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np27 (query)                               142.27        59.63       201.90       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np33 (query)                               142.27        69.10       211.37       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547 (self)                                     142.27       669.75       812.02       0.8120          1.0175            1.0155         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.28       649.98       661.26       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.28     6_101.46     6_112.73       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.98       964.83       980.80       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     15.98     9_711.03     9_727.01       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                                86.70        56.36       143.06       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np16 (query)                                86.70        62.08       148.78       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np23 (query)                                86.70        79.24       165.94       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273 (self)                                      86.70       780.29       866.99       0.7901          1.0280            1.0256         5.76
IVF-SQ8-nl387-np19 (query)                                97.49        56.16       153.65       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387-np27 (query)                                97.49        70.12       167.61       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387 (self)                                      97.49       689.56       787.05       0.7901          1.0281            1.0256         5.77
IVF-SQ8-nl547-np23 (query)                               145.38        53.56       198.94       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np27 (query)                               145.38        58.59       203.97       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np33 (query)                               145.38        65.31       210.69       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547 (self)                                     145.38       647.37       792.75       0.7903          1.0280            1.0256         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### More dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.63     1_209.84     1_258.48       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.63    11_871.58    11_920.21       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    84.71     1_190.77     1_275.48       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     84.71    12_012.50    12_097.21       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               202.07        75.63       277.70       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np16 (query)                               202.07        81.89       283.96       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np23 (query)                               202.07       105.13       307.20       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273 (self)                                     202.07       855.95     1_058.02       0.8866          1.0073            1.0059        19.59
IVF-SQ8-nl387-np19 (query)                               253.07        82.37       335.44       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387-np27 (query)                               253.07       110.04       363.11       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387 (self)                                     253.07       796.04     1_049.10       0.8866          1.0073            1.0059        19.65
IVF-SQ8-nl547-np23 (query)                               362.99        82.36       445.36       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np27 (query)                               362.99        86.61       449.60       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np33 (query)                               362.99        98.72       461.71       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547 (self)                                     362.99       760.46     1_123.46       0.8867          1.0073            1.0059        19.73
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.27       629.49       640.76       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.27     6_134.06     6_145.33       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.82     1_018.56     1_034.38       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               766.98        45.59       812.57       0.9275          1.0173            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              766.98        81.03       848.01       0.9608          1.0116            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              766.98       150.69       917.67       0.9793          1.0070            1.0000        38.52
HNSW-M16-ef100 (self)                                    766.98       801.36     1_568.34       0.9610          1.0116            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_475.81        48.99     1_524.80       0.9583          1.0131            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_475.81        87.79     1_563.60       0.9826          1.0050            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_475.81       158.94     1_634.75       0.9921          1.0012            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_475.81       852.81     2_328.62       0.9830          1.0057            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_594.60        54.41     1_649.01       0.9694          1.0212            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_594.60        97.50     1_692.10       0.9878          1.0099            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_594.60       183.00     1_777.60       0.9950          1.0039            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_594.60     1_130.38     2_724.98       0.9884          1.0100            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_706.19        58.62     1_764.82       0.9705          1.1308            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_706.19       102.52     1_808.71       0.9889          1.0249            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_706.19       184.09     1_890.28       0.9957          1.0020            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_706.19     1_010.36     2_716.55       0.9894          1.0288            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          750.14        39.74       789.88       0.8768          1.0162            1.0032        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         750.14        67.77       817.90       0.9013          1.0116            1.0020        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         750.14       120.50       870.64       0.9147          1.0076            1.0014        26.32
HNSW-SQ8U-M16-ef100 (self)                               750.14       598.72     1_348.85       0.9012          1.0105            1.0020        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_331.17        35.92     1_367.09       0.8963          1.0531            1.0022        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_331.17        66.98     1_398.15       0.9127          1.0368            1.0014        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_331.17       127.98     1_459.15       0.9199          1.0114            1.0011        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_331.17       645.03     1_976.21       0.9129          1.0361            1.0014        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_438.30        39.47     1_477.77       0.9066          1.0092            1.0017        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_438.30        73.97     1_512.27       0.9185          1.0047            1.0012        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_438.30       133.70     1_572.00       0.9230          1.0023            1.0010        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_438.30       697.72     2_136.02       0.9179          1.0053            1.0012        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_527.10        41.23     1_568.33       0.9087          1.0067            1.0016        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_527.10        77.70     1_604.80       0.9195          1.0047            1.0012        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_527.10       144.81     1_671.91       0.9233          1.0027            1.0010        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_527.10       726.46     2_253.56       0.9190          1.0045            1.0012        44.63
HNSW-SQ8U-drop0 (query)                                1_340.59        67.71     1_408.30       0.8946          1.0116            1.0024        26.32
HNSW-SQ8U-drop0.001 (query)                            1_329.65        67.50     1_397.14       0.9130          1.0080            1.0014        26.32
HNSW-SQ8U-drop0.01 (query)                             1_336.07        72.44     1_408.51       0.8986          1.0072            1.0018        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.82       688.11       699.93       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.82     6_765.48     6_777.29       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    18.67       897.34       916.01       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               851.84        48.50       900.34       0.9352          1.0176            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              851.84        84.50       936.34       0.9692          1.0105            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              851.84       154.71     1_006.55       0.9873          1.0053            1.0000        39.09
HNSW-M16-ef100 (self)                                    851.84       831.06     1_682.90       0.9694          1.0090            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_541.87        53.24     1_595.11       0.9650          1.0071            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_541.87        90.98     1_632.85       0.9876          1.0022            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_541.87       166.83     1_708.70       0.9952          1.0013            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_541.87       882.66     2_424.53       0.9872          1.0042            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_653.04        56.64     1_709.67       0.9742          1.0022            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_653.04       102.79     1_755.83       0.9914          1.0006            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_653.04       182.21     1_835.25       0.9972          1.0002            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_653.04       996.78     2_649.82       0.9911          1.0014            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_706.59        59.62     1_766.20       0.9748          1.0489            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_706.59       104.79     1_811.38       0.9913          1.0138            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_706.59       185.97     1_892.56       0.9974          1.0004            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_706.59     1_047.24     2_753.82       0.9915          1.0097            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          706.39        34.81       741.20       0.6850          1.0559            1.0289        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         706.39        63.78       770.17       0.7091          1.0486            1.0238        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         706.39       120.90       827.29       0.7227          1.0427            1.0208        26.32
HNSW-SQ8U-M16-ef100 (self)                               706.39       612.25     1_318.64       0.7085          1.0494            1.0238        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_352.61        36.21     1_388.82       0.7058          1.0647            1.0236        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_352.61        67.55     1_420.16       0.7234          1.0532            1.0201        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_352.61       126.70     1_479.31       0.7312          1.0453            1.0184        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_352.61       658.21     2_010.82       0.7227          1.0512            1.0200        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_481.98        45.92     1_527.90       0.7181          1.0424            1.0208        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_481.98        83.99     1_565.97       0.7304          1.0398            1.0186        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_481.98       133.89     1_615.87       0.7354          1.0373            1.0175        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_481.98       705.52     2_187.50       0.7296          1.0387            1.0182        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_550.17        41.94     1_592.11       0.7224          1.0421            1.0195        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_550.17        77.40     1_627.58       0.7327          1.0374            1.0177        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_550.17       139.76     1_689.93       0.7367          1.0363            1.0169        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_550.17       739.91     2_290.08       0.7321          1.0374            1.0174        44.63
HNSW-SQ8U-drop0 (query)                                1_343.29        65.38     1_408.67       0.6644          1.0665            1.0307        26.32
HNSW-SQ8U-drop0.001 (query)                            1_350.28        66.51     1_416.80       0.7244          1.0417            1.0198        26.32
HNSW-SQ8U-drop0.01 (query)                             1_318.88        65.35     1_384.23       0.6887          1.0544            1.0263        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.19       634.56       645.75       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.19     6_301.27     6_312.46       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.78       968.29       984.07       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               763.77        48.16       811.93       0.9976          1.3193            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              763.77        81.43       845.20       0.9996          1.0001            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              763.77       142.01       905.77       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    763.77       788.86     1_552.63       0.9996          1.0272            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_363.80        48.35     1_412.15       0.9972          1.7282            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_363.80        84.30     1_448.10       0.9996          1.0669            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_363.80       142.31     1_506.11       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_363.80       791.97     2_155.77       0.9998          1.0176            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_385.85        48.75     1_434.60       0.9991          1.0001            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_385.85        85.71     1_471.56       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_385.85       149.25     1_535.09       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_385.85       813.24     2_199.09       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_432.87        51.48     1_484.34       0.9993          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_432.87        85.75     1_518.61       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_432.87       148.99     1_581.86       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_432.87       826.00     2_258.86       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          705.40        35.36       740.76       0.8135          1.0568            1.0149        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         705.40        63.34       768.75       0.8144          1.0269            1.0148        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         705.40       113.83       819.23       0.8145          1.0268            1.0148        26.32
HNSW-SQ8U-M16-ef100 (self)                               705.40       606.17     1_311.57       0.8116          1.0479            1.0156        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_235.26        34.04     1_269.30       0.8142          1.0166            1.0148        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_235.26        63.26     1_298.52       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_235.26       114.43     1_349.69       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_235.26       588.44     1_823.70       0.8119          1.0175            1.0155        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_314.26        36.83     1_351.10       0.8144          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_314.26        66.94     1_381.21       0.8145          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_314.26       121.36     1_435.62       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_314.26       631.61     1_945.87       0.8119          1.0175            1.0155        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_385.12        43.16     1_428.28       0.8144          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_385.12        71.56     1_456.68       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_385.12       123.73     1_508.85       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_385.12       655.18     2_040.31       0.8119          1.0175            1.0155        44.63
HNSW-SQ8U-drop0 (query)                                1_261.01        62.92     1_323.93       0.8078          1.2521            1.0157        26.32
HNSW-SQ8U-drop0.001 (query)                            1_302.34        61.77     1_364.11       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-drop0.01 (query)                             1_227.50        60.28     1_287.78       0.8037          1.4287            1.0164        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.41       639.54       650.95       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.41     6_432.21     6_443.62       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.30       972.05       988.35       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               840.39        50.28       890.67       0.9980          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              840.39        87.63       928.02       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              840.39       165.98     1_006.37       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    840.39       873.86     1_714.25       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_580.09        51.27     1_631.36       0.9986          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_580.09        89.93     1_670.02       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_580.09       162.44     1_742.53       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_580.09       896.17     2_476.26       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_915.73        70.35     1_986.08       0.9993          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_915.73       118.50     2_034.24       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_915.73       205.96     2_121.70       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_915.73     1_088.69     3_004.43       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_757.55        71.44     1_828.99       0.9994          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_757.55       116.17     1_873.72       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_757.55       193.69     1_951.24       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_757.55     1_077.12     2_834.67       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          814.02        37.66       851.68       0.7888          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         814.02        97.70       911.72       0.7892          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         814.02       148.77       962.80       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100 (self)                               814.02       771.62     1_585.64       0.7896          1.0282            1.0258        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_640.39        41.99     1_682.38       0.7890          1.0267            1.0244        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_640.39        83.36     1_723.75       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_640.39       154.20     1_794.58       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_640.39       806.28     2_446.67       0.7897          1.0281            1.0258        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_719.67        57.92     1_777.59       0.7892          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_719.67        85.58     1_805.25       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_719.67       154.96     1_874.63       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_719.67       798.03     2_517.69       0.7897          1.0281            1.0258        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_747.13        49.10     1_796.22       0.7891          1.0274            1.0244        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_747.13        87.25     1_834.38       0.7892          1.0274            1.0244        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_747.13       157.04     1_904.17       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_747.13       837.08     2_584.21       0.7896          1.0287            1.0258        44.63
HNSW-SQ8U-drop0 (query)                                1_433.80        71.10     1_504.89       0.7860          1.0279            1.0254        26.32
HNSW-SQ8U-drop0.001 (query)                            1_369.89        86.13     1_456.03       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-drop0.01 (query)                             1_361.41        67.80     1_429.21       0.7830          1.0294            1.0262        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.79     1_231.03     1_279.83       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.79    12_719.30    12_768.09       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    80.06     1_231.38     1_311.45       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_368.64        79.99     1_448.62       0.9939          1.0287            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_368.64       141.71     1_510.35       0.9958          1.0171            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_368.64       251.76     1_620.40       0.9975          1.0078            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_368.64     1_483.40     2_852.04       0.9959          1.0167            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_655.12        99.71     2_754.83       0.9974          1.0140            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_655.12       163.40     2_818.51       0.9984          1.0081            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_655.12       284.38     2_939.50       0.9992          1.0035            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_655.12     1_463.91     4_119.03       0.9980          1.0113            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_638.65        90.78     2_729.42       0.9972          1.0172            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_638.65       159.69     2_798.34       0.9984          1.0093            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_638.65       275.55     2_914.20       0.9994          1.0029            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_638.65     1_494.02     4_132.67       0.9986          1.0081            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_686.33        90.75     2_777.09       0.9991          1.0037            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_686.33       153.29     2_839.62       0.9996          1.0019            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_686.33       277.82     2_964.16       0.9997          1.0013            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_686.33     1_513.91     4_200.24       0.9996          1.0020            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          852.37        41.57       893.94       0.9289          1.0318            1.0038        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         852.37        75.06       927.44       0.9307          1.0218            1.0038        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         852.37       126.18       978.55       0.9324          1.0109            1.0038        40.05
HNSW-SQ8U-M16-ef100 (self)                               852.37       671.47     1_523.85       0.9303          1.0240            1.0038        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_544.71        39.96     1_584.67       0.9302          1.0303            1.0037        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_544.71        71.32     1_616.03       0.9319          1.0207            1.0037        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_544.71       124.73     1_669.44       0.9332          1.0129            1.0036        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_544.71       658.88     2_203.59       0.9319          1.0201            1.0037        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_627.25        42.60     1_669.85       0.9318          1.0221            1.0036        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_627.25        74.63     1_701.88       0.9330          1.0138            1.0036        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_627.25       131.01     1_758.26       0.9335          1.0097            1.0036        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_627.25       702.57     2_329.82       0.9327          1.0149            1.0036        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_714.85        43.47     1_758.31       0.9334          1.0107            1.0036        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_714.85        74.73     1_789.57       0.9337          1.0090            1.0036        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_714.85       135.45     1_850.30       0.9340          1.0076            1.0036        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_714.85       721.60     2_436.45       0.9335          1.0096            1.0036        58.36
HNSW-SQ8U-drop0 (query)                                1_525.44        68.49     1_593.93       0.8647          1.0381            1.0221        40.05
HNSW-SQ8U-drop0.001 (query)                            1_508.13        71.20     1_579.32       0.9323          1.0194            1.0036        40.05
HNSW-SQ8U-drop0.01 (query)                             1_549.86        71.30     1_621.17       0.9321          1.0395            1.0020        40.05
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        53.46     1_329.62     1_383.08       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         53.46    13_246.37    13_299.83       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    89.56     1_268.02     1_357.58       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_271.99        74.06     1_346.05       0.9945          1.0841            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_271.99       118.74     1_390.73       0.9972          1.0283            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_271.99       208.73     1_480.72       0.9981          1.0170            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_271.99     1_189.93     2_461.92       0.9968          1.0362            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_277.81        72.51     2_350.32       0.9933          1.1570            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_277.81       124.57     2_402.38       0.9964          1.0751            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_277.81       220.17     2_497.98       0.9980          1.0398            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_277.81     1_204.29     3_482.10       0.9960          1.0754            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_448.36       106.12     2_554.48       0.9988          1.0150            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_448.36       151.14     2_599.49       0.9993          1.0078            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_448.36       246.32     2_694.68       0.9997          1.0023            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_448.36     1_313.08     3_761.44       0.9991          1.0104            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_523.01        85.37     2_608.38       0.9992          1.0188            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_523.01       131.11     2_654.13       0.9996          1.0075            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_523.01       236.16     2_759.18       0.9998          1.0048            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_523.01     1_258.29     3_781.31       0.9995          1.0092            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          817.25        40.22       857.47       0.6643          1.4098            1.1635        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         817.25        66.99       884.24       0.6658          1.3756            1.1624        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         817.25       120.30       937.56       0.6665          1.3626            1.1621        40.05
HNSW-SQ8U-M16-ef100 (self)                               817.25       622.43     1_439.68       0.6654          1.3773            1.1630        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_546.31        37.63     1_583.94       0.6634          1.4336            1.1638        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_546.31        64.94     1_611.25       0.6644          1.4093            1.1632        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_546.31       117.38     1_663.68       0.6659          1.3768            1.1622        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_546.31       615.86     2_162.17       0.6644          1.4046            1.1635        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_624.91        43.53     1_668.44       0.6665          1.3610            1.1619        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_624.91        79.74     1_704.65       0.6670          1.3541            1.1616        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_624.91       141.43     1_766.34       0.6673          1.3497            1.1614        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_624.91       673.77     2_298.68       0.6667          1.3578            1.1618        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_680.97        44.46     1_725.43       0.6660          1.3698            1.1621        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_680.97        75.79     1_756.76       0.6666          1.3615            1.1618        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_680.97       134.98     1_815.95       0.6672          1.3521            1.1614        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_680.97       698.59     2_379.57       0.6666          1.3608            1.1619        58.36
HNSW-SQ8U-drop0 (query)                                1_574.26        72.25     1_646.51       0.6202          1.4945            1.2357        40.05
HNSW-SQ8U-drop0.001 (query)                            1_494.95        68.33     1_563.28       0.6648          1.4013            1.1629        40.05
HNSW-SQ8U-drop0.01 (query)                             1_479.35        69.09     1_548.44       0.6793          1.3386            1.1503        40.05
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.71       699.61       732.32       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.71     2_340.21     2_372.92       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                328.71       505.88       834.58       0.2581          1.1827            1.1592         1.26
Exhaustive-PQ-m16 (self)                                 328.71     1_663.98     1_992.68       0.2365          1.1998            1.1748         1.26
Exhaustive-PQ-m32 (query)                                353.87       920.04     1_273.91       0.2961          1.1446            1.1423         2.03
Exhaustive-PQ-m32 (self)                                 353.87     2_927.45     3_281.32       0.2627          1.1633            1.1601         2.03
Exhaustive-PQ-m64 (query)                                454.50     2_021.61     2_476.11       0.3611          1.1111            1.1080         3.55
Exhaustive-PQ-m64 (self)                                 454.50     6_406.15     6_860.65       0.3106          1.1303            1.1270         3.55
IVF-PQ-nl158-m16-np7 (query)                             534.53        66.50       601.03       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16-np12 (query)                            534.53        76.45       610.98       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16-np17 (query)                            534.53        83.51       618.03       0.3766          1.0939            1.0961         3.89
IVF-PQ-nl158-m16 (self)                                  534.53       236.23       770.75       0.3056          1.1267            1.1313         3.89
IVF-PQ-nl158-m32-np7 (query)                             624.31        96.70       721.02       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32-np12 (query)                            624.31       139.54       763.86       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32-np17 (query)                            624.31       140.06       764.37       0.4874          1.0591            1.0556         7.12
IVF-PQ-nl158-m32 (self)                                  624.31       409.52     1_033.83       0.4073          1.0805            1.0788         7.12
IVF-PQ-nl158-m64-np7 (query)                             644.58       138.61       783.19       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64-np12 (query)                            644.58       179.48       824.06       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64-np17 (query)                            644.58       218.23       862.81       0.6937          1.0202            1.0162        13.58
IVF-PQ-nl158-m64 (self)                                  644.58       703.70     1_348.28       0.6343          1.0278            1.0241        13.58
IVF-PQ-nl223-m16-np11 (query)                            455.85        69.33       525.18       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16-np14 (query)                            455.85        76.99       532.84       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16-np21 (query)                            455.85        90.67       546.52       0.3901          1.0871            1.0885         4.97
IVF-PQ-nl223-m16 (self)                                  455.85       252.18       708.03       0.3120          1.1216            1.1261         4.97
IVF-PQ-nl223-m32-np11 (query)                            502.66       108.62       611.27       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32-np14 (query)                            502.66       116.64       619.30       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32-np21 (query)                            502.66       144.64       647.29       0.4995          1.0556            1.0517         9.21
IVF-PQ-nl223-m32 (self)                                  502.66       441.60       944.25       0.4161          1.0769            1.0757         9.21
IVF-PQ-nl223-m64-np11 (query)                            590.73       163.44       754.17       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64-np14 (query)                            590.73       185.29       776.02       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64-np21 (query)                            590.73       246.34       837.07       0.7007          1.0188            1.0153        17.71
IVF-PQ-nl223-m64 (self)                                  590.73       762.88     1_353.61       0.6428          1.0259            1.0230        17.71
IVF-PQ-nl316-m16-np15 (query)                            491.95        78.72       570.67       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16-np17 (query)                            491.95        81.27       573.22       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16-np25 (query)                            491.95        96.49       588.44       0.4014          1.0814            1.0842         6.51
IVF-PQ-nl316-m16 (self)                                  491.95       267.37       759.32       0.3174          1.1174            1.1224         6.51
IVF-PQ-nl316-m32-np15 (query)                            539.23       120.76       659.98       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32-np17 (query)                            539.23       127.11       666.33       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32-np25 (query)                            539.23       157.12       696.35       0.5121          1.0503            1.0485        12.21
IVF-PQ-nl316-m32 (self)                                  539.23       490.26     1_029.49       0.4252          1.0725            1.0727        12.21
IVF-PQ-nl316-m64-np15 (query)                            624.55       187.21       811.77       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64-np17 (query)                            624.55       200.15       824.71       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64-np25 (query)                            624.55       274.91       899.47       0.7103          1.0164            1.0144        23.61
IVF-PQ-nl316-m64 (self)                                  624.55       844.22     1_468.78       0.6508          1.0241            1.0218        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.15     1_304.07     1_372.23       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.15     4_423.74     4_491.90       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                740.16       551.57     1_291.73       0.2443          1.1297            1.1195         1.76
Exhaustive-PQ-m16 (self)                                 740.16     1_729.27     2_469.43       0.2277          1.1396            1.1265         1.76
Exhaustive-PQ-m32 (query)                                655.84       921.10     1_576.94       0.2649          1.1130            1.1155         2.53
Exhaustive-PQ-m32 (self)                                 655.84     3_019.15     3_674.99       0.2433          1.1221            1.1232         2.53
Exhaustive-PQ-m64 (query)                                759.74     2_064.16     2_823.91       0.2958          1.0991            1.1029         4.05
Exhaustive-PQ-m64 (self)                                 759.74     6_529.95     7_289.69       0.2627          1.1103            1.1143         4.05
IVF-PQ-nl158-m16-np7 (query)                             951.40        92.27     1_043.67       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16-np12 (query)                            951.40        99.18     1_050.57       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16-np17 (query)                            951.40       107.95     1_059.35       0.3106          1.0852            1.0893         4.54
IVF-PQ-nl158-m16 (self)                                  951.40       284.60     1_235.99       0.2643          1.1069            1.1139         4.54
IVF-PQ-nl158-m32-np7 (query)                           1_175.64       129.88     1_305.52       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32-np12 (query)                          1_175.64       149.65     1_325.29       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32-np17 (query)                          1_175.64       159.05     1_334.69       0.3586          1.0691            1.0701         7.77
IVF-PQ-nl158-m32 (self)                                1_175.64       445.58     1_621.21       0.2908          1.0912            1.0946         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_145.04       189.58     1_334.62       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_145.04       234.60     1_379.64       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_145.04       253.91     1_398.95       0.4663          1.0454            1.0409        14.24
IVF-PQ-nl158-m64 (self)                                1_145.04       724.01     1_869.05       0.3928          1.0582            1.0562        14.24
IVF-PQ-nl223-m16-np11 (query)                            884.46        99.75       984.21       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16-np14 (query)                            884.46       102.26       986.72       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16-np21 (query)                            884.46       118.75     1_003.21       0.3240          1.0784            1.0829         5.68
IVF-PQ-nl223-m16 (self)                                  884.46       313.14     1_197.60       0.2682          1.1028            1.1098         5.68
IVF-PQ-nl223-m32-np11 (query)                            871.81       133.70     1_005.51       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32-np14 (query)                            871.81       139.52     1_011.33       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32-np21 (query)                            871.81       192.50     1_064.31       0.3733          1.0632            1.0645         9.93
IVF-PQ-nl223-m32 (self)                                  871.81       484.38     1_356.19       0.2958          1.0879            1.0916         9.93
IVF-PQ-nl223-m64-np11 (query)                          1_079.69       191.48     1_271.17       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64-np14 (query)                          1_079.69       213.76     1_293.44       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64-np21 (query)                          1_079.69       269.27     1_348.95       0.4834          1.0412            1.0372        18.43
IVF-PQ-nl223-m64 (self)                                1_079.69       804.16     1_883.84       0.4000          1.0559            1.0544        18.43
IVF-PQ-nl316-m16-np15 (query)                            864.78       108.98       973.76       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16-np17 (query)                            864.78       108.30       973.07       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16-np25 (query)                            864.78       124.07       988.85       0.3302          1.0747            1.0799         7.32
IVF-PQ-nl316-m16 (self)                                  864.78       330.59     1_195.36       0.2716          1.0994            1.1064         7.32
IVF-PQ-nl316-m32-np15 (query)                            932.70       144.68     1_077.39       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32-np17 (query)                            932.70       150.65     1_083.36       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32-np25 (query)                            932.70       171.89     1_104.59       0.3833          1.0594            1.0614        13.02
IVF-PQ-nl316-m32 (self)                                  932.70       492.15     1_424.86       0.3010          1.0845            1.0888        13.02
IVF-PQ-nl316-m64-np15 (query)                            998.80       217.91     1_216.71       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64-np17 (query)                            998.80       231.02     1_229.82       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64-np25 (query)                            998.80       291.35     1_290.16       0.4950          1.0369            1.0355        24.42
IVF-PQ-nl316-m64 (self)                                  998.80       883.59     1_882.39       0.4089          1.0523            1.0526        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.91     1_884.01     1_983.92       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.91     6_329.10     6_429.01       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                899.74       557.08     1_456.82       0.2345          1.1095            1.1000         2.26
Exhaustive-PQ-m16 (self)                                 899.74     1_777.18     2_676.92       0.2206          1.1180            1.1048         2.26
Exhaustive-PQ-m32 (query)                                931.27       939.93     1_871.20       0.2567          1.0943            1.0974         3.03
Exhaustive-PQ-m32 (self)                                 931.27     3_047.76     3_979.02       0.2391          1.1012            1.1021         3.03
Exhaustive-PQ-m64 (query)                              1_013.73     1_922.55     2_936.29       0.2775          1.0855            1.0910         4.55
Exhaustive-PQ-m64 (self)                               1_013.73     6_378.89     7_392.62       0.2515          1.0934            1.0980         4.55
Exhaustive-PQ-m128 (query)                             1_169.65     3_677.68     4_847.33       0.3162          1.0712            1.0752         7.61
Exhaustive-PQ-m128 (self)                              1_169.65    12_195.75    13_365.40       0.2755          1.0817            1.0859         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_323.39       112.95     1_436.34       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_323.39       129.10     1_452.50       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_323.39       128.67     1_452.06       0.2891          1.0761            1.0819         5.20
IVF-PQ-nl158-m16 (self)                                1_323.39       327.77     1_651.16       0.2524          1.0921            1.0990         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_352.03       134.62     1_486.65       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_352.03       154.38     1_506.40       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_352.03       177.93     1_529.96       0.3189          1.0656            1.0692         8.43
IVF-PQ-nl158-m32 (self)                                1_352.03       470.24     1_822.27       0.2635          1.0846            1.0898         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_428.94       185.06     1_614.01       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_428.94       221.57     1_650.51       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_428.94       265.73     1_694.67       0.3786          1.0518            1.0497        14.89
IVF-PQ-nl158-m64 (self)                                1_428.94       764.05     2_192.99       0.3087          1.0667            1.0672        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_613.13       286.53     1_899.66       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_613.13       363.66     1_976.79       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_613.13       446.86     2_060.00       0.5374          1.0268            1.0226        27.82
IVF-PQ-nl158-m128 (self)                               1_613.13     1_369.40     2_982.53       0.4648          1.0341            1.0317        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_156.62       119.63     1_276.25       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_156.62       124.57     1_281.19       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_156.62       139.64     1_296.26       0.2981          1.0713            1.0771         6.40
IVF-PQ-nl223-m16 (self)                                1_156.62       359.25     1_515.87       0.2560          1.0886            1.0956         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_183.01       156.30     1_339.31       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_183.01       158.86     1_341.87       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_183.01       194.25     1_377.25       0.3312          1.0606            1.0639        10.65
IVF-PQ-nl223-m32 (self)                                1_183.01       509.41     1_692.42       0.2676          1.0816            1.0868        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_258.71       209.49     1_468.20       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_258.71       235.40     1_494.11       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_258.71       289.51     1_548.22       0.3963          1.0464            1.0458        19.15
IVF-PQ-nl223-m64 (self)                                1_258.71       845.02     2_103.73       0.3163          1.0635            1.0652        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_445.12       340.10     1_785.22       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_445.12       379.50     1_824.62       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_445.12       501.61     1_946.73       0.5499          1.0236            1.0212        36.14
IVF-PQ-nl223-m128 (self)                               1_445.12     1_546.53     2_991.65       0.4736          1.0318            1.0306        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_234.16       132.01     1_366.18       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_234.16       136.30     1_370.47       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_234.16       149.89     1_384.05       0.3053          1.0677            1.0734         8.13
IVF-PQ-nl316-m16 (self)                                1_234.16       395.64     1_629.80       0.2586          1.0861            1.0927         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_263.65       164.03     1_427.68       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_263.65       174.77     1_438.42       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_263.65       195.04     1_458.69       0.3393          1.0573            1.0609        13.83
IVF-PQ-nl316-m32 (self)                                1_263.65       563.66     1_827.31       0.2711          1.0790            1.0845        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_346.38       240.67     1_587.05       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_346.38       261.29     1_607.68       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_346.38       316.18     1_662.56       0.4084          1.0427            1.0431        25.23
IVF-PQ-nl316-m64 (self)                                1_346.38       944.39     2_290.77       0.3237          1.0605            1.0632        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_544.01       404.01     1_948.01       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_544.01       431.14     1_975.14       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_544.01       588.93     2_132.93       0.5612          1.0212            1.0201        48.03
IVF-PQ-nl316-m128 (self)                               1_544.01     1_846.31     3_390.32       0.4840          1.0301            1.0294        48.03
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Data where the structure resides on a lower-dimensional manifold.

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.07       740.11       773.18       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.07     2_449.12     2_482.19       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                398.43       530.26       928.69       0.2932          1.2577            1.2510         1.26
Exhaustive-PQ-m16 (self)                                 398.43     1_724.43     2_122.85       0.2301          1.3863            1.3798         1.26
Exhaustive-PQ-m32 (query)                                422.91       914.29     1_337.20       0.4008          1.1658            1.1600         2.03
Exhaustive-PQ-m32 (self)                                 422.91     3_029.63     3_452.53       0.3180          1.2686            1.2616         2.03
Exhaustive-PQ-m64 (query)                                508.33     1_946.52     2_454.86       0.5384          1.0881            1.0842         3.55
Exhaustive-PQ-m64 (self)                                 508.33     6_466.75     6_975.08       0.4587          1.1480            1.1426         3.55
IVF-PQ-nl158-m16-np7 (query)                             606.73        71.15       677.89       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16-np12 (query)                            606.73        75.41       682.14       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16-np17 (query)                            606.73       114.18       720.91       0.5357          1.0880            1.0854         3.89
IVF-PQ-nl158-m16 (self)                                  606.73       250.63       857.36       0.4271          1.1656            1.1619         3.89
IVF-PQ-nl158-m32-np7 (query)                             682.27       101.78       784.05       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32-np12 (query)                            682.27       144.22       826.49       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32-np17 (query)                            682.27       185.26       867.53       0.6749          1.0396            1.0375         7.12
IVF-PQ-nl158-m32 (self)                                  682.27       484.82     1_167.10       0.6045          1.0697            1.0649         7.12
IVF-PQ-nl158-m64-np7 (query)                             720.43       170.64       891.07       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64-np12 (query)                            720.43       184.35       904.78       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64-np17 (query)                            720.43       248.63       969.06       0.8335          1.0094            1.0082        13.58
IVF-PQ-nl158-m64 (self)                                  720.43       784.90     1_505.33       0.7984          1.0163            1.0142        13.58
IVF-PQ-nl223-m16-np11 (query)                          1_489.65        83.18     1_572.83       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16-np14 (query)                          1_489.65        77.86     1_567.52       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16-np21 (query)                          1_489.65       110.50     1_600.15       0.5365          1.0877            1.0844         4.97
IVF-PQ-nl223-m16 (self)                                1_489.65       279.34     1_768.99       0.4228          1.1688            1.1643         4.97
IVF-PQ-nl223-m32-np11 (query)                            953.44       117.52     1_070.96       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32-np14 (query)                            953.44       129.38     1_082.82       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32-np21 (query)                            953.44       159.01     1_112.46       0.6763          1.0394            1.0371         9.21
IVF-PQ-nl223-m32 (self)                                  953.44       457.10     1_410.54       0.6025          1.0704            1.0658         9.21
IVF-PQ-nl223-m64-np11 (query)                            845.43       188.96     1_034.39       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64-np14 (query)                            845.43       179.45     1_024.88       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64-np21 (query)                            845.43       252.61     1_098.04       0.8366          1.0091            1.0079        17.71
IVF-PQ-nl223-m64 (self)                                  845.43       795.61     1_641.04       0.8009          1.0158            1.0139        17.71
IVF-PQ-nl316-m16-np15 (query)                            630.38        83.57       713.95       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16-np17 (query)                            630.38        87.67       718.06       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16-np25 (query)                            630.38       103.98       734.37       0.5357          1.0879            1.0852         6.51
IVF-PQ-nl316-m16 (self)                                  630.38       268.82       899.20       0.4167          1.1731            1.1687         6.51
IVF-PQ-nl316-m32-np15 (query)                            963.07       119.29     1_082.36       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32-np17 (query)                            963.07       121.38     1_084.44       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32-np25 (query)                            963.07       153.77     1_116.84       0.6783          1.0386            1.0366        12.21
IVF-PQ-nl316-m32 (self)                                  963.07       465.57     1_428.64       0.6014          1.0706            1.0662        12.21
IVF-PQ-nl316-m64-np15 (query)                          1_250.77       182.04     1_432.81       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64-np17 (query)                          1_250.77       201.93     1_452.70       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64-np25 (query)                          1_250.77       263.75     1_514.52       0.8387          1.0089            1.0077        23.61
IVF-PQ-nl316-m64 (self)                                1_250.77       838.24     2_089.01       0.8029          1.0154            1.0136        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.30     1_298.47     1_366.77       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.30     4_379.22     4_447.53       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                614.28       536.17     1_150.45       0.2128          1.2291            1.2259         1.76
Exhaustive-PQ-m16 (self)                                 614.28     1_732.30     2_346.58       0.1772          1.3099            1.3107         1.76
Exhaustive-PQ-m32 (query)                                647.58       919.40     1_566.97       0.2801          1.1736            1.1700         2.53
Exhaustive-PQ-m32 (self)                                 647.58     3_012.48     3_660.05       0.2228          1.2514            1.2496         2.53
Exhaustive-PQ-m64 (query)                                742.40     1_922.77     2_665.17       0.3752          1.1186            1.1154         4.05
Exhaustive-PQ-m64 (self)                                 742.40     6_473.95     7_216.35       0.2986          1.1838            1.1817         4.05
IVF-PQ-nl158-m16-np7 (query)                             943.31        95.40     1_038.71       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16-np12 (query)                            943.31        99.21     1_042.51       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16-np17 (query)                            943.31       109.85     1_053.16       0.3823          1.1170            1.1170         4.54
IVF-PQ-nl158-m16 (self)                                  943.31       286.84     1_230.15       0.2728          1.2080            1.2113         4.54
IVF-PQ-nl158-m32-np7 (query)                             977.46       125.37     1_102.82       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32-np12 (query)                            977.46       134.94     1_112.40       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32-np17 (query)                            977.46       161.92     1_139.38       0.4902          1.0727            1.0708         7.77
IVF-PQ-nl158-m32 (self)                                  977.46       443.55     1_421.00       0.3902          1.1266            1.1249         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_058.88       159.42     1_218.30       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_058.88       200.93     1_259.81       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_058.88       247.95     1_306.83       0.6310          1.0349            1.0334        14.24
IVF-PQ-nl158-m64 (self)                                1_058.88       735.42     1_794.30       0.5735          1.0546            1.0504        14.24
IVF-PQ-nl223-m16-np11 (query)                            808.57        98.55       907.12       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16-np14 (query)                            808.57       102.66       911.23       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16-np21 (query)                            808.57       116.90       925.47       0.3805          1.1171            1.1170         5.68
IVF-PQ-nl223-m16 (self)                                  808.57       304.83     1_113.40       0.2700          1.2105            1.2146         5.68
IVF-PQ-nl223-m32-np11 (query)                            854.94       132.74       987.68       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32-np14 (query)                            854.94       139.30       994.24       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32-np21 (query)                            854.94       167.95     1_022.88       0.4908          1.0720            1.0705         9.93
IVF-PQ-nl223-m32 (self)                                  854.94       492.47     1_347.41       0.3857          1.1288            1.1269         9.93
IVF-PQ-nl223-m64-np11 (query)                            942.39       187.28     1_129.67       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64-np14 (query)                            942.39       213.37     1_155.76       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64-np21 (query)                            942.39       269.63     1_212.02       0.6333          1.0343            1.0327        18.43
IVF-PQ-nl223-m64 (self)                                  942.39       793.25     1_735.63       0.5729          1.0547            1.0507        18.43
IVF-PQ-nl316-m16-np15 (query)                            860.61       110.00       970.60       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16-np17 (query)                            860.61       111.04       971.65       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16-np25 (query)                            860.61       123.31       983.92       0.3767          1.1183            1.1183         7.32
IVF-PQ-nl316-m16 (self)                                  860.61       337.21     1_197.82       0.2616          1.2171            1.2216         7.32
IVF-PQ-nl316-m32-np15 (query)                            904.68       147.95     1_052.63       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32-np17 (query)                            904.68       151.53     1_056.21       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32-np25 (query)                            904.68       177.90     1_082.58       0.4874          1.0732            1.0720        13.02
IVF-PQ-nl316-m32 (self)                                  904.68       511.78     1_416.47       0.3754          1.1341            1.1326        13.02
IVF-PQ-nl316-m64-np15 (query)                          1_016.25       216.78     1_233.03       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64-np17 (query)                          1_016.25       230.62     1_246.86       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64-np25 (query)                          1_016.25       294.70     1_310.95       0.6347          1.0339            1.0324        24.42
IVF-PQ-nl316-m64 (self)                                1_016.25       923.33     1_939.58       0.5696          1.0555            1.0517        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.91     1_870.41     1_970.32       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.91     6_205.19     6_305.10       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                906.39       549.02     1_455.41       0.2070          1.2190            1.2147         2.26
Exhaustive-PQ-m16 (self)                                 906.39     1_774.38     2_680.77       0.1758          1.3086            1.3089         2.26
Exhaustive-PQ-m32 (query)                                949.02       932.51     1_881.53       0.2712          1.1686            1.1636         3.03
Exhaustive-PQ-m32 (self)                                 949.02     3_038.63     3_987.65       0.2191          1.2527            1.2506         3.03
Exhaustive-PQ-m64 (query)                              1_018.82     1_958.99     2_977.81       0.3546          1.1211            1.1168         4.55
Exhaustive-PQ-m64 (self)                               1_018.82     6_453.87     7_472.69       0.2870          1.1905            1.1878         4.55
Exhaustive-PQ-m128 (query)                             1_204.71     3_754.63     4_959.35       0.4597          1.0781            1.0752         7.61
Exhaustive-PQ-m128 (self)                              1_204.71    13_261.70    14_466.42       0.3908          1.1257            1.1234         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_309.10       114.21     1_423.31       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_309.10       118.43     1_427.53       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_309.10       133.13     1_442.23       0.3621          1.1174            1.1171         5.20
IVF-PQ-nl158-m16 (self)                                1_309.10       336.06     1_645.15       0.2567          1.2219            1.2266         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_355.88       132.72     1_488.60       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_355.88       152.46     1_508.35       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_355.88       176.76     1_532.65       0.4637          1.0762            1.0749         8.43
IVF-PQ-nl158-m32 (self)                                1_355.88       487.69     1_843.57       0.3618          1.1419            1.1414         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_422.04       180.54     1_602.58       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_422.04       223.63     1_645.67       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_422.04       264.83     1_686.87       0.5770          1.0439            1.0425        14.89
IVF-PQ-nl158-m64 (self)                                1_422.04       806.44     2_228.48       0.5142          1.0726            1.0686        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_638.83       282.59     1_921.42       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_638.83       364.72     2_003.55       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_638.83       461.24     2_100.07       0.7397          1.0157            1.0140        27.82
IVF-PQ-nl158-m128 (self)                               1_638.83     1_422.21     3_061.03       0.7142          1.0238            1.0195        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_187.76       120.30     1_308.06       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_187.76       127.77     1_315.53       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_187.76       142.11     1_329.87       0.3623          1.1170            1.1169         6.40
IVF-PQ-nl223-m16 (self)                                1_187.76       365.60     1_553.36       0.2535          1.2252            1.2304         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_202.11       154.31     1_356.42       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_202.11       167.21     1_369.32       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_202.11       190.87     1_392.98       0.4608          1.0765            1.0751        10.65
IVF-PQ-nl223-m32 (self)                                1_202.11       525.67     1_727.78       0.3520          1.1464            1.1458        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_405.80       226.35     1_632.15       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_405.80       244.78     1_650.59       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_405.80       310.13     1_715.94       0.5740          1.0445            1.0427        19.15
IVF-PQ-nl223-m64 (self)                                1_405.80       893.93     2_299.74       0.5029          1.0763            1.0720        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_564.58       351.30     1_915.88       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_564.58       414.29     1_978.87       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_564.58       551.93     2_116.50       0.7396          1.0156            1.0140        36.14
IVF-PQ-nl223-m128 (self)                               1_564.58     1_695.29     3_259.87       0.7120          1.0240            1.0200        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_513.47       151.28     1_664.75       0.3583          1.1192            1.1195         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_513.47       149.81     1_663.28       0.3583          1.1192            1.1195         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_513.47       167.67     1_681.14       0.3583          1.1192            1.1195         8.13
IVF-PQ-nl316-m16 (self)                                1_513.47       442.64     1_956.11       0.2455          1.2315            1.2371         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_520.30       179.80     1_700.10       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_520.30       171.35     1_691.65       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_520.30       219.91     1_740.21       0.4537          1.0788            1.0776        13.83
IVF-PQ-nl316-m32 (self)                                1_520.30       575.37     2_095.68       0.3334          1.1569            1.1576        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_380.65       252.04     1_632.69       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_380.65       262.15     1_642.80       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_380.65       333.79     1_714.44       0.5722          1.0450            1.0435        25.23
IVF-PQ-nl316-m64 (self)                                1_380.65       964.34     2_344.99       0.4877          1.0813            1.0776        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_597.73       405.49     2_003.22       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_597.73       434.95     2_032.68       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_597.73       600.20     2_197.94       0.7406          1.0155            1.0139        48.03
IVF-PQ-nl316-m128 (self)                               1_597.73     1_924.32     3_522.06       0.7100          1.0240            1.0206        48.03
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.16       736.24       769.41       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.16     2_362.09     2_395.26       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                356.93       513.16       870.10       0.7118          1.1576            1.1395         1.26
Exhaustive-PQ-m16 (self)                                 356.93     1_683.53     2_040.47       0.6210          1.2885            1.2506         1.26
Exhaustive-PQ-m32 (query)                                377.02       894.48     1_271.50       0.7717          1.0965            1.0836         2.03
Exhaustive-PQ-m32 (self)                                 377.02     2_960.84     3_337.85       0.6993          1.1778            1.1516         2.03
Exhaustive-PQ-m64 (query)                                513.88     1_909.12     2_423.00       0.8251          1.0574            1.0468         3.55
Exhaustive-PQ-m64 (self)                                 513.88     6_373.81     6_887.70       0.7675          1.1055            1.0855         3.55
IVF-PQ-nl158-m16-np7 (query)                             524.76        74.19       598.95       0.8305          1.0508            1.0429         3.89
IVF-PQ-nl158-m16-np12 (query)                            524.76       124.87       649.63       0.8312          1.0503            1.0425         3.89
IVF-PQ-nl158-m16-np17 (query)                            524.76       146.26       671.02       0.8312          1.0503            1.0425         3.89
IVF-PQ-nl158-m16 (self)                                  524.76       378.60       903.36       0.7709          1.0956            1.0803         3.89
IVF-PQ-nl158-m32-np7 (query)                             653.40       115.16       768.55       0.8763          1.0258            1.0214         7.12
IVF-PQ-nl158-m32-np12 (query)                            653.40       204.61       858.01       0.8772          1.0252            1.0211         7.12
IVF-PQ-nl158-m32-np17 (query)                            653.40       244.12       897.52       0.8773          1.0252            1.0211         7.12
IVF-PQ-nl158-m32 (self)                                  653.40       730.88     1_384.28       0.8315          1.0494            1.0404         7.12
IVF-PQ-nl158-m64-np7 (query)                             707.21       167.35       874.56       0.9051          1.0152            1.0115        13.58
IVF-PQ-nl158-m64-np12 (query)                            707.21       261.30       968.51       0.9061          1.0146            1.0113        13.58
IVF-PQ-nl158-m64-np17 (query)                            707.21       361.35     1_068.56       0.9062          1.0146            1.0113        13.58
IVF-PQ-nl158-m64 (self)                                  707.21     1_163.11     1_870.32       0.8723          1.0278            1.0215        13.58
IVF-PQ-nl223-m16-np11 (query)                            855.76        81.87       937.63       0.8437          1.0424            1.0363         4.97
IVF-PQ-nl223-m16-np14 (query)                            855.76       103.28       959.03       0.8438          1.0424            1.0363         4.97
IVF-PQ-nl223-m16-np21 (query)                            855.76       154.38     1_010.13       0.8438          1.0424            1.0363         4.97
IVF-PQ-nl223-m16 (self)                                  855.76       363.32     1_219.08       0.7859          1.0824            1.0687         4.97
IVF-PQ-nl223-m32-np11 (query)                            919.47       142.35     1_061.82       0.8841          1.0224            1.0184         9.21
IVF-PQ-nl223-m32-np14 (query)                            919.47       158.61     1_078.08       0.8842          1.0224            1.0183         9.21
IVF-PQ-nl223-m32-np21 (query)                            919.47       222.56     1_142.04       0.8842          1.0224            1.0183         9.21
IVF-PQ-nl223-m32 (self)                                  919.47       634.65     1_554.12       0.8399          1.0442            1.0358         9.21
IVF-PQ-nl223-m64-np11 (query)                            615.96       193.84       809.79       0.9100          1.0133            1.0101        17.71
IVF-PQ-nl223-m64-np14 (query)                            615.96       233.93       849.88       0.9102          1.0132            1.0101        17.71
IVF-PQ-nl223-m64-np21 (query)                            615.96       344.00       959.95       0.9102          1.0132            1.0101        17.71
IVF-PQ-nl223-m64 (self)                                  615.96     1_065.08     1_681.04       0.8764          1.0258            1.0201        17.71
IVF-PQ-nl316-m16-np15 (query)                            517.85        82.81       600.65       0.8515          1.0384            1.0325         6.51
IVF-PQ-nl316-m16-np17 (query)                            517.85        88.01       605.86       0.8516          1.0384            1.0325         6.51
IVF-PQ-nl316-m16-np25 (query)                            517.85       113.26       631.11       0.8516          1.0384            1.0324         6.51
IVF-PQ-nl316-m16 (self)                                  517.85       341.71       859.56       0.7932          1.0774            1.0628         6.51
IVF-PQ-nl316-m32-np15 (query)                            513.70       132.88       646.58       0.8886          1.0207            1.0168        12.21
IVF-PQ-nl316-m32-np17 (query)                            513.70       140.64       654.34       0.8886          1.0207            1.0168        12.21
IVF-PQ-nl316-m32-np25 (query)                            513.70       190.97       704.67       0.8887          1.0207            1.0168        12.21
IVF-PQ-nl316-m32 (self)                                  513.70       603.65     1_117.35       0.8445          1.0420            1.0330        12.21
IVF-PQ-nl316-m64-np15 (query)                            620.14       210.65       830.79       0.9133          1.0124            1.0096        23.61
IVF-PQ-nl316-m64-np17 (query)                            620.14       228.81       848.95       0.9133          1.0124            1.0096        23.61
IVF-PQ-nl316-m64-np25 (query)                            620.14       325.92       946.06       0.9134          1.0124            1.0096        23.61
IVF-PQ-nl316-m64 (self)                                  620.14     1_031.91     1_652.05       0.8800          1.0242            1.0186        23.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.97     1_337.11     1_405.07       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.97     4_398.31     4_466.28       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                690.20       569.53     1_259.73       0.6791          1.1977            1.1746         1.76
Exhaustive-PQ-m16 (self)                                 690.20     1_823.95     2_514.15       0.5853          1.3494            1.3061         1.76
Exhaustive-PQ-m32 (query)                                657.38       922.48     1_579.87       0.7374          1.1283            1.1129         2.53
Exhaustive-PQ-m32 (self)                                 657.38     3_027.41     3_684.79       0.6552          1.2348            1.2026         2.53
Exhaustive-PQ-m64 (query)                                743.83     1_947.11     2_690.95       0.7805          1.0879            1.0755         4.05
Exhaustive-PQ-m64 (self)                                 743.83     6_518.00     7_261.83       0.7136          1.1583            1.1336         4.05
IVF-PQ-nl158-m16-np7 (query)                             934.31       106.83     1_041.14       0.8415          1.0469            1.0377         4.54
IVF-PQ-nl158-m16-np12 (query)                            934.31       125.14     1_059.45       0.8418          1.0467            1.0375         4.54
IVF-PQ-nl158-m16-np17 (query)                            934.31       153.01     1_087.32       0.8418          1.0467            1.0375         4.54
IVF-PQ-nl158-m16 (self)                                  934.31       428.98     1_363.29       0.7803          1.0947            1.0676         4.54
IVF-PQ-nl158-m32-np7 (query)                             991.11       134.26     1_125.37       0.8688          1.0313            1.0244         7.77
IVF-PQ-nl158-m32-np12 (query)                            991.11       185.43     1_176.54       0.8692          1.0311            1.0243         7.77
IVF-PQ-nl158-m32-np17 (query)                            991.11       233.49     1_224.60       0.8692          1.0311            1.0243         7.77
IVF-PQ-nl158-m32 (self)                                  991.11       683.53     1_674.64       0.8174          1.0636            1.0455         7.77
IVF-PQ-nl158-m64-np7 (query)                           1_076.50       189.25     1_265.76       0.8922          1.0204            1.0155        14.24
IVF-PQ-nl158-m64-np12 (query)                          1_076.50       283.80     1_360.30       0.8927          1.0202            1.0154        14.24
IVF-PQ-nl158-m64-np17 (query)                          1_076.50       365.05     1_441.55       0.8927          1.0202            1.0154        14.24
IVF-PQ-nl158-m64 (self)                                1_076.50     1_174.78     2_251.29       0.8484          1.0422            1.0299        14.24
IVF-PQ-nl223-m16-np11 (query)                            873.25       107.40       980.65       0.8603          1.0367            1.0290         5.68
IVF-PQ-nl223-m16-np14 (query)                            873.25       119.30       992.55       0.8603          1.0367            1.0289         5.68
IVF-PQ-nl223-m16-np21 (query)                            873.25       150.68     1_023.93       0.8603          1.0367            1.0289         5.68
IVF-PQ-nl223-m16 (self)                                  873.25       409.37     1_282.62       0.8010          1.0770            1.0539         5.68
IVF-PQ-nl223-m32-np11 (query)                            854.26       149.49     1_003.74       0.8835          1.0251            1.0189         9.93
IVF-PQ-nl223-m32-np14 (query)                            854.26       174.11     1_028.37       0.8836          1.0251            1.0188         9.93
IVF-PQ-nl223-m32-np21 (query)                            854.26       231.33     1_085.58       0.8836          1.0251            1.0188         9.93
IVF-PQ-nl223-m32 (self)                                  854.26       667.32     1_521.57       0.8325          1.0539            1.0361         9.93
IVF-PQ-nl223-m64-np11 (query)                            960.67       240.84     1_201.51       0.9025          1.0170            1.0127        18.43
IVF-PQ-nl223-m64-np14 (query)                            960.67       259.48     1_220.15       0.9026          1.0170            1.0126        18.43
IVF-PQ-nl223-m64-np21 (query)                            960.67       375.62     1_336.29       0.9026          1.0170            1.0126        18.43
IVF-PQ-nl223-m64 (self)                                  960.67     1_113.74     2_074.41       0.8592          1.0367            1.0244        18.43
IVF-PQ-nl316-m16-np15 (query)                            890.51       115.60     1_006.11       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16-np17 (query)                            890.51       122.63     1_013.14       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16-np25 (query)                            890.51       147.15     1_037.66       0.8711          1.0314            1.0245         7.32
IVF-PQ-nl316-m16 (self)                                  890.51       415.02     1_305.54       0.8168          1.0640            1.0447         7.32
IVF-PQ-nl316-m32-np15 (query)                            931.62       164.20     1_095.82       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32-np17 (query)                            931.62       172.11     1_103.72       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32-np25 (query)                            931.62       227.80     1_159.42       0.8924          1.0213            1.0158        13.02
IVF-PQ-nl316-m32 (self)                                  931.62       660.76     1_592.38       0.8456          1.0448            1.0302        13.02
IVF-PQ-nl316-m64-np15 (query)                          1_021.58       246.92     1_268.50       0.9077          1.0150            1.0109        24.42
IVF-PQ-nl316-m64-np17 (query)                          1_021.58       271.31     1_292.89       0.9077          1.0150            1.0109        24.42
IVF-PQ-nl316-m64-np25 (query)                          1_021.58       365.96     1_387.54       0.9077          1.0150            1.0108        24.42
IVF-PQ-nl316-m64 (self)                                1_021.58     1_144.46     2_166.04       0.8670          1.0322            1.0213        24.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.57     1_910.69     2_013.26       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.57     6_318.33     6_420.90       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                900.10       555.86     1_455.96       0.6502          1.2419            1.2113         2.26
Exhaustive-PQ-m16 (self)                                 900.10     1_782.92     2_683.01       0.5522          1.4109            1.3575         2.26
Exhaustive-PQ-m32 (query)                                937.03       937.12     1_874.15       0.7657          1.0989            1.0852         3.03
Exhaustive-PQ-m32 (self)                                 937.03     3_054.07     3_991.10       0.6925          1.1782            1.1510         3.03
Exhaustive-PQ-m64 (query)                              1_016.80     1_956.66     2_973.46       0.8202          1.0558            1.0466         4.55
Exhaustive-PQ-m64 (self)                               1_016.80     6_410.73     7_427.53       0.7633          1.1010            1.0854         4.55
Exhaustive-PQ-m128 (query)                             1_180.63     3_663.27     4_843.89       0.8668          1.0289            1.0236         7.61
Exhaustive-PQ-m128 (self)                              1_180.63    12_093.40    13_274.02       0.8261          1.0515            1.0424         7.61
IVF-PQ-nl158-m16-np7 (query)                           1_307.47       117.24     1_424.71       0.8560          1.0399            1.0308         5.20
IVF-PQ-nl158-m16-np12 (query)                          1_307.47       143.53     1_451.00       0.8562          1.0399            1.0307         5.20
IVF-PQ-nl158-m16-np17 (query)                          1_307.47       165.89     1_473.36       0.8561          1.0399            1.0307         5.20
IVF-PQ-nl158-m16 (self)                                1_307.47       452.97     1_760.44       0.7979          1.0768            1.0566         5.20
IVF-PQ-nl158-m32-np7 (query)                           1_345.56       152.86     1_498.42       0.9037          1.0190            1.0117         8.43
IVF-PQ-nl158-m32-np12 (query)                          1_345.56       198.22     1_543.78       0.9039          1.0190            1.0117         8.43
IVF-PQ-nl158-m32-np17 (query)                          1_345.56       248.75     1_594.31       0.9039          1.0190            1.0117         8.43
IVF-PQ-nl158-m32 (self)                                1_345.56       703.39     2_048.95       0.8613          1.0392            1.0217         8.43
IVF-PQ-nl158-m64-np7 (query)                           1_403.18       210.53     1_613.71       0.9233          1.0120            1.0064        14.89
IVF-PQ-nl158-m64-np12 (query)                          1_403.18       295.21     1_698.39       0.9235          1.0119            1.0064        14.89
IVF-PQ-nl158-m64-np17 (query)                          1_403.18       387.63     1_790.81       0.9235          1.0119            1.0064        14.89
IVF-PQ-nl158-m64 (self)                                1_403.18     1_166.46     2_569.64       0.8877          1.0258            1.0122        14.89
IVF-PQ-nl158-m128-np7 (query)                          1_562.07       335.62     1_897.69       0.9405          1.0068            1.0029        27.82
IVF-PQ-nl158-m128-np12 (query)                         1_562.07       530.78     2_092.85       0.9407          1.0067            1.0029        27.82
IVF-PQ-nl158-m128-np17 (query)                         1_562.07       694.96     2_257.02       0.9407          1.0067            1.0029        27.82
IVF-PQ-nl158-m128 (self)                               1_562.07     2_112.66     3_674.72       0.9097          1.0162            1.0065        27.82
IVF-PQ-nl223-m16-np11 (query)                          1_132.88       125.91     1_258.79       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16-np14 (query)                          1_132.88       144.58     1_277.47       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16-np21 (query)                          1_132.88       164.50     1_297.38       0.8629          1.0361            1.0286         6.40
IVF-PQ-nl223-m16 (self)                                1_132.88       450.42     1_583.31       0.8071          1.0694            1.0515         6.40
IVF-PQ-nl223-m32-np11 (query)                          1_174.25       168.00     1_342.24       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32-np14 (query)                          1_174.25       192.88     1_367.12       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32-np21 (query)                          1_174.25       237.46     1_411.71       0.9087          1.0175            1.0106        10.65
IVF-PQ-nl223-m32 (self)                                1_174.25       689.55     1_863.80       0.8679          1.0350            1.0196        10.65
IVF-PQ-nl223-m64-np11 (query)                          1_239.40       245.23     1_484.62       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64-np14 (query)                          1_239.40       287.37     1_526.76       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64-np21 (query)                          1_239.40       394.41     1_633.80       0.9273          1.0109            1.0055        19.15
IVF-PQ-nl223-m64 (self)                                1_239.40     1_159.23     2_398.62       0.8928          1.0233            1.0110        19.15
IVF-PQ-nl223-m128-np11 (query)                         1_569.64       416.64     1_986.28       0.9442          1.0059            1.0024        36.14
IVF-PQ-nl223-m128-np14 (query)                         1_569.64       507.96     2_077.61       0.9443          1.0059            1.0024        36.14
IVF-PQ-nl223-m128-np21 (query)                         1_569.64       745.31     2_314.96       0.9443          1.0059            1.0024        36.14
IVF-PQ-nl223-m128 (self)                               1_569.64     2_144.87     3_714.52       0.9138          1.0145            1.0057        36.14
IVF-PQ-nl316-m16-np15 (query)                          1_207.43       141.12     1_348.55       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16-np17 (query)                          1_207.43       144.94     1_352.37       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16-np25 (query)                          1_207.43       167.03     1_374.46       0.8688          1.0324            1.0254         8.13
IVF-PQ-nl316-m16 (self)                                1_207.43       479.63     1_687.06       0.8156          1.0629            1.0467         8.13
IVF-PQ-nl316-m32-np15 (query)                          1_263.35       183.32     1_446.67       0.9116          1.0160            1.0097        13.83
IVF-PQ-nl316-m32-np17 (query)                          1_263.35       196.83     1_460.18       0.9116          1.0160            1.0096        13.83
IVF-PQ-nl316-m32-np25 (query)                          1_263.35       238.74     1_502.09       0.9116          1.0160            1.0096        13.83
IVF-PQ-nl316-m32 (self)                                1_263.35       694.33     1_957.68       0.8732          1.0323            1.0180        13.83
IVF-PQ-nl316-m64-np15 (query)                          1_354.13       266.59     1_620.72       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64-np17 (query)                          1_354.13       292.45     1_646.58       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64-np25 (query)                          1_354.13       393.65     1_747.78       0.9296          1.0100            1.0049        25.23
IVF-PQ-nl316-m64 (self)                                1_354.13     1_184.90     2_539.03       0.8978          1.0211            1.0099        25.23
IVF-PQ-nl316-m128-np15 (query)                         1_525.91       446.37     1_972.29       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128-np17 (query)                         1_525.91       501.80     2_027.72       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128-np25 (query)                         1_525.91       733.79     2_259.70       0.9458          1.0054            1.0021        48.03
IVF-PQ-nl316-m128 (self)                               1_525.91     2_268.36     3_794.28       0.9179          1.0132            1.0050        48.03
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.51       687.59       720.10       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.51     2_221.08     2_253.59       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_402.56       517.62     1_920.19       0.2858          1.1535            1.1327         1.51
Exhaustive-OPQ-m16 (self)                              1_402.56     1_648.99     3_051.55       0.2593          1.1721            1.1492         1.51
Exhaustive-OPQ-m32 (query)                             1_574.90       882.88     2_457.78       0.3254          1.1214            1.1169         2.28
Exhaustive-OPQ-m32 (self)                              1_574.90     2_894.84     4_469.74       0.2827          1.1442            1.1387         2.28
Exhaustive-OPQ-m64 (query)                             1_936.53     1_872.29     3_808.82       0.3791          1.0985            1.0952         3.80
Exhaustive-OPQ-m64 (self)                              1_936.53     6_693.88     8_630.41       0.3227          1.1203            1.1168         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_516.49        79.47     1_595.96       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_516.49        94.78     1_611.27       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_516.49       104.64     1_621.12       0.3900          1.0871            1.0887         4.39
IVF-OPQ-nl158-m16 (self)                               1_516.49       282.95     1_799.43       0.3177          1.1185            1.1226         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_699.55       107.17     1_806.72       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_699.55       123.30     1_822.84       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_699.55       136.87     1_836.42       0.4949          1.0553            1.0538         7.62
IVF-OPQ-nl158-m32 (self)                               1_699.55       394.51     2_094.06       0.4166          1.0763            1.0763         7.62
IVF-OPQ-nl158-m64-np7 (query)                          2_023.20       144.89     2_168.09       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64-np12 (query)                         2_023.20       182.91     2_206.11       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64-np17 (query)                         2_023.20       230.36     2_253.56       0.6959          1.0189            1.0159        14.08
IVF-OPQ-nl158-m64 (self)                               2_023.20       682.43     2_705.63       0.6384          1.0264            1.0236        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_537.91        90.67     1_628.58       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_537.91        97.57     1_635.48       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_537.91       111.92     1_649.83       0.3980          1.0832            1.0845         5.47
IVF-OPQ-nl223-m16 (self)                               1_537.91       291.66     1_829.56       0.3219          1.1154            1.1195         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_707.65       114.89     1_822.54       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_707.65       124.85     1_832.50       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_707.65       154.21     1_861.86       0.5079          1.0516            1.0497         9.71
IVF-OPQ-nl223-m32 (self)                               1_707.65       434.24     2_141.89       0.4258          1.0726            1.0728         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_062.79       170.48     2_233.27       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_062.79       197.88     2_260.68       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_062.79       252.39     2_315.19       0.7048          1.0174            1.0150        18.21
IVF-OPQ-nl223-m64 (self)                               2_062.79       757.17     2_819.96       0.6468          1.0247            1.0224        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_563.82       100.92     1_664.74       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_563.82        99.99     1_663.81       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_563.82       121.48     1_685.30       0.4097          1.0777            1.0806         7.26
IVF-OPQ-nl316-m16 (self)                               1_563.82       308.30     1_872.12       0.3273          1.1114            1.1168         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_783.40       126.85     1_910.25       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_783.40       158.33     1_941.73       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_783.40       156.83     1_940.24       0.5185          1.0477            1.0472        12.96
IVF-OPQ-nl316-m32 (self)                               1_783.40       589.88     2_373.29       0.4330          1.0696            1.0707        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_207.36       200.05     2_407.41       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_207.36       209.64     2_417.00       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_207.36       281.88     2_489.23       0.7140          1.0156            1.0141        24.36
IVF-OPQ-nl316-m64 (self)                               2_207.36       893.29     3_100.65       0.6548          1.0232            1.0214        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.95     1_230.73     1_298.68       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.95     4_095.75     4_163.70       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_800.87       561.92     3_362.79       0.2661          1.1132            1.1009         2.76
Exhaustive-OPQ-m16 (self)                              2_800.87     1_678.04     4_478.91       0.2457          1.1248            1.1094         2.76
Exhaustive-OPQ-m32 (query)                             2_958.09       935.53     3_893.62       0.2884          1.0947            1.0956         3.53
Exhaustive-OPQ-m32 (self)                              2_958.09     2_935.06     5_893.15       0.2623          1.1062            1.1056         3.53
Exhaustive-OPQ-m64 (query)                             3_361.46     1_940.89     5_302.35       0.3208          1.0819            1.0844         5.05
Exhaustive-OPQ-m64 (self)                              3_361.46     6_267.11     9_628.57       0.2785          1.0968            1.0989         5.05
Exhaustive-OPQ-m128 (query)                            4_030.26     3_615.32     7_645.59       0.3695          1.0682            1.0688         8.11
Exhaustive-OPQ-m128 (self)                             4_030.26    11_831.17    15_861.43       0.3153          1.0826            1.0834         8.11
IVF-OPQ-nl158-m16-np7 (query)                          2_922.28       133.86     3_056.14       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16-np12 (query)                         2_922.28       143.67     3_065.94       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16-np17 (query)                         2_922.28       156.11     3_078.38       0.3263          1.0775            1.0815         6.04
IVF-OPQ-nl158-m16 (self)                               2_922.28       309.43     3_231.71       0.2765          1.0975            1.1034         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_103.91       159.68     3_263.58       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_103.91       177.56     3_281.47       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_103.91       196.85     3_300.76       0.3695          1.0643            1.0658         9.27
IVF-OPQ-nl158-m32 (self)                               3_103.91       438.46     3_542.37       0.3017          1.0856            1.0895         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_392.36       196.32     3_588.67       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_392.36       242.66     3_635.02       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_392.36       280.49     3_672.84       0.4768          1.0414            1.0396        15.74
IVF-OPQ-nl158-m64 (self)                               3_392.36       710.25     4_102.61       0.4016          1.0548            1.0547        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_077.89       293.76     4_371.65       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_077.89       375.38     4_453.27       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_077.89       459.20     4_537.09       0.6830          1.0143            1.0118        28.67
IVF-OPQ-nl158-m128 (self)                              4_077.89     1_294.09     5_371.98       0.6263          1.0191            1.0170        28.67
IVF-OPQ-nl223-m16-np11 (query)                         2_917.08       141.75     3_058.82       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16-np14 (query)                         2_917.08       151.79     3_068.87       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16-np21 (query)                         2_917.08       168.21     3_085.29       0.3319          1.0744            1.0789         7.18
IVF-OPQ-nl223-m16 (self)                               2_917.08       336.45     3_253.53       0.2771          1.0966            1.1032         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_098.15       168.14     3_266.30       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_098.15       179.72     3_277.87       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_098.15       205.85     3_304.01       0.3815          1.0601            1.0619        11.43
IVF-OPQ-nl223-m32 (self)                               3_098.15       478.01     3_576.16       0.3046          1.0836            1.0876        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_459.10       233.47     3_692.57       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_459.10       260.44     3_719.54       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_459.10       299.64     3_758.74       0.4927          1.0377            1.0365        19.93
IVF-OPQ-nl223-m64 (self)                               3_459.10       780.20     4_239.31       0.4096          1.0526            1.0530        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_177.49       350.94     4_528.43       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_177.49       397.81     4_575.30       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_177.49       529.78     4_707.27       0.6925          1.0129            1.0110        36.92
IVF-OPQ-nl223-m128 (self)                              4_177.49     1_503.55     5_681.04       0.6353          1.0180            1.0161        36.92
IVF-OPQ-nl316-m16-np15 (query)                         3_009.49       157.31     3_166.80       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16-np17 (query)                         3_009.49       156.83     3_166.32       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16-np25 (query)                         3_009.49       172.71     3_182.20       0.3389          1.0714            1.0763         9.32
IVF-OPQ-nl316-m16 (self)                               3_009.49       365.06     3_374.55       0.2802          1.0943            1.1011         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_358.10       198.94     3_557.04       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_358.10       198.57     3_556.66       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_358.10       219.14     3_577.23       0.3910          1.0568            1.0594        15.02
IVF-OPQ-nl316-m32 (self)                               3_358.10       539.25     3_897.34       0.3099          1.0809            1.0857        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_854.60       259.70     4_114.30       0.5031          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_854.60       268.94     4_123.54       0.5031          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_854.60       326.35     4_180.95       0.5031          1.0346            1.0346        26.42
IVF-OPQ-nl316-m64 (self)                               3_854.60       899.76     4_754.36       0.4170          1.0500            1.0512        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_444.60       420.36     4_864.96       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_444.60       512.24     4_956.84       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_444.60       674.29     5_118.89       0.7007          1.0115            1.0104        49.23
IVF-OPQ-nl316-m128 (self)                              4_444.60     1_813.78     6_258.38       0.6420          1.0168            1.0155        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.16     1_872.65     1_972.81       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.16     6_338.86     6_439.02       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_696.95       658.20     5_355.15       0.2572          1.0953            1.0838         4.51
Exhaustive-OPQ-m16 (self)                              4_696.95     1_781.11     6_478.05       0.2411          1.1050            1.0895         4.51
Exhaustive-OPQ-m32 (query)                             4_817.71     1_024.60     5_842.30       0.2801          1.0776            1.0800         5.28
Exhaustive-OPQ-m32 (self)                              4_817.71     3_023.48     7_841.19       0.2567          1.0868            1.0870         5.28
Exhaustive-OPQ-m64 (query)                             5_049.20     2_019.97     7_069.18       0.2974          1.0722            1.0753         6.80
Exhaustive-OPQ-m64 (self)                              5_049.20     6_410.24    11_459.45       0.2647          1.0826            1.0853         6.80
Exhaustive-OPQ-m128 (query)                            5_643.42     3_738.29     9_381.71       0.3296          1.0636            1.0665         9.86
Exhaustive-OPQ-m128 (self)                             5_643.42    12_151.27    17_794.69       0.2830          1.0757            1.0788         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_585.88       206.64     4_792.52       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_585.88       210.93     4_796.81       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_585.88       228.91     4_814.79       0.3045          1.0684            1.0736         8.20
IVF-OPQ-nl158-m16 (self)                               4_585.88       366.16     4_952.04       0.2654          1.0828            1.0887         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_694.06       222.16     4_916.22       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_694.06       242.26     4_936.32       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_694.06       255.56     4_949.62       0.3322          1.0600            1.0631        11.43
IVF-OPQ-nl158-m32 (self)                               4_694.06       470.78     5_164.84       0.2747          1.0782            1.0828        11.43
IVF-OPQ-nl158-m64-np7 (query)                          4_984.98       267.59     5_252.57       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64-np12 (query)                         4_984.98       307.00     5_291.97       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64-np17 (query)                         4_984.98       340.59     5_325.56       0.3914          1.0472            1.0472        17.89
IVF-OPQ-nl158-m64 (self)                               4_984.98       765.90     5_750.88       0.3189          1.0629            1.0647        17.89
IVF-OPQ-nl158-m128-np7 (query)                         5_667.45       361.02     6_028.47       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128-np12 (query)                        5_667.45       438.06     6_105.51       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128-np17 (query)                        5_667.45       527.81     6_195.26       0.5442          1.0238            1.0221        30.82
IVF-OPQ-nl158-m128 (self)                              5_667.45     1_355.66     7_023.11       0.4727          1.0316            1.0308        30.82
IVF-OPQ-nl223-m16-np11 (query)                         4_700.38       211.64     4_912.02       0.3096          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16-np14 (query)                         4_700.38       220.49     4_920.86       0.3096          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16-np21 (query)                         4_700.38       234.31     4_934.69       0.3096          1.0660            1.0711         9.40
IVF-OPQ-nl223-m16 (self)                               4_700.38       406.72     5_107.10       0.2676          1.0815            1.0878         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_870.04       238.45     5_108.49       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_870.04       249.82     5_119.87       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_870.04       270.87     5_140.91       0.3403          1.0574            1.0606        13.65
IVF-OPQ-nl223-m32 (self)                               4_870.04       537.41     5_407.45       0.2769          1.0770            1.0821        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_254.04       302.80     5_556.84       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_254.04       314.92     5_568.96       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_254.04       374.66     5_628.71       0.4056          1.0433            1.0443        22.15
IVF-OPQ-nl223-m64 (self)                               5_254.04       880.78     6_134.82       0.3253          1.0604            1.0630        22.15
IVF-OPQ-nl223-m128-np11 (query)                        6_028.93       428.51     6_457.44       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128-np14 (query)                        6_028.93       472.83     6_501.76       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128-np21 (query)                        6_028.93       591.81     6_620.74       0.5564          1.0216            1.0207        39.14
IVF-OPQ-nl223-m128 (self)                              6_028.93     1_594.01     7_622.94       0.4809          1.0301            1.0299        39.14
IVF-OPQ-nl316-m16-np15 (query)                         4_972.42       234.65     5_207.07       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16-np17 (query)                         4_972.42       232.25     5_204.67       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16-np25 (query)                         4_972.42       246.42     5_218.84       0.3145          1.0637            1.0691        11.88
IVF-OPQ-nl316-m16 (self)                               4_972.42       445.57     5_417.99       0.2683          1.0802            1.0863        11.88
IVF-OPQ-nl316-m32-np15 (query)                         5_185.68       265.43     5_451.11       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32-np17 (query)                         5_185.68       269.00     5_454.68       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32-np25 (query)                         5_185.68       294.79     5_480.47       0.3455          1.0554            1.0591        17.58
IVF-OPQ-nl316-m32 (self)                               5_185.68       587.39     5_773.07       0.2785          1.0757            1.0812        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_559.47       356.96     5_916.43       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_559.47       357.76     5_917.23       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_559.47       410.72     5_970.19       0.4164          1.0405            1.0421        28.98
IVF-OPQ-nl316-m64 (self)                               5_559.47       989.18     6_548.65       0.3306          1.0583            1.0617        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_313.52       507.30     6_820.82       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_313.52       532.33     6_845.86       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_313.52       695.74     7_009.27       0.5665          1.0199            1.0196        51.78
IVF-OPQ-nl316-m128 (self)                              6_313.52     1_926.27     8_239.79       0.4890          1.0288            1.0288        51.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.95       710.65       743.60       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.95     2_390.93     2_423.88       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_438.57       521.02     1_959.58       0.3003          1.2501            1.2435         1.51
Exhaustive-OPQ-m16 (self)                              1_438.57     1_676.26     3_114.83       0.2369          1.3780            1.3719         1.51
Exhaustive-OPQ-m32 (query)                             1_602.23       899.16     2_501.39       0.4212          1.1523            1.1471         2.28
Exhaustive-OPQ-m32 (self)                              1_602.23     2_945.86     4_548.09       0.3382          1.2478            1.2414         2.28
Exhaustive-OPQ-m64 (query)                             1_969.27     1_888.67     3_857.94       0.5666          1.0762            1.0730         3.80
Exhaustive-OPQ-m64 (self)                              1_969.27     6_206.47     8_175.73       0.4882          1.1282            1.1236         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_509.72        68.74     1_578.46       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_509.72        81.75     1_591.46       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_509.72        94.64     1_604.36       0.7011          1.0321            1.0305         4.39
IVF-OPQ-nl158-m16 (self)                               1_509.72       244.27     1_753.99       0.6228          1.0616            1.0590         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_689.93        99.54     1_789.46       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_689.93       129.07     1_819.00       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_689.93       149.64     1_839.56       0.8019          1.0135            1.0123         7.62
IVF-OPQ-nl158-m32 (self)                               1_689.93       423.64     2_113.57       0.7523          1.0251            1.0231         7.62
IVF-OPQ-nl158-m64-np7 (query)                          2_009.07       140.09     2_149.16       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64-np12 (query)                         2_009.07       184.54     2_193.62       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64-np17 (query)                         2_009.07       237.02     2_246.09       0.8620          1.0063            1.0053        14.08
IVF-OPQ-nl158-m64 (self)                               2_009.07       713.59     2_722.67       0.8324          1.0110            1.0095        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_524.05        83.32     1_607.37       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_524.05        86.39     1_610.45       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_524.05       106.51     1_630.56       0.7077          1.0308            1.0292         5.47
IVF-OPQ-nl223-m16 (self)                               1_524.05       278.94     1_802.99       0.6294          1.0591            1.0566         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_696.57       109.06     1_805.63       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_696.57       125.94     1_822.51       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_696.57       160.40     1_856.98       0.8047          1.0131            1.0119         9.71
IVF-OPQ-nl223-m32 (self)                               1_696.57       456.52     2_153.09       0.7555          1.0241            1.0224         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_046.60       168.04     2_214.64       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_046.60       186.97     2_233.57       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_046.60       266.71     2_313.31       0.8648          1.0060            1.0051        18.21
IVF-OPQ-nl223-m64 (self)                               2_046.60       764.05     2_810.65       0.8351          1.0105            1.0093        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_516.38        84.00     1_600.39       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_516.38        87.78     1_604.16       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_516.38        98.48     1_614.87       0.7116          1.0299            1.0284         7.26
IVF-OPQ-nl316-m16 (self)                               1_516.38       260.38     1_776.76       0.6361          1.0569            1.0546         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_728.23       121.72     1_849.94       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_728.23       126.67     1_854.89       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_728.23       165.08     1_893.31       0.8070          1.0127            1.0115        12.96
IVF-OPQ-nl316-m32 (self)                               1_728.23       461.59     2_189.82       0.7593          1.0233            1.0217        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_076.16       190.42     2_266.57       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_076.16       205.05     2_281.21       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_076.16       285.62     2_361.77       0.8655          1.0060            1.0050        24.36
IVF-OPQ-nl316-m64 (self)                               2_076.16       820.45     2_896.61       0.8374          1.0102            1.0089        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.18     1_298.24     1_366.42       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.18     4_372.15     4_440.33       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_981.05       587.34     3_568.38       0.2324          1.2133            1.2098         2.76
Exhaustive-OPQ-m16 (self)                              2_981.05     1_831.92     4_812.97       0.1872          1.2980            1.2978         2.76
Exhaustive-OPQ-m32 (query)                             3_192.62       972.06     4_164.68       0.3204          1.1503            1.1475         3.53
Exhaustive-OPQ-m32 (self)                              3_192.62     3_050.55     6_243.17       0.2587          1.2172            1.2150         3.53
Exhaustive-OPQ-m64 (query)                             3_658.02     1_979.96     5_637.98       0.4335          1.0938            1.0916         5.05
Exhaustive-OPQ-m64 (self)                              3_658.02     6_408.22    10_066.24       0.3626          1.1418            1.1394         5.05
Exhaustive-OPQ-m128 (query)                            4_182.49     3_893.87     8_076.36       0.5701          1.0489            1.0475         8.11
Exhaustive-OPQ-m128 (self)                             4_182.49    12_184.63    16_367.12       0.5007          1.0772            1.0753         8.11
IVF-OPQ-nl158-m16-np7 (query)                          3_093.62       135.91     3_229.53       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16-np12 (query)                         3_093.62       141.57     3_235.20       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16-np17 (query)                         3_093.62       155.48     3_249.11       0.5408          1.0561            1.0547         6.04
IVF-OPQ-nl158-m16 (self)                               3_093.62       321.05     3_414.68       0.4384          1.1011            1.0990         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_463.16       152.48     3_615.64       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_463.16       175.96     3_639.11       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_463.16       203.29     3_666.45       0.6850          1.0244            1.0233         9.27
IVF-OPQ-nl158-m32 (self)                               3_463.16       448.53     3_911.69       0.6101          1.0436            1.0415         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_691.53       200.21     3_891.73       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_691.53       238.21     3_929.73       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_691.53       281.68     3_973.21       0.7785          1.0115            1.0104        15.74
IVF-OPQ-nl158-m64 (self)                               3_691.53       712.28     4_403.80       0.7314          1.0197            1.0178        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_210.41       289.65     4_500.06       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_210.41       369.74     4_580.15       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_210.41       466.37     4_676.78       0.8391          1.0060            1.0050        28.67
IVF-OPQ-nl158-m128 (self)                              4_210.41     1_316.94     5_527.35       0.8113          1.0097            1.0079        28.67
IVF-OPQ-nl223-m16-np11 (query)                         3_012.64       141.83     3_154.48       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16-np14 (query)                         3_012.64       146.65     3_159.30       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16-np21 (query)                         3_012.64       167.95     3_180.59       0.5464          1.0543            1.0531         7.18
IVF-OPQ-nl223-m16 (self)                               3_012.64       343.97     3_356.62       0.4469          1.0974            1.0955         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_160.98       168.81     3_329.79       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_160.98       175.72     3_336.70       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_160.98       208.14     3_369.12       0.6906          1.0234            1.0224        11.43
IVF-OPQ-nl223-m32 (self)                               3_160.98       464.77     3_625.75       0.6182          1.0416            1.0399        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_589.62       224.00     3_813.62       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_589.62       248.31     3_837.93       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_589.62       294.69     3_884.31       0.7831          1.0111            1.0101        19.93
IVF-OPQ-nl223-m64 (self)                               3_589.62       787.00     4_376.62       0.7362          1.0189            1.0173        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_276.49       342.14     4_618.63       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_276.49       388.29     4_664.79       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_276.49       527.43     4_803.92       0.8403          1.0059            1.0050        36.92
IVF-OPQ-nl223-m128 (self)                              4_276.49     1_536.66     5_813.16       0.8131          1.0093            1.0080        36.92
IVF-OPQ-nl316-m16-np15 (query)                         3_081.53       156.31     3_237.84       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16-np17 (query)                         3_081.53       156.91     3_238.44       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16-np25 (query)                         3_081.53       176.10     3_257.63       0.5522          1.0526            1.0515         9.32
IVF-OPQ-nl316-m16 (self)                               3_081.53       375.37     3_456.90       0.4548          1.0943            1.0926         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_288.04       186.52     3_474.56       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_288.04       187.10     3_475.13       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_288.04       215.57     3_503.61       0.6949          1.0228            1.0215        15.02
IVF-OPQ-nl316-m32 (self)                               3_288.04       507.75     3_795.79       0.6243          1.0401            1.0384        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_712.13       260.44     3_972.57       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_712.13       277.37     3_989.50       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_712.13       333.54     4_045.67       0.7844          1.0109            1.0100        26.42
IVF-OPQ-nl316-m64 (self)                               3_712.13       903.95     4_616.08       0.7395          1.0183            1.0168        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_465.49       418.09     4_883.58       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_465.49       445.34     4_910.83       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_465.49       616.50     5_081.99       0.8407          1.0059            1.0050        49.23
IVF-OPQ-nl316-m128 (self)                              4_465.49     1_794.02     6_259.51       0.8145          1.0092            1.0078        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.74     1_903.23     2_002.97       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.74     6_287.54     6_387.28       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_658.44       656.19     5_314.63       0.2278          1.2029            1.1998         4.51
Exhaustive-OPQ-m16 (self)                              4_658.44     1_780.24     6_438.67       0.1860          1.2978            1.2978         4.51
Exhaustive-OPQ-m32 (query)                             4_792.20     1_036.18     5_828.37       0.3126          1.1446            1.1410         5.28
Exhaustive-OPQ-m32 (self)                              4_792.20     3_082.69     7_874.89       0.2577          1.2169            1.2141         5.28
Exhaustive-OPQ-m64 (query)                             5_119.01     2_023.84     7_142.85       0.4083          1.0976            1.0948         6.80
Exhaustive-OPQ-m64 (self)                              5_119.01     6_351.33    11_470.33       0.3475          1.1498            1.1473         6.80
Exhaustive-OPQ-m128 (query)                            5_651.06     3_726.76     9_377.82       0.5288          1.0566            1.0549         9.86
Exhaustive-OPQ-m128 (self)                             5_651.06    12_728.20    18_379.26       0.4657          1.0908            1.0889         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_505.53       200.04     4_705.58       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_505.53       209.63     4_715.16       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_505.53       224.25     4_729.78       0.5354          1.0536            1.0525         8.20
IVF-OPQ-nl158-m16 (self)                               4_505.53       379.35     4_884.89       0.4336          1.1027            1.1010         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_569.04       215.61     4_784.65       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_569.04       232.56     4_801.60       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_569.04       258.97     4_828.01       0.6793          1.0236            1.0224        11.43
IVF-OPQ-nl158-m32 (self)                               4_569.04       481.09     5_050.13       0.6051          1.0446            1.0425        11.43
IVF-OPQ-nl158-m64-np7 (query)                          5_233.44       259.80     5_493.24       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64-np12 (query)                         5_233.44       302.04     5_535.48       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64-np17 (query)                         5_233.44       346.54     5_579.98       0.7765          1.0111            1.0100        17.89
IVF-OPQ-nl158-m64 (self)                               5_233.44       813.56     6_047.00       0.7271          1.0202            1.0182        17.89
IVF-OPQ-nl158-m128-np7 (query)                         5_535.79       351.58     5_887.38       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128-np12 (query)                        5_535.79       437.32     5_973.12       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128-np17 (query)                        5_535.79       529.30     6_065.09       0.8336          1.0061            1.0051        30.82
IVF-OPQ-nl158-m128 (self)                              5_535.79     1_370.98     6_906.77       0.8051          1.0103            1.0084        30.82
IVF-OPQ-nl223-m16-np11 (query)                         4_574.35       210.50     4_784.85       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16-np14 (query)                         4_574.35       214.34     4_788.69       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16-np21 (query)                         4_574.35       232.73     4_807.08       0.5404          1.0522            1.0508         9.40
IVF-OPQ-nl223-m16 (self)                               4_574.35       401.21     4_975.56       0.4416          1.0993            1.0976         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_723.60       231.39     4_954.98       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_723.60       249.08     4_972.68       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_723.60       269.54     4_993.14       0.6813          1.0233            1.0221        13.65
IVF-OPQ-nl223-m32 (self)                               4_723.60       511.06     5_234.66       0.6076          1.0439            1.0421        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_144.56       290.83     5_435.39       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_144.56       309.95     5_454.51       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_144.56       373.69     5_518.26       0.7776          1.0109            1.0099        22.15
IVF-OPQ-nl223-m64 (self)                               5_144.56       860.52     6_005.08       0.7310          1.0196            1.0179        22.15
IVF-OPQ-nl223-m128-np11 (query)                        5_871.49       423.25     6_294.74       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128-np14 (query)                        5_871.49       460.03     6_331.52       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128-np21 (query)                        5_871.49       599.95     6_471.44       0.8350          1.0059            1.0050        39.14
IVF-OPQ-nl223-m128 (self)                              5_871.49     1_609.52     7_481.01       0.8079          1.0099            1.0083        39.14
IVF-OPQ-nl316-m16-np15 (query)                         4_795.77       229.02     5_024.80       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16-np17 (query)                         4_795.77       227.46     5_023.23       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16-np25 (query)                         4_795.77       244.09     5_039.86       0.5469          1.0509            1.0498        11.88
IVF-OPQ-nl316-m16 (self)                               4_795.77       436.33     5_232.10       0.4474          1.0969            1.0954        11.88
IVF-OPQ-nl316-m32-np15 (query)                         5_001.25       260.66     5_261.91       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32-np17 (query)                         5_001.25       262.43     5_263.68       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32-np25 (query)                         5_001.25       292.62     5_293.87       0.6834          1.0230            1.0219        17.58
IVF-OPQ-nl316-m32 (self)                               5_001.25       580.59     5_581.84       0.6111          1.0430            1.0413        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_431.53       338.45     5_769.98       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_431.53       342.44     5_773.96       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_431.53       417.14     5_848.67       0.7798          1.0106            1.0097        28.98
IVF-OPQ-nl316-m64 (self)                               5_431.53       973.63     6_405.15       0.7344          1.0190            1.0175        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_103.19       505.67     6_608.86       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_103.19       529.17     6_632.36       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_103.19       683.61     6_786.80       0.8359          1.0058            1.0049        51.78
IVF-OPQ-nl316-m128 (self)                              6_103.19     1_911.40     8_014.59       0.8096          1.0096            1.0082        51.78
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.38       679.75       712.13       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.38     2_235.55     2_267.93       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_372.18       513.50     1_885.68       0.7915          1.0818            1.0683         1.51
Exhaustive-OPQ-m16 (self)                              1_372.18     1_643.05     3_015.23       0.7235          1.1500            1.1252         1.51
Exhaustive-OPQ-m32 (query)                             1_537.89       878.89     2_416.78       0.8299          1.0537            1.0423         2.28
Exhaustive-OPQ-m32 (self)                              1_537.89     2_869.95     4_407.84       0.7758          1.0978            1.0769         2.28
Exhaustive-OPQ-m64 (query)                             1_891.53     1_872.06     3_763.59       0.8560          1.0398            1.0294         3.80
Exhaustive-OPQ-m64 (self)                              1_891.53     6_203.52     8_095.06       0.8090          1.0724            1.0538         3.80
IVF-OPQ-nl158-m16-np7 (query)                          1_471.19        88.24     1_559.43       0.8911          1.0206            1.0158         4.39
IVF-OPQ-nl158-m16-np12 (query)                         1_471.19       125.45     1_596.64       0.8920          1.0201            1.0156         4.39
IVF-OPQ-nl158-m16-np17 (query)                         1_471.19       149.00     1_620.19       0.8920          1.0200            1.0156         4.39
IVF-OPQ-nl158-m16 (self)                               1_471.19       431.62     1_902.81       0.8499          1.0393            1.0311         4.39
IVF-OPQ-nl158-m32-np7 (query)                          1_657.66       116.97     1_774.63       0.9117          1.0132            1.0097         7.62
IVF-OPQ-nl158-m32-np12 (query)                         1_657.66       174.92     1_832.58       0.9128          1.0127            1.0094         7.62
IVF-OPQ-nl158-m32-np17 (query)                         1_657.66       221.59     1_879.25       0.9128          1.0126            1.0094         7.62
IVF-OPQ-nl158-m32 (self)                               1_657.66       658.87     2_316.53       0.8783          1.0253            1.0190         7.62
IVF-OPQ-nl158-m64-np7 (query)                          1_999.07       170.47     2_169.55       0.9248          1.0100            1.0064        14.08
IVF-OPQ-nl158-m64-np12 (query)                         1_999.07       266.78     2_265.86       0.9259          1.0095            1.0062        14.08
IVF-OPQ-nl158-m64-np17 (query)                         1_999.07       360.14     2_359.22       0.9259          1.0094            1.0062        14.08
IVF-OPQ-nl158-m64 (self)                               1_999.07     1_085.14     3_084.22       0.8977          1.0182            1.0127        14.08
IVF-OPQ-nl223-m16-np11 (query)                         1_477.25        96.01     1_573.26       0.8969          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16-np14 (query)                         1_477.25       113.83     1_591.09       0.8970          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16-np21 (query)                         1_477.25       143.53     1_620.78       0.8970          1.0180            1.0137         5.47
IVF-OPQ-nl223-m16 (self)                               1_477.25       392.18     1_869.43       0.8574          1.0356            1.0273         5.47
IVF-OPQ-nl223-m32-np11 (query)                         1_697.14       132.88     1_830.02       0.9154          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32-np14 (query)                         1_697.14       158.90     1_856.04       0.9155          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32-np21 (query)                         1_697.14       213.73     1_910.87       0.9156          1.0118            1.0085         9.71
IVF-OPQ-nl223-m32 (self)                               1_697.14       624.18     2_321.32       0.8825          1.0234            1.0176         9.71
IVF-OPQ-nl223-m64-np11 (query)                         2_052.22       196.75     2_248.97       0.9266          1.0091            1.0060        18.21
IVF-OPQ-nl223-m64-np14 (query)                         2_052.22       243.33     2_295.55       0.9268          1.0090            1.0060        18.21
IVF-OPQ-nl223-m64-np21 (query)                         2_052.22       343.03     2_395.25       0.9268          1.0090            1.0060        18.21
IVF-OPQ-nl223-m64 (self)                               2_052.22     1_029.93     3_082.15       0.8988          1.0176            1.0123        18.21
IVF-OPQ-nl316-m16-np15 (query)                         1_491.51       104.00     1_595.50       0.9020          1.0163            1.0125         7.26
IVF-OPQ-nl316-m16-np17 (query)                         1_491.51       109.52     1_601.02       0.9021          1.0163            1.0124         7.26
IVF-OPQ-nl316-m16-np25 (query)                         1_491.51       143.27     1_634.78       0.9021          1.0163            1.0124         7.26
IVF-OPQ-nl316-m16 (self)                               1_491.51       383.31     1_874.81       0.8638          1.0327            1.0243         7.26
IVF-OPQ-nl316-m32-np15 (query)                         1_732.72       138.19     1_870.91       0.9182          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32-np17 (query)                         1_732.72       168.29     1_901.02       0.9182          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32-np25 (query)                         1_732.72       227.27     1_959.99       0.9183          1.0111            1.0079        12.96
IVF-OPQ-nl316-m32 (self)                               1_732.72       703.00     2_435.72       0.8852          1.0224            1.0166        12.96
IVF-OPQ-nl316-m64-np15 (query)                         2_548.95       229.42     2_778.37       0.9288          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64-np17 (query)                         2_548.95       266.26     2_815.21       0.9289          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64-np25 (query)                         2_548.95       374.22     2_923.18       0.9289          1.0085            1.0055        24.36
IVF-OPQ-nl316-m64 (self)                               2_548.95     1_187.98     3_736.93       0.9011          1.0166            1.0116        24.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.81     1_315.13     1_383.94       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.81     4_370.96     4_439.77       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_982.61       586.94     3_569.55       0.7548          1.1139            1.0983         2.76
Exhaustive-OPQ-m16 (self)                              2_982.61     1_742.43     4_725.03       0.6783          1.2043            1.1743         2.76
Exhaustive-OPQ-m32 (query)                             3_130.36       970.85     4_101.22       0.8059          1.0693            1.0573         3.53
Exhaustive-OPQ-m32 (self)                              3_130.36     3_015.13     6_145.49       0.7451          1.1246            1.1019         3.53
Exhaustive-OPQ-m64 (query)                             3_522.45     1_983.13     5_505.59       0.8410          1.0458            1.0366         5.05
Exhaustive-OPQ-m64 (self)                              3_522.45     6_372.20     9_894.65       0.7916          1.0819            1.0657         5.05
Exhaustive-OPQ-m128 (query)                            4_128.72     3_700.12     7_828.84       0.9200          1.0107            1.0070         8.11
Exhaustive-OPQ-m128 (self)                             4_128.72    12_152.95    16_281.68       0.8931          1.0194            1.0139         8.11
IVF-OPQ-nl158-m16-np7 (query)                          2_973.53       142.86     3_116.38       0.8896          1.0221            1.0167         6.04
IVF-OPQ-nl158-m16-np12 (query)                         2_973.53       174.35     3_147.88       0.8900          1.0219            1.0166         6.04
IVF-OPQ-nl158-m16-np17 (query)                         2_973.53       203.22     3_176.75       0.8900          1.0219            1.0166         6.04
IVF-OPQ-nl158-m16 (self)                               2_973.53       508.99     3_482.52       0.8455          1.0452            1.0316         6.04
IVF-OPQ-nl158-m32-np7 (query)                          3_137.23       168.20     3_305.43       0.9010          1.0174            1.0128         9.27
IVF-OPQ-nl158-m32-np12 (query)                         3_137.23       222.85     3_360.08       0.9015          1.0172            1.0126         9.27
IVF-OPQ-nl158-m32-np17 (query)                         3_137.23       271.20     3_408.43       0.9015          1.0171            1.0126         9.27
IVF-OPQ-nl158-m32 (self)                               3_137.23       675.40     3_812.63       0.8625          1.0350            1.0243         9.27
IVF-OPQ-nl158-m64-np7 (query)                          3_491.29       230.68     3_721.97       0.9110          1.0140            1.0098        15.74
IVF-OPQ-nl158-m64-np12 (query)                         3_491.29       320.84     3_812.13       0.9115          1.0137            1.0097        15.74
IVF-OPQ-nl158-m64-np17 (query)                         3_491.29       408.82     3_900.10       0.9115          1.0137            1.0097        15.74
IVF-OPQ-nl158-m64 (self)                               3_491.29     1_140.21     4_631.50       0.8769          1.0274            1.0192        15.74
IVF-OPQ-nl158-m128-np7 (query)                         4_158.32       347.50     4_505.82       0.9609          1.0028            1.0001        28.67
IVF-OPQ-nl158-m128-np12 (query)                        4_158.32       540.54     4_698.87       0.9616          1.0026            1.0000        28.67
IVF-OPQ-nl158-m128-np17 (query)                        4_158.32       713.65     4_871.97       0.9616          1.0026            1.0000        28.67
IVF-OPQ-nl158-m128 (self)                              4_158.32     2_171.40     6_329.72       0.9449          1.0056            1.0015        28.67
IVF-OPQ-nl223-m16-np11 (query)                         3_002.41       155.18     3_157.59       0.9001          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16-np14 (query)                         3_002.41       173.59     3_176.00       0.9002          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16-np21 (query)                         3_002.41       205.97     3_208.38       0.9002          1.0182            1.0133         7.18
IVF-OPQ-nl223-m16 (self)                               3_002.41       464.13     3_466.54       0.8566          1.0385            1.0263         7.18
IVF-OPQ-nl223-m32-np11 (query)                         3_151.75       187.72     3_339.47       0.9080          1.0153            1.0109        11.43
IVF-OPQ-nl223-m32-np14 (query)                         3_151.75       209.81     3_361.56       0.9081          1.0152            1.0109        11.43
IVF-OPQ-nl223-m32-np21 (query)                         3_151.75       261.97     3_413.72       0.9081          1.0152            1.0109        11.43
IVF-OPQ-nl223-m32 (self)                               3_151.75       690.96     3_842.71       0.8697          1.0314            1.0212        11.43
IVF-OPQ-nl223-m64-np11 (query)                         3_612.91       263.93     3_876.84       0.9179          1.0117            1.0080        19.93
IVF-OPQ-nl223-m64-np14 (query)                         3_612.91       302.85     3_915.75       0.9180          1.0116            1.0080        19.93
IVF-OPQ-nl223-m64-np21 (query)                         3_612.91       409.05     4_021.96       0.9180          1.0116            1.0080        19.93
IVF-OPQ-nl223-m64 (self)                               3_612.91     1_120.41     4_733.32       0.8844          1.0239            1.0161        19.93
IVF-OPQ-nl223-m128-np11 (query)                        4_271.67       408.40     4_680.07       0.9664          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128-np14 (query)                        4_271.67       490.57     4_762.24       0.9666          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128-np21 (query)                        4_271.67       705.87     4_977.54       0.9666          1.0021            1.0000        36.92
IVF-OPQ-nl223-m128 (self)                              4_271.67     2_111.36     6_383.03       0.9487          1.0052            1.0008        36.92
IVF-OPQ-nl316-m16-np15 (query)                         3_120.77       177.24     3_298.00       0.9059          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16-np17 (query)                         3_120.77       178.79     3_299.56       0.9060          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16-np25 (query)                         3_120.77       211.24     3_332.01       0.9060          1.0164            1.0118         9.32
IVF-OPQ-nl316-m16 (self)                               3_120.77       463.49     3_584.25       0.8666          1.0330            1.0223         9.32
IVF-OPQ-nl316-m32-np15 (query)                         3_402.20       200.61     3_602.82       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32-np17 (query)                         3_402.20       213.05     3_615.25       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32-np25 (query)                         3_402.20       267.57     3_669.77       0.9150          1.0128            1.0089        15.02
IVF-OPQ-nl316-m32 (self)                               3_402.20       658.19     4_060.39       0.8798          1.0262            1.0176        15.02
IVF-OPQ-nl316-m64-np15 (query)                         3_739.98       291.07     4_031.04       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64-np17 (query)                         3_739.98       306.72     4_046.69       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64-np25 (query)                         3_739.98       400.88     4_140.85       0.9219          1.0105            1.0071        26.42
IVF-OPQ-nl316-m64 (self)                               3_739.98     1_134.07     4_874.05       0.8895          1.0216            1.0145        26.42
IVF-OPQ-nl316-m128-np15 (query)                        4_430.86       475.17     4_906.03       0.9684          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128-np17 (query)                        4_430.86       526.66     4_957.52       0.9685          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128-np25 (query)                        4_430.86       743.30     5_174.16       0.9685          1.0019            1.0000        49.23
IVF-OPQ-nl316-m128 (self)                              4_430.86     2_242.88     6_673.74       0.9511          1.0049            1.0005        49.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.13     1_876.03     1_976.15       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.13     6_218.41     6_318.53       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_538.46       646.94     5_185.40       0.7383          1.1310            1.1123         4.51
Exhaustive-OPQ-m16 (self)                              4_538.46     1_754.46     6_292.92       0.6589          1.2299            1.1973         4.51
Exhaustive-OPQ-m32 (query)                             4_654.26     1_020.24     5_674.49       0.8490          1.0412            1.0319         5.28
Exhaustive-OPQ-m32 (self)                              4_654.26     2_992.55     7_646.81       0.8007          1.0713            1.0579         5.28
Exhaustive-OPQ-m64 (query)                             4_931.04     1_992.79     6_923.83       0.8797          1.0255            1.0187         6.80
Exhaustive-OPQ-m64 (self)                              4_931.04     6_281.21    11_212.24       0.8412          1.0441            1.0343         6.80
Exhaustive-OPQ-m128 (query)                            5_706.94     3_662.96     9_369.90       0.9050          1.0147            1.0105         9.86
Exhaustive-OPQ-m128 (self)                             5_706.94    11_763.39    17_470.33       0.8745          1.0261            1.0199         9.86
IVF-OPQ-nl158-m16-np7 (query)                          4_567.28       208.21     4_775.49       0.8940          1.0213            1.0153         8.20
IVF-OPQ-nl158-m16-np12 (query)                         4_567.28       243.83     4_811.11       0.8941          1.0212            1.0153         8.20
IVF-OPQ-nl158-m16-np17 (query)                         4_567.28       272.61     4_839.89       0.8941          1.0212            1.0153         8.20
IVF-OPQ-nl158-m16 (self)                               4_567.28       509.86     5_077.14       0.8504          1.0416            1.0289         8.20
IVF-OPQ-nl158-m32-np7 (query)                          4_512.73       236.25     4_748.98       0.9385          1.0080            1.0032        11.43
IVF-OPQ-nl158-m32-np12 (query)                         4_512.73       290.23     4_802.96       0.9387          1.0079            1.0032        11.43
IVF-OPQ-nl158-m32-np17 (query)                         4_512.73       341.52     4_854.25       0.9387          1.0079            1.0032        11.43
IVF-OPQ-nl158-m32 (self)                               4_512.73       748.69     5_261.42       0.9109          1.0168            1.0067        11.43
IVF-OPQ-nl158-m64-np7 (query)                          4_848.03       294.65     5_142.68       0.9532          1.0047            1.0009        17.89
IVF-OPQ-nl158-m64-np12 (query)                         4_848.03       384.90     5_232.93       0.9534          1.0046            1.0009        17.89
IVF-OPQ-nl158-m64-np17 (query)                         4_848.03       481.64     5_329.67       0.9534          1.0046            1.0009        17.89
IVF-OPQ-nl158-m64 (self)                               4_848.03     1_185.94     6_033.97       0.9305          1.0107            1.0029        17.89
IVF-OPQ-nl158-m128-np7 (query)                         5_534.37       417.66     5_952.03       0.9617          1.0032            1.0000        30.82
IVF-OPQ-nl158-m128-np12 (query)                        5_534.37       583.44     6_117.81       0.9620          1.0031            1.0000        30.82
IVF-OPQ-nl158-m128-np17 (query)                        5_534.37       771.72     6_306.09       0.9620          1.0031            1.0000        30.82
IVF-OPQ-nl158-m128 (self)                              5_534.37     2_116.10     7_650.47       0.9405          1.0077            1.0015        30.82
IVF-OPQ-nl223-m16-np11 (query)                         4_533.15       219.90     4_753.05       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16-np14 (query)                         4_533.15       232.35     4_765.50       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16-np21 (query)                         4_533.15       276.38     4_809.53       0.8998          1.0192            1.0137         9.40
IVF-OPQ-nl223-m16 (self)                               4_533.15       500.77     5_033.92       0.8589          1.0368            1.0255         9.40
IVF-OPQ-nl223-m32-np11 (query)                         4_752.31       252.72     5_005.04       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32-np14 (query)                         4_752.31       280.29     5_032.60       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32-np21 (query)                         4_752.31       337.22     5_089.53       0.9411          1.0074            1.0028        13.65
IVF-OPQ-nl223-m32 (self)                               4_752.31       724.73     5_477.04       0.9147          1.0151            1.0061        13.65
IVF-OPQ-nl223-m64-np11 (query)                         5_148.63       328.25     5_476.88       0.9547          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64-np14 (query)                         5_148.63       365.56     5_514.19       0.9547          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64-np21 (query)                         5_148.63       464.52     5_613.14       0.9548          1.0045            1.0008        22.15
IVF-OPQ-nl223-m64 (self)                               5_148.63     1_142.68     6_291.31       0.9334          1.0097            1.0026        22.15
IVF-OPQ-nl223-m128-np11 (query)                        5_829.37       485.15     6_314.52       0.9642          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128-np14 (query)                        5_829.37       566.45     6_395.82       0.9642          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128-np21 (query)                        5_829.37       770.13     6_599.50       0.9642          1.0027            1.0000        39.14
IVF-OPQ-nl223-m128 (self)                              5_829.37     2_162.82     7_992.19       0.9436          1.0066            1.0011        39.14
IVF-OPQ-nl316-m16-np15 (query)                         4_814.79       238.71     5_053.50       0.9043          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16-np17 (query)                         4_814.79       240.23     5_055.02       0.9044          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16-np25 (query)                         4_814.79       269.16     5_083.96       0.9044          1.0171            1.0119        11.88
IVF-OPQ-nl316-m16 (self)                               4_814.79       517.45     5_332.25       0.8654          1.0330            1.0227        11.88
IVF-OPQ-nl316-m32-np15 (query)                         5_002.82       276.80     5_279.62       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32-np17 (query)                         5_002.82       287.83     5_290.64       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32-np25 (query)                         5_002.82       333.87     5_336.69       0.9435          1.0068            1.0024        17.58
IVF-OPQ-nl316-m32 (self)                               5_002.82       733.62     5_736.44       0.9189          1.0138            1.0053        17.58
IVF-OPQ-nl316-m64-np15 (query)                         5_456.66       360.32     5_816.98       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64-np17 (query)                         5_456.66       385.40     5_842.07       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64-np25 (query)                         5_456.66       483.18     5_939.84       0.9560          1.0042            1.0006        28.98
IVF-OPQ-nl316-m64 (self)                               5_456.66     1_200.88     6_657.55       0.9358          1.0090            1.0023        28.98
IVF-OPQ-nl316-m128-np15 (query)                        6_111.78       550.60     6_662.38       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128-np17 (query)                        6_111.78       596.03     6_707.81       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128-np25 (query)                        6_111.78       809.08     6_920.86       0.9650          1.0025            1.0000        51.78
IVF-OPQ-nl316-m128 (self)                              6_111.78     2_295.82     8_407.60       0.9466          1.0060            1.0008        51.78
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
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.76     1_249.14     1_316.90       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      767.93        81.55       849.47       0.3451          1.0762            1.0750         6.21
IVFPQ-m64-nl111-np1                                      855.46       103.52       958.97       0.4472          1.0523            1.0446        11.21
SOARPQ-shift0.5-m32-nl111-np1                            894.50        92.07       986.57       0.3227          1.2313            1.0783         8.69
IVFPQ-m32-nl111-np2                                      767.93        93.54       861.47       0.3476          1.0754            1.0744         6.21
IVFPQ-m64-nl111-np2                                      855.46       119.70       975.16       0.4530          1.0507            1.0437        11.21
SOARPQ-shift0.5-m32-nl111-np2                            894.50       107.39     1_001.89       0.3464          1.0767            1.0748         8.69
IVFPQ-m32-nl111-np4                                      767.93       106.98       874.90       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np4                                      855.46       142.93       998.39       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np4                            894.50       122.17     1_016.67       0.3477          1.0754            1.0743         8.69
IVFPQ-m32-nl111-np5                                      767.93       111.13       879.06       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np5                                      855.46       150.26     1_005.71       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np5                            894.50       128.40     1_022.90       0.3477          1.0754            1.0743         8.69
IVFPQ-m32-nl111-np8                                      767.93       126.67       894.60       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np8                                      855.46       177.81     1_033.27       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np8                            894.50       148.63     1_043.13       0.3477          1.0754            1.0743         8.69
IVFPQ-m32-nl111-np10                                     767.93       135.33       903.25       0.3477          1.0754            1.0743         6.21
IVFPQ-m64-nl111-np10                                     855.46       192.36     1_047.82       0.4535          1.0506            1.0436        11.21
SOARPQ-shift0.5-m32-nl111-np10                           894.50       167.94     1_062.44       0.3477          1.0754            1.0743         8.69
IVFPQ-m32-nl158-np1                                    1_037.52        81.36     1_118.88       0.3514          1.0716            1.0718         7.77
IVFPQ-m64-nl158-np1                                    1_131.37        97.87     1_229.25       0.4508          1.0490            1.0433        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_157.25       103.58     1_260.83       0.3173          1.1979            1.0779        10.25
IVFPQ-m32-nl158-np2                                    1_037.52       100.10     1_137.62       0.3581          1.0692            1.0703         7.77
IVFPQ-m64-nl158-np2                                    1_131.37       118.28     1_249.65       0.4651          1.0456            1.0411        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_157.25       123.01     1_280.26       0.3560          1.0732            1.0709        10.25
IVFPQ-m32-nl158-np4                                    1_037.52       107.55     1_145.07       0.3585          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np4                                    1_131.37       135.47     1_266.84       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_157.25       123.17     1_280.42       0.3582          1.0695            1.0702        10.25
IVFPQ-m32-nl158-np7                                    1_037.52       119.07     1_156.59       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np7                                    1_131.37       164.06     1_295.43       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_157.25       147.58     1_304.83       0.3586          1.0691            1.0701        10.25
IVFPQ-m32-nl158-np8                                    1_037.52       124.46     1_161.98       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np8                                    1_131.37       172.20     1_303.57       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_157.25       145.78     1_303.02       0.3586          1.0691            1.0701        10.25
IVFPQ-m32-nl158-np12                                   1_037.52       139.59     1_177.11       0.3586          1.0691            1.0701         7.77
IVFPQ-m64-nl158-np12                                   1_131.37       199.08     1_330.45       0.4663          1.0454            1.0409        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_157.25       170.01     1_327.26       0.3586          1.0691            1.0701        10.25
IVFPQ-m32-nl223-np1                                      858.61        73.02       931.63       0.3569          1.0685            1.0684         9.93
IVFPQ-m64-nl223-np1                                      941.19        85.48     1_026.68       0.4445          1.0490            1.0434        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_005.94        87.53     1_093.47       0.3207          1.1636            1.0760        12.41
IVFPQ-m32-nl223-np2                                      858.61        86.12       944.72       0.3711          1.0638            1.0651         9.93
IVFPQ-m64-nl223-np2                                      941.19       103.59     1_044.78       0.4763          1.0424            1.0385        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_005.94       106.81     1_112.76       0.3662          1.0680            1.0675        12.41
IVFPQ-m32-nl223-np4                                      858.61       104.64       963.25       0.3732          1.0633            1.0645         9.93
IVFPQ-m64-nl223-np4                                      941.19       132.15     1_073.34       0.4831          1.0413            1.0373        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_005.94       124.96     1_130.90       0.3720          1.0645            1.0650        12.41
IVFPQ-m32-nl223-np8                                      858.61       125.73       984.33       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np8                                      941.19       168.77     1_109.96       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_005.94       145.71     1_151.66       0.3733          1.0632            1.0645        12.41
IVFPQ-m32-nl223-np11                                     858.61       142.75     1_001.35       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np11                                     941.19       219.82     1_161.01       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_005.94       176.18     1_182.13       0.3733          1.0632            1.0645        12.41
IVFPQ-m32-nl223-np14                                     858.61       158.90     1_017.51       0.3733          1.0632            1.0645         9.93
IVFPQ-m64-nl223-np14                                     941.19       229.23     1_170.42       0.4834          1.0412            1.0372        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_005.94       187.89     1_193.83       0.3733          1.0632            1.0645        12.41
IVFPQ-m32-nl316-np1                                    1_011.93        79.54     1_091.46       0.3578          1.0669            1.0670        13.02
IVFPQ-m64-nl316-np1                                    1_085.54        84.20     1_169.74       0.4396          1.0476            1.0440        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_157.95        88.32     1_246.27       0.3204          1.1512            1.0722        15.50
IVFPQ-m32-nl316-np2                                    1_011.93        86.65     1_098.58       0.3781          1.0605            1.0626        13.02
IVFPQ-m64-nl316-np2                                    1_085.54        96.91     1_182.45       0.4802          1.0391            1.0376        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_157.95       108.58     1_266.53       0.3719          1.0646            1.0654        15.50
IVFPQ-m32-nl316-np4                                    1_011.93       105.46     1_117.38       0.3827          1.0595            1.0615        13.02
IVFPQ-m64-nl316-np4                                    1_085.54       124.68     1_210.22       0.4931          1.0373            1.0357        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_157.95       124.46     1_282.41       0.3789          1.0620            1.0627        15.50
IVFPQ-m32-nl316-np8                                    1_011.93       124.97     1_136.89       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np8                                    1_085.54       164.53     1_250.07       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_157.95       150.22     1_308.17       0.3830          1.0597            1.0615        15.50
IVFPQ-m32-nl316-np15                                   1_011.93       152.17     1_164.09       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np15                                   1_085.54       222.80     1_308.35       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_157.95       175.09     1_333.04       0.3833          1.0594            1.0614        15.50
IVFPQ-m32-nl316-np17                                   1_011.93       154.15     1_166.08       0.3833          1.0594            1.0614        13.02
IVFPQ-m64-nl316-np17                                   1_085.54       234.85     1_320.40       0.4950          1.0369            1.0355        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_157.95       184.32     1_342.27       0.3833          1.0594            1.0614        15.50
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
Exhaustive (query)                                        67.76     1_249.14     1_316.90       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_082.12        95.84     1_177.96       0.3188          1.1944            1.0778        10.25
SOARPQ-near-np2                                        1_082.12       109.88     1_191.99       0.3563          1.0730            1.0708        10.25
SOARPQ-near-np4                                        1_082.12       119.59     1_201.71       0.3583          1.0694            1.0702        10.25
SOARPQ-near-np7                                        1_082.12       143.02     1_225.14       0.3586          1.0691            1.0701        10.25
SOARPQ-near-np8                                        1_082.12       147.43     1_229.55       0.3586          1.0691            1.0701        10.25
SOARPQ-near-np12                                       1_082.12       170.30     1_252.41       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.3-np1                                    1_087.58        93.67     1_181.25       0.3180          1.1966            1.0779        10.25
SOARPQ-shift0.3-np2                                    1_087.58       110.53     1_198.11       0.3561          1.0731            1.0709        10.25
SOARPQ-shift0.3-np4                                    1_087.58       122.17     1_209.76       0.3582          1.0694            1.0702        10.25
SOARPQ-shift0.3-np7                                    1_087.58       145.16     1_232.74       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.3-np8                                    1_087.58       152.24     1_239.82       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.3-np12                                   1_087.58       167.40     1_254.98       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.7-np1                                    1_091.66        94.31     1_185.97       0.3170          1.1989            1.0779        10.25
SOARPQ-shift0.7-np2                                    1_091.66       122.83     1_214.50       0.3560          1.0734            1.0709        10.25
SOARPQ-shift0.7-np4                                    1_091.66       128.69     1_220.35       0.3582          1.0695            1.0702        10.25
SOARPQ-shift0.7-np7                                    1_091.66       143.38     1_235.05       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.7-np8                                    1_091.66       146.42     1_238.09       0.3586          1.0691            1.0701        10.25
SOARPQ-shift0.7-np12                                   1_091.66       173.62     1_265.28       0.3586          1.0691            1.0701        10.25
SOARPQ-orth1-np1                                       1_135.39        95.00     1_230.39       0.3184          1.1979            1.0777        10.25
SOARPQ-orth1-np2                                       1_135.39       109.51     1_244.90       0.3564          1.0729            1.0709        10.25
SOARPQ-orth1-np4                                       1_135.39       123.65     1_259.04       0.3583          1.0694            1.0702        10.25
SOARPQ-orth1-np7                                       1_135.39       140.73     1_276.12       0.3586          1.0691            1.0701        10.25
SOARPQ-orth1-np8                                       1_135.39       147.26     1_282.65       0.3586          1.0691            1.0701        10.25
SOARPQ-orth1-np12                                      1_135.39       166.40     1_301.79       0.3586          1.0691            1.0701        10.25
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.17     1_316.30     1_384.47       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      823.44        82.66       906.10       0.4796          1.0770            1.0730         6.21
IVFPQ-m64-nl111-np1                                      888.49       106.86       995.36       0.6137          1.0402            1.0354        11.21
SOARPQ-shift0.5-m32-nl111-np1                            966.24        88.50     1_054.74       0.4718          1.0889            1.0754         8.69
IVFPQ-m32-nl111-np2                                      823.44        90.82       914.26       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np2                                      888.49       118.27     1_006.77       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np2                            966.24        98.62     1_064.85       0.4866          1.0752            1.0720         8.69
IVFPQ-m32-nl111-np4                                      823.44        96.28       919.73       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np4                                      888.49       143.30     1_031.79       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np4                            966.24       109.32     1_075.56       0.4876          1.0738            1.0717         8.69
IVFPQ-m32-nl111-np5                                      823.44       102.27       925.72       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np5                                      888.49       146.35     1_034.84       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np5                            966.24       119.96     1_086.19       0.4876          1.0738            1.0717         8.69
IVFPQ-m32-nl111-np8                                      823.44       116.61       940.06       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np8                                      888.49       175.23     1_063.72       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np8                            966.24       146.57     1_112.81       0.4876          1.0738            1.0717         8.69
IVFPQ-m32-nl111-np10                                     823.44       129.94       953.39       0.4876          1.0738            1.0717         6.21
IVFPQ-m64-nl111-np10                                     888.49       196.56     1_085.05       0.6272          1.0359            1.0340        11.21
SOARPQ-shift0.5-m32-nl111-np10                           966.24       168.32     1_134.56       0.4876          1.0738            1.0717         8.69
IVFPQ-m32-nl158-np1                                      961.18        79.81     1_041.00       0.4505          1.0868            1.0811         7.77
IVFPQ-m64-nl158-np1                                    1_054.41        98.53     1_152.94       0.5626          1.0523            1.0430        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_112.32        89.36     1_201.68       0.4772          1.0794            1.0759        10.25
IVFPQ-m32-nl158-np2                                      961.18        87.48     1_048.67       0.4848          1.0742            1.0725         7.77
IVFPQ-m64-nl158-np2                                    1_054.41       116.13     1_170.54       0.6213          1.0369            1.0349        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_112.32        98.64     1_210.96       0.4886          1.0735            1.0718        10.25
IVFPQ-m32-nl158-np4                                      961.18        97.73     1_058.92       0.4900          1.0727            1.0709         7.77
IVFPQ-m64-nl158-np4                                    1_054.41       137.88     1_192.29       0.6307          1.0350            1.0335        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_112.32       111.95     1_224.27       0.4902          1.0727            1.0709        10.25
IVFPQ-m32-nl158-np7                                      961.18       109.79     1_070.98       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np7                                    1_054.41       156.64     1_211.05       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_112.32       125.16     1_237.48       0.4902          1.0727            1.0708        10.25
IVFPQ-m32-nl158-np8                                      961.18       113.63     1_074.81       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np8                                    1_054.41       168.01     1_222.42       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_112.32       132.51     1_244.83       0.4902          1.0727            1.0708        10.25
IVFPQ-m32-nl158-np12                                     961.18       133.34     1_094.53       0.4902          1.0727            1.0708         7.77
IVFPQ-m64-nl158-np12                                   1_054.41       199.82     1_254.23       0.6310          1.0349            1.0334        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_112.32       157.04     1_269.36       0.4902          1.0727            1.0708        10.25
IVFPQ-m32-nl223-np1                                      908.28        79.44       987.72       0.4066          1.1013            1.0959         9.93
IVFPQ-m64-nl223-np1                                      954.76        91.91     1_046.67       0.4893          1.0696            1.0618        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_092.75        90.06     1_182.81       0.4648          1.0813            1.0790        12.41
IVFPQ-m32-nl223-np2                                      908.28        88.91       997.19       0.4684          1.0788            1.0771         9.93
IVFPQ-m64-nl223-np2                                      954.76       109.42     1_064.18       0.5916          1.0430            1.0398        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_092.75       102.56     1_195.31       0.4845          1.0744            1.0728        12.41
IVFPQ-m32-nl223-np4                                      908.28        98.22     1_006.50       0.4891          1.0725            1.0711         9.93
IVFPQ-m64-nl223-np4                                      954.76       128.42     1_083.18       0.6300          1.0350            1.0334        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_092.75       117.23     1_209.98       0.4906          1.0722            1.0705        12.41
IVFPQ-m32-nl223-np8                                      908.28       114.17     1_022.45       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np8                                      954.76       163.84     1_118.60       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_092.75       130.70     1_223.45       0.4908          1.0720            1.0705        12.41
IVFPQ-m32-nl223-np11                                     908.28       123.37     1_031.65       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np11                                     954.76       185.48     1_140.24       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_092.75       147.69     1_240.44       0.4908          1.0720            1.0705        12.41
IVFPQ-m32-nl223-np14                                     908.28       133.26     1_041.54       0.4908          1.0720            1.0705         9.93
IVFPQ-m64-nl223-np14                                     954.76       206.88     1_161.64       0.6333          1.0343            1.0327        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_092.75       159.30     1_252.05       0.4908          1.0720            1.0705        12.41
IVFPQ-m32-nl316-np1                                      918.84        77.99       996.83       0.3502          1.1213            1.1184        13.02
IVFPQ-m64-nl316-np1                                    1_030.23        83.15     1_113.38       0.3988          1.0925            1.0871        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_166.72        85.29     1_252.01       0.4262          1.0932            1.0924        15.50
IVFPQ-m32-nl316-np2                                      918.84        84.24     1_003.08       0.4262          1.0916            1.0907        13.02
IVFPQ-m64-nl316-np2                                    1_030.23        99.46     1_129.69       0.5213          1.0576            1.0551        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_166.72        97.48     1_264.20       0.4657          1.0802            1.0791        15.50
IVFPQ-m32-nl316-np4                                      918.84        98.30     1_017.14       0.4737          1.0772            1.0760        13.02
IVFPQ-m64-nl316-np4                                    1_030.23       123.25     1_153.48       0.6052          1.0397            1.0378        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_166.72       119.25     1_285.98       0.4837          1.0745            1.0734        15.50
IVFPQ-m32-nl316-np8                                      918.84       114.43     1_033.27       0.4869          1.0734            1.0721        13.02
IVFPQ-m64-nl316-np8                                    1_030.23       164.23     1_194.46       0.6333          1.0342            1.0327        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_166.72       136.50     1_303.23       0.4871          1.0733            1.0721        15.50
IVFPQ-m32-nl316-np15                                     918.84       137.41     1_056.25       0.4874          1.0732            1.0720        13.02
IVFPQ-m64-nl316-np15                                   1_030.23       214.74     1_244.97       0.6347          1.0339            1.0324        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_166.72       164.97     1_331.69       0.4874          1.0732            1.0720        15.50
IVFPQ-m32-nl316-np17                                     918.84       152.95     1_071.79       0.4874          1.0732            1.0720        13.02
IVFPQ-m64-nl316-np17                                   1_030.23       230.49     1_260.72       0.6347          1.0339            1.0324        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_166.72       175.64     1_342.37       0.4874          1.0732            1.0720        15.50
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
Exhaustive (query)                                        68.17     1_316.30     1_384.47       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_111.02        88.19     1_199.21       0.4766          1.0798            1.0760        10.25
SOARPQ-near-np2                                        1_111.02        98.75     1_209.78       0.4888          1.0735            1.0718        10.25
SOARPQ-near-np4                                        1_111.02       110.43     1_221.45       0.4902          1.0727            1.0709        10.25
SOARPQ-near-np7                                        1_111.02       123.77     1_234.79       0.4902          1.0727            1.0708        10.25
SOARPQ-near-np8                                        1_111.02       129.17     1_240.19       0.4902          1.0727            1.0708        10.25
SOARPQ-near-np12                                       1_111.02       158.30     1_269.32       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.3-np1                                    1_109.63        91.73     1_201.36       0.4772          1.0795            1.0759        10.25
SOARPQ-shift0.3-np2                                    1_109.63        99.67     1_209.29       0.4887          1.0735            1.0718        10.25
SOARPQ-shift0.3-np4                                    1_109.63       112.93     1_222.55       0.4902          1.0727            1.0709        10.25
SOARPQ-shift0.3-np7                                    1_109.63       123.86     1_233.48       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.3-np8                                    1_109.63       137.21     1_246.83       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.3-np12                                   1_109.63       165.85     1_275.48       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.7-np1                                    1_145.79        92.97     1_238.76       0.4772          1.0794            1.0759        10.25
SOARPQ-shift0.7-np2                                    1_145.79       101.17     1_246.96       0.4887          1.0735            1.0718        10.25
SOARPQ-shift0.7-np4                                    1_145.79       110.57     1_256.36       0.4902          1.0727            1.0709        10.25
SOARPQ-shift0.7-np7                                    1_145.79       126.50     1_272.29       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.7-np8                                    1_145.79       130.39     1_276.18       0.4902          1.0727            1.0708        10.25
SOARPQ-shift0.7-np12                                   1_145.79       173.54     1_319.33       0.4902          1.0727            1.0708        10.25
SOARPQ-orth1-np1                                       1_187.52        90.17     1_277.69       0.4770          1.0795            1.0760        10.25
SOARPQ-orth1-np2                                       1_187.52       103.82     1_291.34       0.4886          1.0735            1.0718        10.25
SOARPQ-orth1-np4                                       1_187.52       137.11     1_324.63       0.4902          1.0727            1.0709        10.25
SOARPQ-orth1-np7                                       1_187.52       124.68     1_312.20       0.4902          1.0727            1.0708        10.25
SOARPQ-orth1-np8                                       1_187.52       131.41     1_318.93       0.4902          1.0727            1.0708        10.25
SOARPQ-orth1-np12                                      1_187.52       163.43     1_350.96       0.4902          1.0727            1.0708        10.25
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.32     1_344.65     1_413.97       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      871.48        72.92       944.40       0.6957          1.1946            1.1094         6.21
IVFPQ-m64-nl111-np1                                      934.67        83.20     1_017.87       0.7084          1.1834            1.0985        11.21
SOARPQ-shift0.5-m32-nl111-np1                            986.03        88.92     1_074.96       0.8094          1.0790            1.0499         8.69
IVFPQ-m32-nl111-np2                                      871.48       100.56       972.04       0.8166          1.0678            1.0432         6.21
IVFPQ-m64-nl111-np2                                      934.67       113.03     1_047.70       0.8398          1.0553            1.0292        11.21
SOARPQ-shift0.5-m32-nl111-np2                            986.03       129.80     1_115.83       0.8469          1.0463            1.0348         8.69
IVFPQ-m32-nl111-np4                                      871.48       108.70       980.18       0.8524          1.0396            1.0315         6.21
IVFPQ-m64-nl111-np4                                      934.67       158.27     1_092.94       0.8810          1.0262            1.0192        11.21
SOARPQ-shift0.5-m32-nl111-np4                            986.03       185.14     1_171.17       0.8546          1.0397            1.0314         8.69
IVFPQ-m32-nl111-np5                                      871.48       132.36     1_003.84       0.8548          1.0382            1.0308         6.21
IVFPQ-m64-nl111-np5                                      934.67       193.36     1_128.04       0.8834          1.0248            1.0185        11.21
SOARPQ-shift0.5-m32-nl111-np5                            986.03       214.46     1_200.49       0.8552          1.0389            1.0310         8.69
IVFPQ-m32-nl111-np8                                      871.48       158.73     1_030.22       0.8559          1.0375            1.0303         6.21
IVFPQ-m64-nl111-np8                                      934.67       288.45     1_223.13       0.8847          1.0241            1.0182        11.21
SOARPQ-shift0.5-m32-nl111-np8                            986.03       303.17     1_289.20       0.8559          1.0378            1.0304         8.69
IVFPQ-m32-nl111-np10                                     871.48       188.27     1_059.75       0.8560          1.0375            1.0303         6.21
IVFPQ-m64-nl111-np10                                     934.67       327.30     1_261.97       0.8848          1.0240            1.0182        11.21
SOARPQ-shift0.5-m32-nl111-np10                           986.03       330.38     1_316.41       0.8559          1.0376            1.0303         8.69
IVFPQ-m32-nl158-np1                                      979.64        67.98     1_047.62       0.6948          1.1931            1.1102         7.77
IVFPQ-m64-nl158-np1                                    1_083.63        82.08     1_165.71       0.7041          1.1842            1.1019        14.24
SOARPQ-shift0.5-m32-nl158-np1                          1_124.24        80.75     1_204.99       0.8148          1.0760            1.0451        10.25
IVFPQ-m32-nl158-np2                                      979.64        77.70     1_057.35       0.8250          1.0626            1.0379         7.77
IVFPQ-m64-nl158-np2                                    1_083.63        96.42     1_180.05       0.8439          1.0526            1.0271        14.24
SOARPQ-shift0.5-m32-nl158-np2                          1_124.24       103.28     1_227.52       0.8575          1.0419            1.0289        10.25
IVFPQ-m32-nl158-np4                                      979.64        97.78     1_077.43       0.8648          1.0337            1.0256         7.77
IVFPQ-m64-nl158-np4                                    1_083.63       135.93     1_219.56       0.8875          1.0231            1.0166        14.24
SOARPQ-shift0.5-m32-nl158-np4                          1_124.24       148.11     1_272.35       0.8669          1.0346            1.0257        10.25
IVFPQ-m32-nl158-np7                                      979.64       124.87     1_104.52       0.8688          1.0313            1.0244         7.77
IVFPQ-m64-nl158-np7                                    1_083.63       192.73     1_276.36       0.8922          1.0204            1.0155        14.24
SOARPQ-shift0.5-m32-nl158-np7                          1_124.24       210.77     1_335.00       0.8685          1.0323            1.0248        10.25
IVFPQ-m32-nl158-np8                                      979.64       132.53     1_112.17       0.8690          1.0312            1.0243         7.77
IVFPQ-m64-nl158-np8                                    1_083.63       210.73     1_294.36       0.8925          1.0203            1.0155        14.24
SOARPQ-shift0.5-m32-nl158-np8                          1_124.24       228.82     1_353.05       0.8688          1.0319            1.0245        10.25
IVFPQ-m32-nl158-np12                                     979.64       171.60     1_151.25       0.8692          1.0311            1.0243         7.77
IVFPQ-m64-nl158-np12                                   1_083.63       286.22     1_369.85       0.8927          1.0202            1.0154        14.24
SOARPQ-shift0.5-m32-nl158-np12                         1_124.24       296.10     1_420.34       0.8691          1.0313            1.0243        10.25
IVFPQ-m32-nl223-np1                                      891.33        71.11       962.44       0.6921          1.1905            1.1182         9.93
IVFPQ-m64-nl223-np1                                      987.22        77.08     1_064.30       0.6987          1.1833            1.1111        18.43
SOARPQ-shift0.5-m32-nl223-np1                          1_022.10        78.20     1_100.30       0.8163          1.0795            1.0457        12.41
IVFPQ-m32-nl223-np2                                      891.33        77.32       968.65       0.8302          1.0608            1.0354         9.93
IVFPQ-m64-nl223-np2                                      987.22        91.17     1_078.39       0.8444          1.0531            1.0266        18.43
SOARPQ-shift0.5-m32-nl223-np2                          1_022.10        96.53     1_118.63       0.8673          1.0408            1.0255        12.41
IVFPQ-m32-nl223-np4                                      891.33        94.41       985.74       0.8770          1.0286            1.0206         9.93
IVFPQ-m64-nl223-np4                                      987.22       119.37     1_106.60       0.8951          1.0206            1.0142        18.43
SOARPQ-shift0.5-m32-nl223-np4                          1_022.10       126.17     1_148.26       0.8788          1.0317            1.0210        12.41
IVFPQ-m32-nl223-np8                                      891.33       123.83     1_015.16       0.8831          1.0253            1.0190         9.93
IVFPQ-m64-nl223-np8                                      987.22       174.38     1_161.60       0.9021          1.0172            1.0128        18.43
SOARPQ-shift0.5-m32-nl223-np8                          1_022.10       192.89     1_214.98       0.8825          1.0270            1.0194        12.41
IVFPQ-m32-nl223-np11                                     891.33       144.17     1_035.50       0.8835          1.0251            1.0189         9.93
IVFPQ-m64-nl223-np11                                     987.22       224.77     1_211.99       0.9025          1.0170            1.0127        18.43
SOARPQ-shift0.5-m32-nl223-np11                         1_022.10       236.07     1_258.16       0.8832          1.0259            1.0191        12.41
IVFPQ-m32-nl223-np14                                     891.33       175.13     1_066.46       0.8836          1.0251            1.0188         9.93
IVFPQ-m64-nl223-np14                                     987.22       266.15     1_253.37       0.9026          1.0170            1.0126        18.43
SOARPQ-shift0.5-m32-nl223-np14                         1_022.10       272.17     1_294.27       0.8835          1.0254            1.0189        12.41
IVFPQ-m32-nl316-np1                                      979.79        78.57     1_058.36       0.6743          1.2059            1.1402        13.02
IVFPQ-m64-nl316-np1                                    1_185.13        82.21     1_267.34       0.6787          1.2003            1.1353        24.42
SOARPQ-shift0.5-m32-nl316-np1                          1_336.18        81.51     1_417.70       0.8095          1.0848            1.0492        15.50
IVFPQ-m32-nl316-np2                                      979.79        87.36     1_067.15       0.8249          1.0652            1.0373        13.02
IVFPQ-m64-nl316-np2                                    1_185.13        92.65     1_277.77       0.8350          1.0594            1.0302        24.42
SOARPQ-shift0.5-m32-nl316-np2                          1_336.18        93.73     1_429.91       0.8720          1.0397            1.0237        15.50
IVFPQ-m32-nl316-np4                                      979.79       125.13     1_104.91       0.8833          1.0261            1.0182        13.02
IVFPQ-m64-nl316-np4                                    1_185.13       116.19     1_301.32       0.8979          1.0200            1.0129        24.42
SOARPQ-shift0.5-m32-nl316-np4                          1_336.18       119.14     1_455.33       0.8862          1.0293            1.0184        15.50
IVFPQ-m32-nl316-np8                                      979.79       112.80     1_092.59       0.8918          1.0216            1.0159        13.02
IVFPQ-m64-nl316-np8                                    1_185.13       162.14     1_347.27       0.9071          1.0153            1.0110        24.42
SOARPQ-shift0.5-m32-nl316-np8                          1_336.18       161.35     1_497.54       0.8908          1.0239            1.0166        15.50
IVFPQ-m32-nl316-np15                                     979.79       152.04     1_131.83       0.8924          1.0213            1.0158        13.02
IVFPQ-m64-nl316-np15                                   1_185.13       244.16     1_429.29       0.9077          1.0150            1.0109        24.42
SOARPQ-shift0.5-m32-nl316-np15                         1_336.18       235.11     1_571.30       0.8921          1.0217            1.0159        15.50
IVFPQ-m32-nl316-np17                                     979.79       163.56     1_143.35       0.8924          1.0213            1.0158        13.02
IVFPQ-m64-nl316-np17                                   1_185.13       265.33     1_450.46       0.9077          1.0150            1.0109        24.42
SOARPQ-shift0.5-m32-nl316-np17                         1_336.18       260.46     1_596.65       0.8922          1.0215            1.0158        15.50
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
Exhaustive (query)                                        69.32     1_344.65     1_413.97       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_141.04        82.10     1_223.14       0.8136          1.0778            1.0433        10.25
SOARPQ-near-np2                                        1_141.04       103.11     1_244.15       0.8589          1.0401            1.0281        10.25
SOARPQ-near-np4                                        1_141.04       150.22     1_291.26       0.8676          1.0332            1.0251        10.25
SOARPQ-near-np7                                        1_141.04       210.43     1_351.46       0.8688          1.0317            1.0245        10.25
SOARPQ-near-np8                                        1_141.04       229.31     1_370.35       0.8690          1.0315            1.0244        10.25
SOARPQ-near-np12                                       1_141.04       299.78     1_440.82       0.8692          1.0312            1.0243        10.25
SOARPQ-shift0.3-np1                                    1_130.21        81.70     1_211.91       0.8171          1.0739            1.0436        10.25
SOARPQ-shift0.3-np2                                    1_130.21       103.39     1_233.60       0.8588          1.0406            1.0283        10.25
SOARPQ-shift0.3-np4                                    1_130.21       148.20     1_278.41       0.8672          1.0340            1.0255        10.25
SOARPQ-shift0.3-np7                                    1_130.21       212.15     1_342.36       0.8686          1.0321            1.0247        10.25
SOARPQ-shift0.3-np8                                    1_130.21       227.06     1_357.27       0.8689          1.0317            1.0244        10.25
SOARPQ-shift0.3-np12                                   1_130.21       299.49     1_429.70       0.8691          1.0312            1.0243        10.25
SOARPQ-shift0.7-np1                                    1_125.13        83.51     1_208.64       0.8120          1.0789            1.0467        10.25
SOARPQ-shift0.7-np2                                    1_125.13       104.80     1_229.93       0.8562          1.0433            1.0295        10.25
SOARPQ-shift0.7-np4                                    1_125.13       151.25     1_276.38       0.8666          1.0352            1.0259        10.25
SOARPQ-shift0.7-np7                                    1_125.13       210.26     1_335.39       0.8684          1.0326            1.0249        10.25
SOARPQ-shift0.7-np8                                    1_125.13       227.74     1_352.88       0.8688          1.0321            1.0246        10.25
SOARPQ-shift0.7-np12                                   1_125.13       301.57     1_426.71       0.8691          1.0313            1.0243        10.25
SOARPQ-orth1-np1                                       1_143.60        83.07     1_226.67       0.8140          1.0778            1.0441        10.25
SOARPQ-orth1-np2                                       1_143.60       102.10     1_245.70       0.8584          1.0413            1.0285        10.25
SOARPQ-orth1-np4                                       1_143.60       154.53     1_298.13       0.8672          1.0342            1.0254        10.25
SOARPQ-orth1-np7                                       1_143.60       211.19     1_354.79       0.8686          1.0322            1.0247        10.25
SOARPQ-orth1-np8                                       1_143.60       234.86     1_378.46       0.8689          1.0318            1.0244        10.25
SOARPQ-orth1-np12                                      1_143.60       300.59     1_444.19       0.8691          1.0312            1.0243        10.25
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        72.00     1_396.84     1_468.84       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                      872.08        69.87       941.95       0.7656          1.1464            1.0636         6.21
IVFPQ-m64-nl111-np1                                      942.01        76.96     1_018.97       0.7732          1.1390            1.0537        11.21
SOARPQ-orth1-m32-nl111-np1                               995.64        83.02     1_078.66       0.8458          1.0693            1.0366         8.69
IVFPQ-m32-nl111-np2                                      872.08        80.71       952.78       0.8632          1.0458            1.0276         6.21
IVFPQ-m64-nl111-np2                                      942.01       100.29     1_042.30       0.8766          1.0377            1.0203        11.21
SOARPQ-orth1-m32-nl111-np2                               995.64       114.12     1_109.76       0.8754          1.0407            1.0261         8.69
IVFPQ-m32-nl111-np4                                      872.08       105.10       977.18       0.8824          1.0311            1.0231         6.21
IVFPQ-m64-nl111-np4                                      942.01       144.02     1_086.02       0.8976          1.0229            1.0164        11.21
SOARPQ-orth1-m32-nl111-np4                               995.64       163.90     1_159.54       0.8822          1.0330            1.0235         8.69
IVFPQ-m32-nl111-np5                                      872.08       120.29       992.37       0.8833          1.0306            1.0228         6.21
IVFPQ-m64-nl111-np5                                      942.01       164.17     1_106.17       0.8985          1.0224            1.0161        11.21
SOARPQ-orth1-m32-nl111-np5                               995.64       196.25     1_191.89       0.8829          1.0320            1.0232         8.69
IVFPQ-m32-nl111-np8                                      872.08       155.24     1_027.32       0.8837          1.0304            1.0227         6.21
IVFPQ-m64-nl111-np8                                      942.01       231.20     1_173.20       0.8991          1.0221            1.0160        11.21
SOARPQ-orth1-m32-nl111-np8                               995.64       274.96     1_270.60       0.8836          1.0308            1.0228         8.69
IVFPQ-m32-nl111-np10                                     872.08       182.82     1_054.90       0.8838          1.0304            1.0227         6.21
IVFPQ-m64-nl111-np10                                     942.01       278.80     1_220.80       0.8992          1.0221            1.0160        11.21
SOARPQ-orth1-m32-nl111-np10                              995.64       311.89     1_307.53       0.8838          1.0306            1.0227         8.69
IVFPQ-m32-nl158-np1                                    1_016.33        71.51     1_087.84       0.7548          1.1545            1.0732         7.77
IVFPQ-m64-nl158-np1                                    1_120.03        76.21     1_196.24       0.7599          1.1495            1.0669        14.24
SOARPQ-orth1-m32-nl158-np1                             1_186.39        80.22     1_266.61       0.8459          1.0754            1.0343        10.25
IVFPQ-m32-nl158-np2                                    1_016.33        79.34     1_095.66       0.8655          1.0455            1.0254         7.77
IVFPQ-m64-nl158-np2                                    1_120.03        95.24     1_215.27       0.8747          1.0401            1.0200        14.24
SOARPQ-orth1-m32-nl158-np2                             1_186.39       105.90     1_292.29       0.8830          1.0403            1.0229        10.25
IVFPQ-m32-nl158-np4                                    1_016.33        98.69     1_115.02       0.8931          1.0254            1.0186         7.77
IVFPQ-m64-nl158-np4                                    1_120.03       132.23     1_252.27       0.9032          1.0200            1.0142        14.24
SOARPQ-orth1-m32-nl158-np4                             1_186.39       148.67     1_335.06       0.8929          1.0294            1.0192        10.25
IVFPQ-m32-nl158-np7                                    1_016.33       130.40     1_146.73       0.8955          1.0241            1.0180         7.77
IVFPQ-m64-nl158-np7                                    1_120.03       182.97     1_303.00       0.9057          1.0188            1.0135        14.24
SOARPQ-orth1-m32-nl158-np7                             1_186.39       209.75     1_396.14       0.8951          1.0255            1.0182        10.25
IVFPQ-m32-nl158-np8                                    1_016.33       143.18     1_159.51       0.8956          1.0241            1.0180         7.77
IVFPQ-m64-nl158-np8                                    1_120.03       206.11     1_326.14       0.9059          1.0187            1.0134        14.24
SOARPQ-orth1-m32-nl158-np8                             1_186.39       231.89     1_418.28       0.8953          1.0250            1.0181        10.25
IVFPQ-m32-nl158-np12                                   1_016.33       180.48     1_196.81       0.8957          1.0241            1.0179         7.77
IVFPQ-m64-nl158-np12                                   1_120.03       282.73     1_402.76       0.9059          1.0187            1.0134        14.24
SOARPQ-orth1-m32-nl158-np12                            1_186.39       301.65     1_488.04       0.8956          1.0243            1.0180        10.25
IVFPQ-m32-nl223-np1                                      915.00        74.64       989.63       0.7294          1.1793            1.1017         9.93
IVFPQ-m64-nl223-np1                                    1_007.01        76.75     1_083.76       0.7345          1.1745            1.0950        18.43
SOARPQ-orth1-m32-nl223-np1                             1_125.95        77.75     1_203.70       0.8375          1.0792            1.0390        12.41
IVFPQ-m32-nl223-np2                                      915.00        77.49       992.49       0.8598          1.0505            1.0273         9.93
IVFPQ-m64-nl223-np2                                    1_007.01        88.25     1_095.26       0.8698          1.0453            1.0215        18.43
SOARPQ-orth1-m32-nl223-np2                             1_125.95        92.65     1_218.60       0.8844          1.0391            1.0216        12.41
IVFPQ-m32-nl223-np4                                      915.00        92.79     1_007.79       0.8955          1.0240            1.0173         9.93
IVFPQ-m64-nl223-np4                                    1_007.01       116.24     1_123.25       0.9083          1.0185            1.0126        18.43
SOARPQ-orth1-m32-nl223-np4                             1_125.95       123.86     1_249.81       0.8948          1.0289            1.0180        12.41
IVFPQ-m32-nl223-np8                                      915.00       119.10     1_034.10       0.8985          1.0224            1.0166         9.93
IVFPQ-m64-nl223-np8                                    1_007.01       162.75     1_169.77       0.9116          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np8                             1_125.95       176.46     1_302.40       0.8980          1.0237            1.0169        12.41
IVFPQ-m32-nl223-np11                                     915.00       143.28     1_058.28       0.8986          1.0223            1.0166         9.93
IVFPQ-m64-nl223-np11                                   1_007.01       199.21     1_206.22       0.9118          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np11                            1_125.95       219.06     1_345.01       0.8984          1.0228            1.0167        12.41
IVFPQ-m32-nl223-np14                                     915.00       159.84     1_074.84       0.8986          1.0223            1.0166         9.93
IVFPQ-m64-nl223-np14                                   1_007.01       245.52     1_252.53       0.9118          1.0168            1.0119        18.43
SOARPQ-orth1-m32-nl223-np14                            1_125.95       260.68     1_386.63       0.8986          1.0224            1.0166        12.41
IVFPQ-m32-nl316-np1                                      957.44        80.22     1_037.66       0.7028          1.2100            1.1294        13.02
IVFPQ-m64-nl316-np1                                    1_065.36        78.95     1_144.31       0.7064          1.2057            1.1253        24.42
SOARPQ-orth1-m32-nl316-np1                             1_212.57        81.99     1_294.57       0.8276          1.0861            1.0428        15.50
IVFPQ-m32-nl316-np2                                      957.44        80.58     1_038.02       0.8497          1.0588            1.0299        13.02
IVFPQ-m64-nl316-np2                                    1_065.36        91.07     1_156.43       0.8576          1.0543            1.0245        24.42
SOARPQ-orth1-m32-nl316-np2                             1_212.57        93.38     1_305.96       0.8875          1.0362            1.0203        15.50
IVFPQ-m32-nl316-np4                                      957.44        92.26     1_049.69       0.8998          1.0219            1.0153        13.02
IVFPQ-m64-nl316-np4                                    1_065.36       108.90     1_174.27       0.9104          1.0175            1.0115        24.42
SOARPQ-orth1-m32-nl316-np4                             1_212.57       114.95     1_327.52       0.9009          1.0258            1.0159        15.50
IVFPQ-m32-nl316-np8                                      957.44       114.90     1_072.34       0.9045          1.0193            1.0143        13.02
IVFPQ-m64-nl316-np8                                    1_065.36       152.41     1_217.77       0.9154          1.0150            1.0105        24.42
SOARPQ-orth1-m32-nl316-np8                             1_212.57       154.44     1_367.01       0.9040          1.0211            1.0146        15.50
IVFPQ-m32-nl316-np15                                     957.44       152.91     1_110.35       0.9047          1.0192            1.0143        13.02
IVFPQ-m64-nl316-np15                                   1_065.36       231.01     1_296.37       0.9156          1.0149            1.0105        24.42
SOARPQ-orth1-m32-nl316-np15                            1_212.57       228.85     1_441.43       0.9046          1.0195            1.0143        15.50
IVFPQ-m32-nl316-np17                                     957.44       167.67     1_125.11       0.9047          1.0192            1.0143        13.02
IVFPQ-m64-nl316-np17                                   1_065.36       253.45     1_318.81       0.9156          1.0149            1.0105        24.42
SOARPQ-orth1-m32-nl316-np17                            1_212.57       250.18     1_462.75       0.9047          1.0193            1.0143        15.50
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
Exhaustive (query)                                        72.00     1_396.84     1_468.84       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        1_148.22        83.16     1_231.38       0.8522          1.0645            1.0324        10.25
SOARPQ-near-np2                                        1_148.22       106.84     1_255.06       0.8856          1.0347            1.0219        10.25
SOARPQ-near-np4                                        1_148.22       146.40     1_294.62       0.8939          1.0269            1.0187        10.25
SOARPQ-near-np7                                        1_148.22       201.82     1_350.04       0.8955          1.0245            1.0181        10.25
SOARPQ-near-np8                                        1_148.22       227.88     1_376.09       0.8955          1.0243            1.0180        10.25
SOARPQ-near-np12                                       1_148.22       297.51     1_445.72       0.8957          1.0241            1.0179        10.25
SOARPQ-shift0.3-np1                                    1_147.76        80.94     1_228.70       0.8504          1.0685            1.0339        10.25
SOARPQ-shift0.3-np2                                    1_147.76       103.34     1_251.10       0.8839          1.0381            1.0227        10.25
SOARPQ-shift0.3-np4                                    1_147.76       147.04     1_294.80       0.8928          1.0287            1.0192        10.25
SOARPQ-shift0.3-np7                                    1_147.76       206.42     1_354.18       0.8951          1.0251            1.0182        10.25
SOARPQ-shift0.3-np8                                    1_147.76       226.15     1_373.90       0.8953          1.0247            1.0181        10.25
SOARPQ-shift0.3-np12                                   1_147.76       299.86     1_447.62       0.8956          1.0242            1.0180        10.25
SOARPQ-shift0.7-np1                                    1_145.68        82.19     1_227.87       0.8449          1.0770            1.0360        10.25
SOARPQ-shift0.7-np2                                    1_145.68       104.34     1_250.02       0.8806          1.0438            1.0238        10.25
SOARPQ-shift0.7-np4                                    1_145.68       149.57     1_295.25       0.8912          1.0318            1.0198        10.25
SOARPQ-shift0.7-np7                                    1_145.68       207.37     1_353.05       0.8946          1.0263            1.0184        10.25
SOARPQ-shift0.7-np8                                    1_145.68       223.97     1_369.65       0.8949          1.0256            1.0183        10.25
SOARPQ-shift0.7-np12                                   1_145.68       295.70     1_441.38       0.8956          1.0244            1.0180        10.25
SOARPQ-orth1-np1                                       1_171.08        82.67     1_253.74       0.8459          1.0754            1.0343        10.25
SOARPQ-orth1-np2                                       1_171.08       103.25     1_274.33       0.8830          1.0403            1.0229        10.25
SOARPQ-orth1-np4                                       1_171.08       147.15     1_318.22       0.8929          1.0294            1.0192        10.25
SOARPQ-orth1-np7                                       1_171.08       211.38     1_382.45       0.8951          1.0255            1.0182        10.25
SOARPQ-orth1-np8                                       1_171.08       232.19     1_403.27       0.8953          1.0250            1.0181        10.25
SOARPQ-orth1-np12                                      1_171.08       299.88     1_470.95       0.8956          1.0243            1.0180        10.25
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.28     1_255.90     1_324.18       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_811.82       120.70     2_932.52       0.3599          1.0700            1.0689         7.46
IVFOPQ-m64-nl111-np1                                   3_184.48       145.20     3_329.68       0.4580          1.0469            1.0429        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_069.06       128.69     3_197.74       0.3418          1.1816            1.0712         9.94
IVFOPQ-m32-nl111-np2                                   2_811.82       130.94     2_942.76       0.3628          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np2                                   3_184.48       159.28     3_343.76       0.4645          1.0449            1.0419        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_069.06       141.03     3_210.09       0.3618          1.0699            1.0685         9.94
IVFOPQ-m32-nl111-np4                                   2_811.82       140.72     2_952.55       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np4                                   3_184.48       178.22     3_362.70       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_069.06       156.60     3_225.65       0.3630          1.0687            1.0682         9.94
IVFOPQ-m32-nl111-np5                                   2_811.82       145.35     2_957.17       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np5                                   3_184.48       192.77     3_377.25       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_069.06       167.98     3_237.04       0.3630          1.0687            1.0682         9.94
IVFOPQ-m32-nl111-np8                                   2_811.82       159.78     2_971.61       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np8                                   3_184.48       215.08     3_399.56       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_069.06       195.01     3_264.06       0.3630          1.0687            1.0682         9.94
IVFOPQ-m32-nl111-np10                                  2_811.82       168.17     2_979.99       0.3630          1.0687            1.0682         7.46
IVFOPQ-m64-nl111-np10                                  3_184.48       239.36     3_423.84       0.4649          1.0448            1.0418        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_069.06       200.54     3_269.60       0.3630          1.0687            1.0682         9.94
IVFOPQ-m32-nl158-np1                                   3_259.03       123.16     3_382.19       0.3622          1.0670            1.0676         9.27
IVFOPQ-m64-nl158-np1                                   3_642.71       135.57     3_778.28       0.4614          1.0452            1.0420        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_510.76       131.10     3_641.85       0.3354          1.1484            1.0714        11.75
IVFOPQ-m32-nl158-np2                                   3_259.03       130.86     3_389.89       0.3691          1.0644            1.0659         9.27
IVFOPQ-m64-nl158-np2                                   3_642.71       156.50     3_799.22       0.4755          1.0416            1.0398        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_510.76       148.09     3_658.85       0.3674          1.0679            1.0663        11.75
IVFOPQ-m32-nl158-np4                                   3_259.03       140.33     3_399.36       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np4                                   3_642.71       179.08     3_821.79       0.4767          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_510.76       157.81     3_668.56       0.3694          1.0645            1.0659        11.75
IVFOPQ-m32-nl158-np7                                   3_259.03       151.55     3_410.58       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np7                                   3_642.71       203.42     3_846.14       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_510.76       177.47     3_688.23       0.3695          1.0643            1.0658        11.75
IVFOPQ-m32-nl158-np8                                   3_259.03       153.34     3_412.37       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np8                                   3_642.71       210.68     3_853.39       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_510.76       184.25     3_695.01       0.3695          1.0643            1.0658        11.75
IVFOPQ-m32-nl158-np12                                  3_259.03       171.81     3_430.84       0.3695          1.0643            1.0658         9.27
IVFOPQ-m64-nl158-np12                                  3_642.71       241.98     3_884.69       0.4768          1.0414            1.0396        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_510.76       203.41     3_714.17       0.3695          1.0643            1.0658        11.75
IVFOPQ-m32-nl223-np1                                   3_426.30       118.55     3_544.85       0.3637          1.0657            1.0659        11.43
IVFOPQ-m64-nl223-np1                                   3_790.51       119.41     3_909.92       0.4528          1.0457            1.0425        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_641.41       128.12     3_769.53       0.3376          1.1180            1.0711        13.91
IVFOPQ-m32-nl223-np2                                   3_426.30       125.45     3_551.75       0.3792          1.0607            1.0625        11.43
IVFOPQ-m64-nl223-np2                                   3_790.51       138.74     3_929.25       0.4857          1.0388            1.0377        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_641.41       143.73     3_785.14       0.3752          1.0635            1.0642        13.91
IVFOPQ-m32-nl223-np4                                   3_426.30       136.77     3_563.06       0.3813          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np4                                   3_790.51       171.43     3_961.93       0.4923          1.0378            1.0366        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_641.41       157.70     3_799.11       0.3804          1.0609            1.0622        13.91
IVFOPQ-m32-nl223-np8                                   3_426.30       157.04     3_583.34       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np8                                   3_790.51       209.03     3_999.53       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_641.41       175.44     3_816.85       0.3815          1.0601            1.0619        13.91
IVFOPQ-m32-nl223-np11                                  3_426.30       163.40     3_589.70       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np11                                  3_790.51       229.63     4_020.14       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_641.41       193.37     3_834.78       0.3815          1.0601            1.0619        13.91
IVFOPQ-m32-nl223-np14                                  3_426.30       176.21     3_602.51       0.3815          1.0601            1.0619        11.43
IVFOPQ-m64-nl223-np14                                  3_790.51       254.10     4_044.61       0.4927          1.0377            1.0365        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_641.41       214.78     3_856.19       0.3815          1.0601            1.0619        13.91
IVFOPQ-m32-nl316-np1                                   3_465.19       127.22     3_592.41       0.3642          1.0646            1.0652        15.02
IVFOPQ-m64-nl316-np1                                   3_893.13       122.05     4_015.18       0.4453          1.0454            1.0429        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_710.07       122.04     3_832.10       0.3360          1.1165            1.0691        17.50
IVFOPQ-m32-nl316-np2                                   3_465.19       123.92     3_589.11       0.3854          1.0581            1.0608        15.02
IVFOPQ-m64-nl316-np2                                   3_893.13       135.34     4_028.47       0.4877          1.0369            1.0368        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_710.07       135.18     3_845.24       0.3812          1.0609            1.0625        17.50
IVFOPQ-m32-nl316-np4                                   3_465.19       135.36     3_600.55       0.3905          1.0570            1.0595        15.02
IVFOPQ-m64-nl316-np4                                   3_893.13       159.87     4_053.00       0.5013          1.0349            1.0349        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_710.07       156.53     3_866.59       0.3874          1.0588            1.0605        17.50
IVFOPQ-m32-nl316-np8                                   3_465.19       153.49     3_618.68       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np8                                   3_893.13       201.51     4_094.64       0.5031          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_710.07       178.49     3_888.55       0.3908          1.0570            1.0595        17.50
IVFOPQ-m32-nl316-np15                                  3_465.19       190.68     3_655.87       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np15                                  3_893.13       256.71     4_149.84       0.5031          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_710.07       210.37     3_920.43       0.3910          1.0568            1.0594        17.50
IVFOPQ-m32-nl316-np17                                  3_465.19       187.80     3_652.98       0.3910          1.0568            1.0594        15.02
IVFOPQ-m64-nl316-np17                                  3_893.13       273.17     4_166.30       0.5031          1.0346            1.0346        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_710.07       212.30     3_922.37       0.3910          1.0568            1.0594        17.50
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
Exhaustive (query)                                        68.28     1_255.90     1_324.18       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_503.57       141.23     3_644.80       0.3369          1.1458            1.0712        11.75
SOAROPQ-near-np2                                       3_503.57       144.91     3_648.48       0.3678          1.0673            1.0663        11.75
SOAROPQ-near-np4                                       3_503.57       155.28     3_658.86       0.3694          1.0645            1.0659        11.75
SOAROPQ-near-np7                                       3_503.57       173.22     3_676.79       0.3695          1.0643            1.0658        11.75
SOAROPQ-near-np8                                       3_503.57       180.35     3_683.93       0.3695          1.0643            1.0658        11.75
SOAROPQ-near-np12                                      3_503.57       204.50     3_708.07       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.3-np1                                   3_544.02       141.59     3_685.60       0.3360          1.1474            1.0713        11.75
SOAROPQ-shift0.3-np2                                   3_544.02       143.49     3_687.50       0.3675          1.0678            1.0663        11.75
SOAROPQ-shift0.3-np4                                   3_544.02       157.11     3_701.13       0.3694          1.0645            1.0659        11.75
SOAROPQ-shift0.3-np7                                   3_544.02       178.28     3_722.30       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.3-np8                                   3_544.02       181.13     3_725.15       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.3-np12                                  3_544.02       200.37     3_744.39       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.7-np1                                   3_529.69       142.77     3_672.46       0.3353          1.1487            1.0714        11.75
SOAROPQ-shift0.7-np2                                   3_529.69       143.32     3_673.02       0.3673          1.0679            1.0664        11.75
SOAROPQ-shift0.7-np4                                   3_529.69       157.72     3_687.41       0.3694          1.0645            1.0659        11.75
SOAROPQ-shift0.7-np7                                   3_529.69       174.40     3_704.09       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.7-np8                                   3_529.69       186.52     3_716.21       0.3695          1.0643            1.0658        11.75
SOAROPQ-shift0.7-np12                                  3_529.69       200.26     3_729.95       0.3695          1.0643            1.0658        11.75
SOAROPQ-orth1-np1                                      3_617.67       142.81     3_760.48       0.3365          1.1479            1.0713        11.75
SOAROPQ-orth1-np2                                      3_617.67       146.84     3_764.51       0.3676          1.0677            1.0663        11.75
SOAROPQ-orth1-np4                                      3_617.67       160.60     3_778.28       0.3694          1.0645            1.0659        11.75
SOAROPQ-orth1-np7                                      3_617.67       180.38     3_798.05       0.3695          1.0643            1.0658        11.75
SOAROPQ-orth1-np8                                      3_617.67       181.56     3_799.24       0.3695          1.0643            1.0658        11.75
SOAROPQ-orth1-np12                                     3_617.67       202.48     3_820.16       0.3695          1.0643            1.0658        11.75
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.87     1_340.40     1_408.26       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   3_023.76       137.04     3_160.80       0.6633          1.0306            1.0248         7.46
IVFOPQ-m64-nl111-np1                                   3_342.70       147.21     3_489.91       0.7519          1.0179            1.0115        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_222.66       129.35     3_352.01       0.6701          1.0307            1.0246         9.94
IVFOPQ-m32-nl111-np2                                   3_023.76       129.24     3_153.00       0.6816          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np2                                   3_342.70       155.96     3_498.66       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_222.66       139.96     3_362.62       0.6813          1.0253            1.0238         9.94
IVFOPQ-m32-nl111-np4                                   3_023.76       141.30     3_165.06       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np4                                   3_342.70       174.58     3_517.29       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_222.66       154.25     3_376.92       0.6817          1.0251            1.0237         9.94
IVFOPQ-m32-nl111-np5                                   3_023.76       147.95     3_171.71       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np5                                   3_342.70       182.54     3_525.24       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_222.66       158.62     3_381.28       0.6817          1.0251            1.0237         9.94
IVFOPQ-m32-nl111-np8                                   3_023.76       159.31     3_183.07       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np8                                   3_342.70       213.14     3_555.84       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_222.66       194.73     3_417.39       0.6817          1.0251            1.0237         9.94
IVFOPQ-m32-nl111-np10                                  3_023.76       175.74     3_199.51       0.6817          1.0251            1.0237         7.46
IVFOPQ-m64-nl111-np10                                  3_342.70       235.49     3_578.20       0.7742          1.0121            1.0109        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_222.66       209.43     3_432.10       0.6817          1.0251            1.0237         9.94
IVFOPQ-m32-nl158-np1                                   3_350.75       128.44     3_479.19       0.6010          1.0434            1.0325         9.27
IVFOPQ-m64-nl158-np1                                   3_765.02       138.16     3_903.17       0.6719          1.0313            1.0180        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_577.96       133.41     3_711.37       0.6627          1.0298            1.0259        11.75
IVFOPQ-m32-nl158-np2                                   3_350.75       128.43     3_479.17       0.6732          1.0265            1.0245         9.27
IVFOPQ-m64-nl158-np2                                   3_765.02       156.82     3_921.83       0.7628          1.0137            1.0113        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_577.96       139.05     3_717.02       0.6827          1.0249            1.0236        11.75
IVFOPQ-m32-nl158-np4                                   3_350.75       139.05     3_489.80       0.6846          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np4                                   3_765.02       171.48     3_936.50       0.7779          1.0116            1.0105        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_577.96       158.82     3_736.78       0.6851          1.0244            1.0233        11.75
IVFOPQ-m32-nl158-np7                                   3_350.75       151.95     3_502.70       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np7                                   3_765.02       200.01     3_965.02       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_577.96       171.60     3_749.57       0.6850          1.0244            1.0233        11.75
IVFOPQ-m32-nl158-np8                                   3_350.75       156.95     3_507.70       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np8                                   3_765.02       206.95     3_971.97       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_577.96       182.45     3_760.41       0.6850          1.0244            1.0233        11.75
IVFOPQ-m32-nl158-np12                                  3_350.75       172.73     3_523.48       0.6850          1.0244            1.0233         9.27
IVFOPQ-m64-nl158-np12                                  3_765.02       239.16     4_004.17       0.7785          1.0115            1.0104        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_577.96       207.90     3_785.87       0.6850          1.0244            1.0233        11.75
IVFOPQ-m32-nl223-np1                                   3_559.58       125.02     3_684.60       0.5147          1.0613            1.0521        11.43
IVFOPQ-m64-nl223-np1                                   3_841.02       127.78     3_968.80       0.5596          1.0501            1.0411        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_602.43       128.35     3_730.78       0.6280          1.0356            1.0312        13.91
IVFOPQ-m32-nl223-np2                                   3_559.58       127.92     3_687.50       0.6391          1.0326            1.0291        11.43
IVFOPQ-m64-nl223-np2                                   3_841.02       146.77     3_987.79       0.7153          1.0207            1.0158        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_602.43       144.62     3_747.05       0.6771          1.0259            1.0245        13.91
IVFOPQ-m32-nl223-np4                                   3_559.58       138.03     3_697.61       0.6861          1.0242            1.0230        11.43
IVFOPQ-m64-nl223-np4                                   3_841.02       168.00     4_009.01       0.7776          1.0119            1.0108        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_602.43       158.26     3_760.68       0.6903          1.0235            1.0224        13.91
IVFOPQ-m32-nl223-np8                                   3_559.58       155.69     3_715.27       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np8                                   3_841.02       200.78     4_041.80       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_602.43       179.99     3_782.42       0.6906          1.0234            1.0224        13.91
IVFOPQ-m32-nl223-np11                                  3_559.58       170.93     3_730.51       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np11                                  3_841.02       225.23     4_066.25       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_602.43       192.09     3_794.52       0.6906          1.0234            1.0224        13.91
IVFOPQ-m32-nl223-np14                                  3_559.58       176.46     3_736.04       0.6906          1.0234            1.0224        11.43
IVFOPQ-m64-nl223-np14                                  3_841.02       248.89     4_089.90       0.7831          1.0111            1.0101        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_602.43       204.14     3_806.57       0.6906          1.0234            1.0224        13.91
IVFOPQ-m32-nl316-np1                                   3_444.30       124.59     3_568.89       0.4109          1.0844            1.0784        15.02
IVFOPQ-m64-nl316-np1                                   3_932.17       121.08     4_053.25       0.4291          1.0745            1.0679        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_714.45       122.79     3_837.23       0.5459          1.0506            1.0474        17.50
IVFOPQ-m32-nl316-np2                                   3_444.30       133.03     3_577.33       0.5565          1.0475            1.0444        15.02
IVFOPQ-m64-nl316-np2                                   3_932.17       136.82     4_068.99       0.6052          1.0365            1.0328        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_714.45       142.30     3_856.75       0.6412          1.0320            1.0298        17.50
IVFOPQ-m32-nl316-np4                                   3_444.30       137.95     3_582.25       0.6580          1.0290            1.0270        15.02
IVFOPQ-m64-nl316-np4                                   3_932.17       163.55     4_095.72       0.7361          1.0174            1.0146        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_714.45       160.18     3_874.62       0.6857          1.0243            1.0229        17.50
IVFOPQ-m32-nl316-np8                                   3_444.30       162.69     3_606.99       0.6929          1.0231            1.0217        15.02
IVFOPQ-m64-nl316-np8                                   3_932.17       199.54     4_131.71       0.7821          1.0112            1.0102        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_714.45       178.25     3_892.70       0.6944          1.0228            1.0215        17.50
IVFOPQ-m32-nl316-np15                                  3_444.30       183.18     3_627.48       0.6949          1.0228            1.0215        15.02
IVFOPQ-m64-nl316-np15                                  3_932.17       251.68     4_183.85       0.7844          1.0109            1.0100        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_714.45       209.31     3_923.75       0.6949          1.0228            1.0215        17.50
IVFOPQ-m32-nl316-np17                                  3_444.30       188.15     3_632.45       0.6949          1.0228            1.0215        15.02
IVFOPQ-m64-nl316-np17                                  3_932.17       269.66     4_201.83       0.7844          1.0109            1.0100        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_714.45       232.37     3_946.81       0.6949          1.0228            1.0215        17.50
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
Exhaustive (query)                                        67.87     1_340.40     1_408.26       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_531.61       154.99     3_686.61       0.6615          1.0303            1.0259        11.75
SOAROPQ-near-np2                                       3_531.61       140.93     3_672.55       0.6828          1.0249            1.0236        11.75
SOAROPQ-near-np4                                       3_531.61       153.95     3_685.57       0.6851          1.0244            1.0233        11.75
SOAROPQ-near-np7                                       3_531.61       168.82     3_700.44       0.6850          1.0244            1.0233        11.75
SOAROPQ-near-np8                                       3_531.61       174.55     3_706.17       0.6850          1.0244            1.0233        11.75
SOAROPQ-near-np12                                      3_531.61       208.56     3_740.17       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.3-np1                                   3_523.42       139.58     3_663.00       0.6624          1.0299            1.0258        11.75
SOAROPQ-shift0.3-np2                                   3_523.42       139.64     3_663.06       0.6827          1.0249            1.0236        11.75
SOAROPQ-shift0.3-np4                                   3_523.42       153.13     3_676.55       0.6851          1.0244            1.0233        11.75
SOAROPQ-shift0.3-np7                                   3_523.42       167.62     3_691.04       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.3-np8                                   3_523.42       173.41     3_696.83       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.3-np12                                  3_523.42       205.77     3_729.19       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.7-np1                                   3_512.05       138.51     3_650.56       0.6626          1.0297            1.0259        11.75
SOAROPQ-shift0.7-np2                                   3_512.05       140.38     3_652.43       0.6827          1.0249            1.0236        11.75
SOAROPQ-shift0.7-np4                                   3_512.05       152.36     3_664.40       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.7-np7                                   3_512.05       168.72     3_680.77       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.7-np8                                   3_512.05       172.68     3_684.73       0.6850          1.0244            1.0233        11.75
SOAROPQ-shift0.7-np12                                  3_512.05       209.85     3_721.90       0.6850          1.0244            1.0233        11.75
SOAROPQ-orth1-np1                                      3_708.39       141.65     3_850.04       0.6622          1.0299            1.0260        11.75
SOAROPQ-orth1-np2                                      3_708.39       141.28     3_849.67       0.6826          1.0249            1.0236        11.75
SOAROPQ-orth1-np4                                      3_708.39       156.28     3_864.67       0.6850          1.0244            1.0233        11.75
SOAROPQ-orth1-np7                                      3_708.39       168.72     3_877.11       0.6850          1.0244            1.0233        11.75
SOAROPQ-orth1-np8                                      3_708.39       177.47     3_885.86       0.6850          1.0244            1.0233        11.75
SOAROPQ-orth1-np12                                     3_708.39       202.84     3_911.23       0.6850          1.0244            1.0233        11.75
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.66     1_362.31     1_429.97       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   3_030.61       116.91     3_147.52       0.7108          1.1808            1.0958         7.46
IVFOPQ-m64-nl111-np1                                   3_388.65       117.97     3_506.62       0.7147          1.1780            1.0933        12.46
SOAROPQ-shift0.5-m32-nl111-np1                         3_163.05       124.92     3_287.97       0.8389          1.0602            1.0300         9.94
IVFOPQ-m32-nl111-np2                                   3_030.61       122.28     3_152.89       0.8467          1.0518            1.0256         7.46
IVFOPQ-m64-nl111-np2                                   3_388.65       142.38     3_531.03       0.8548          1.0486            1.0219        12.46
SOAROPQ-shift0.5-m32-nl111-np2                         3_163.05       157.54     3_320.59       0.8840          1.0270            1.0175         9.94
IVFOPQ-m32-nl111-np4                                   3_030.61       145.26     3_175.87       0.8894          1.0225            1.0159         7.46
IVFOPQ-m64-nl111-np4                                   3_388.65       192.59     3_581.24       0.9004          1.0190            1.0125        12.46
SOAROPQ-shift0.5-m32-nl111-np4                         3_163.05       225.74     3_388.79       0.8922          1.0214            1.0154         9.94
IVFOPQ-m32-nl111-np5                                   3_030.61       161.88     3_192.49       0.8921          1.0210            1.0154         7.46
IVFOPQ-m64-nl111-np5                                   3_388.65       218.79     3_607.44       0.9033          1.0175            1.0120        12.46
SOAROPQ-shift0.5-m32-nl111-np5                         3_163.05       254.30     3_417.35       0.8928          1.0208            1.0153         9.94
IVFOPQ-m32-nl111-np8                                   3_030.61       206.10     3_236.71       0.8934          1.0202            1.0151         7.46
IVFOPQ-m64-nl111-np8                                   3_388.65       302.42     3_691.07       0.9049          1.0167            1.0117        12.46
SOAROPQ-shift0.5-m32-nl111-np8                         3_163.05       331.06     3_494.11       0.8934          1.0203            1.0151         9.94
IVFOPQ-m32-nl111-np10                                  3_030.61       227.24     3_257.85       0.8935          1.0202            1.0150         7.46
IVFOPQ-m64-nl111-np10                                  3_388.65       344.84     3_733.49       0.9050          1.0167            1.0116        12.46
SOAROPQ-shift0.5-m32-nl111-np10                        3_163.05       370.30     3_533.35       0.8935          1.0202            1.0150         9.94
IVFOPQ-m32-nl158-np1                                   3_361.30       112.60     3_473.89       0.7063          1.1823            1.1002         9.27
IVFOPQ-m64-nl158-np1                                   3_748.37       116.26     3_864.63       0.7095          1.1793            1.0982        15.74
SOAROPQ-shift0.5-m32-nl158-np1                         3_518.81       118.86     3_637.68       0.8383          1.0603            1.0294        11.75
IVFOPQ-m32-nl158-np2                                   3_361.30       115.06     3_476.36       0.8499          1.0500            1.0244         9.27
IVFOPQ-m64-nl158-np2                                   3_748.37       140.31     3_888.68       0.8570          1.0470            1.0204        15.74
SOAROPQ-shift0.5-m32-nl158-np2                         3_518.81       140.81     3_659.62       0.8896          1.0251            1.0157        11.75
IVFOPQ-m32-nl158-np4                                   3_361.30       137.09     3_498.38       0.8960          1.0201            1.0139         9.27
IVFOPQ-m64-nl158-np4                                   3_748.37       170.71     3_919.09       0.9055          1.0167            1.0109        15.74
SOAROPQ-shift0.5-m32-nl158-np4                         3_518.81       189.28     3_708.09       0.8998          1.0188            1.0132        11.75
IVFOPQ-m32-nl158-np7                                   3_361.30       163.00     3_524.30       0.9010          1.0174            1.0128         9.27
IVFOPQ-m64-nl158-np7                                   3_748.37       226.76     3_975.13       0.9110          1.0140            1.0098        15.74
SOAROPQ-shift0.5-m32-nl158-np7                         3_518.81       251.94     3_770.75       0.9011          1.0177            1.0128        11.75
IVFOPQ-m32-nl158-np8                                   3_361.30       175.64     3_536.94       0.9013          1.0173            1.0127         9.27
IVFOPQ-m64-nl158-np8                                   3_748.37       247.00     3_995.37       0.9112          1.0138            1.0098        15.74
SOAROPQ-shift0.5-m32-nl158-np8                         3_518.81       267.49     3_786.30       0.9013          1.0175            1.0128        11.75
IVFOPQ-m32-nl158-np12                                  3_361.30       217.25     3_578.55       0.9015          1.0172            1.0126         9.27
IVFOPQ-m64-nl158-np12                                  3_748.37       322.50     4_070.87       0.9115          1.0137            1.0097        15.74
SOAROPQ-shift0.5-m32-nl158-np12                        3_518.81       349.20     3_868.01       0.9015          1.0172            1.0127        11.75
IVFOPQ-m32-nl223-np1                                   3_442.58       116.94     3_559.51       0.6999          1.1824            1.1101        11.43
IVFOPQ-m64-nl223-np1                                   3_777.01       115.28     3_892.29       0.7028          1.1792            1.1075        19.93
SOAROPQ-shift0.5-m32-nl223-np1                         3_563.15       119.86     3_683.01       0.8355          1.0643            1.0322        13.91
IVFOPQ-m32-nl223-np2                                   3_442.58       114.57     3_557.15       0.8479          1.0517            1.0252        11.43
IVFOPQ-m64-nl223-np2                                   3_777.01       129.50     3_906.51       0.8547          1.0483            1.0209        19.93
SOAROPQ-shift0.5-m32-nl223-np2                         3_563.15       131.30     3_694.45       0.8928          1.0254            1.0149        13.91
IVFOPQ-m32-nl223-np4                                   3_442.58       132.69     3_575.27       0.9003          1.0190            1.0125        11.43
IVFOPQ-m64-nl223-np4                                   3_777.01       156.70     3_933.71       0.9096          1.0155            1.0096        19.93
SOAROPQ-shift0.5-m32-nl223-np4                         3_563.15       172.18     3_735.33       0.9049          1.0181            1.0119        13.91
IVFOPQ-m32-nl223-np8                                   3_442.58       162.13     3_604.71       0.9076          1.0155            1.0110        11.43
IVFOPQ-m64-nl223-np8                                   3_777.01       212.16     3_989.17       0.9174          1.0119            1.0081        19.93
SOAROPQ-shift0.5-m32-nl223-np8                         3_563.15       232.61     3_795.76       0.9075          1.0160            1.0111        13.91
IVFOPQ-m32-nl223-np11                                  3_442.58       185.92     3_628.50       0.9080          1.0153            1.0109        11.43
IVFOPQ-m64-nl223-np11                                  3_777.01       264.92     4_041.93       0.9179          1.0117            1.0080        19.93
SOAROPQ-shift0.5-m32-nl223-np11                        3_563.15       273.86     3_837.01       0.9079          1.0156            1.0110        13.91
IVFOPQ-m32-nl223-np14                                  3_442.58       203.31     3_645.89       0.9081          1.0152            1.0109        11.43
IVFOPQ-m64-nl223-np14                                  3_777.01       306.92     4_083.93       0.9180          1.0116            1.0080        19.93
SOAROPQ-shift0.5-m32-nl223-np14                        3_563.15       312.20     3_875.35       0.9081          1.0154            1.0109        13.91
IVFOPQ-m32-nl316-np1                                   3_449.61       121.90     3_571.51       0.6802          1.1986            1.1335        15.02
IVFOPQ-m64-nl316-np1                                   3_843.68       117.11     3_960.78       0.6816          1.1968            1.1316        26.42
SOAROPQ-shift0.5-m32-nl316-np1                         3_655.74       117.78     3_773.52       0.8253          1.0714            1.0362        17.50
IVFOPQ-m32-nl316-np2                                   3_449.61       119.18     3_568.79       0.8395          1.0575            1.0274        15.02
IVFOPQ-m64-nl316-np2                                   3_843.68       132.79     3_976.47       0.8437          1.0555            1.0255        26.42
SOAROPQ-shift0.5-m32-nl316-np2                         3_655.74       132.92     3_788.66       0.8947          1.0260            1.0140        17.50
IVFOPQ-m32-nl316-np4                                   3_449.61       130.96     3_580.57       0.9044          1.0178            1.0109        15.02
IVFOPQ-m64-nl316-np4                                   3_843.68       150.86     3_994.53       0.9109          1.0157            1.0090        26.42
SOAROPQ-shift0.5-m32-nl316-np4                         3_655.74       163.07     3_818.81       0.9104          1.0168            1.0101        17.50
IVFOPQ-m32-nl316-np8                                   3_449.61       154.84     3_604.45       0.9143          1.0131            1.0090        15.02
IVFOPQ-m64-nl316-np8                                   3_843.68       199.85     4_043.53       0.9211          1.0108            1.0073        26.42
SOAROPQ-shift0.5-m32-nl316-np8                         3_655.74       201.42     3_857.16       0.9140          1.0139            1.0092        17.50
IVFOPQ-m32-nl316-np15                                  3_449.61       201.54     3_651.15       0.9150          1.0128            1.0089        15.02
IVFOPQ-m64-nl316-np15                                  3_843.68       286.45     4_130.13       0.9219          1.0105            1.0071        26.42
SOAROPQ-shift0.5-m32-nl316-np15                        3_655.74       279.25     3_934.99       0.9149          1.0129            1.0089        17.50
IVFOPQ-m32-nl316-np17                                  3_449.61       207.00     3_656.62       0.9150          1.0128            1.0089        15.02
IVFOPQ-m64-nl316-np17                                  3_843.68       307.95     4_151.63       0.9219          1.0105            1.0071        26.42
SOAROPQ-shift0.5-m32-nl316-np17                        3_655.74       302.90     3_958.64       0.9150          1.0129            1.0089        17.50
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
Exhaustive (query)                                        67.66     1_362.31     1_429.97       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_470.44       123.37     3_593.81       0.8364          1.0630            1.0279        11.75
SOAROPQ-near-np2                                       3_470.44       141.41     3_611.85       0.8905          1.0243            1.0153        11.75
SOAROPQ-near-np4                                       3_470.44       188.00     3_658.44       0.9004          1.0182            1.0131        11.75
SOAROPQ-near-np7                                       3_470.44       250.22     3_720.66       0.9013          1.0174            1.0128        11.75
SOAROPQ-near-np8                                       3_470.44       271.33     3_741.77       0.9014          1.0173            1.0127        11.75
SOAROPQ-near-np12                                      3_470.44       348.79     3_819.23       0.9015          1.0172            1.0127        11.75
SOAROPQ-shift0.3-np1                                   3_521.79       126.23     3_648.02       0.8406          1.0586            1.0282        11.75
SOAROPQ-shift0.3-np2                                   3_521.79       142.74     3_664.54       0.8906          1.0243            1.0154        11.75
SOAROPQ-shift0.3-np4                                   3_521.79       192.10     3_713.89       0.9000          1.0185            1.0132        11.75
SOAROPQ-shift0.3-np7                                   3_521.79       251.17     3_772.96       0.9012          1.0175            1.0128        11.75
SOAROPQ-shift0.3-np8                                   3_521.79       271.23     3_793.02       0.9013          1.0174            1.0127        11.75
SOAROPQ-shift0.3-np12                                  3_521.79       351.36     3_873.15       0.9015          1.0172            1.0127        11.75
SOAROPQ-shift0.7-np1                                   3_502.05       127.26     3_629.32       0.8354          1.0629            1.0307        11.75
SOAROPQ-shift0.7-np2                                   3_502.05       145.61     3_647.66       0.8886          1.0260            1.0160        11.75
SOAROPQ-shift0.7-np4                                   3_502.05       190.81     3_692.86       0.8995          1.0191            1.0133        11.75
SOAROPQ-shift0.7-np7                                   3_502.05       248.48     3_750.54       0.9010          1.0178            1.0129        11.75
SOAROPQ-shift0.7-np8                                   3_502.05       270.42     3_772.48       0.9012          1.0176            1.0128        11.75
SOAROPQ-shift0.7-np12                                  3_502.05       344.47     3_846.52       0.9015          1.0172            1.0127        11.75
SOAROPQ-orth1-np1                                      3_552.74       134.09     3_686.83       0.8374          1.0621            1.0285        11.75
SOAROPQ-orth1-np2                                      3_552.74       142.15     3_694.88       0.8900          1.0248            1.0154        11.75
SOAROPQ-orth1-np4                                      3_552.74       202.17     3_754.91       0.9000          1.0186            1.0131        11.75
SOAROPQ-orth1-np7                                      3_552.74       247.36     3_800.10       0.9012          1.0176            1.0128        11.75
SOAROPQ-orth1-np8                                      3_552.74       272.04     3_824.78       0.9013          1.0175            1.0128        11.75
SOAROPQ-orth1-np12                                     3_552.74       338.32     3_891.06       0.9015          1.0172            1.0127        11.75
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-40-gc3db0bc), run on 2026-11-16
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        74.37     1_408.46     1_482.83       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   3_190.39       117.79     3_308.18       0.7747          1.1390            1.0539         7.46
IVFOPQ-m64-nl111-np1                                   3_395.49       113.79     3_509.29       0.7778          1.1357            1.0498        12.46
SOAROPQ-orth1-m32-nl111-np1                            3_213.49       122.93     3_336.42       0.8622          1.0564            1.0265         9.94
IVFOPQ-m32-nl111-np2                                   3_190.39       117.78     3_308.17       0.8792          1.0372            1.0196         7.46
IVFOPQ-m64-nl111-np2                                   3_395.49       135.18     3_530.67       0.8848          1.0339            1.0168        12.46
SOAROPQ-orth1-m32-nl111-np2                            3_213.49       149.22     3_362.71       0.8943          1.0287            1.0173         9.94
IVFOPQ-m32-nl111-np4                                   3_190.39       145.27     3_335.66       0.9006          1.0223            1.0155         7.46
IVFOPQ-m64-nl111-np4                                   3_395.49       179.11     3_574.60       0.9071          1.0189            1.0130        12.46
SOAROPQ-orth1-m32-nl111-np4                            3_213.49       206.99     3_420.48       0.9008          1.0232            1.0156         9.94
IVFOPQ-m32-nl111-np5                                   3_190.39       152.60     3_342.99       0.9015          1.0218            1.0153         7.46
IVFOPQ-m64-nl111-np5                                   3_395.49       202.31     3_597.81       0.9081          1.0184            1.0127        12.46
SOAROPQ-orth1-m32-nl111-np5                            3_213.49       229.88     3_443.37       0.9014          1.0226            1.0154         9.94
IVFOPQ-m32-nl111-np8                                   3_190.39       193.46     3_383.85       0.9020          1.0216            1.0151         7.46
IVFOPQ-m64-nl111-np8                                   3_395.49       272.45     3_667.94       0.9088          1.0181            1.0125        12.46
SOAROPQ-orth1-m32-nl111-np8                            3_213.49       303.44     3_516.93       0.9019          1.0218            1.0151         9.94
IVFOPQ-m32-nl111-np10                                  3_190.39       215.30     3_405.69       0.9021          1.0216            1.0151         7.46
IVFOPQ-m64-nl111-np10                                  3_395.49       312.82     3_708.32       0.9089          1.0181            1.0125        12.46
SOAROPQ-orth1-m32-nl111-np10                           3_213.49       351.14     3_564.63       0.9020          1.0217            1.0151         9.94
IVFOPQ-m32-nl158-np1                                   3_387.83       115.70     3_503.53       0.7615          1.1488            1.0665         9.27
IVFOPQ-m64-nl158-np1                                   3_776.22       115.40     3_891.62       0.7638          1.1465            1.0637        15.74
SOAROPQ-orth1-m32-nl158-np1                            3_591.85       117.89     3_709.74       0.8579          1.0630            1.0259        11.75
IVFOPQ-m32-nl158-np2                                   3_387.83       115.36     3_503.18       0.8775          1.0392            1.0188         9.27
IVFOPQ-m64-nl158-np2                                   3_776.22       132.74     3_908.96       0.8819          1.0371            1.0168        15.74
SOAROPQ-orth1-m32-nl158-np2                            3_591.85       142.79     3_734.64       0.8982          1.0286            1.0155        11.75
IVFOPQ-m32-nl158-np4                                   3_387.83       135.13     3_522.96       0.9071          1.0189            1.0130         9.27
IVFOPQ-m64-nl158-np4                                   3_776.22       167.35     3_943.57       0.9119          1.0171            1.0115        15.74
SOAROPQ-orth1-m32-nl158-np4                            3_591.85       181.96     3_773.81       0.9077          1.0205            1.0131        11.75
IVFOPQ-m32-nl158-np7                                   3_387.83       169.73     3_557.56       0.9096          1.0176            1.0124         9.27
IVFOPQ-m64-nl158-np7                                   3_776.22       224.83     4_001.05       0.9145          1.0158            1.0109        15.74
SOAROPQ-orth1-m32-nl158-np7                            3_591.85       242.49     3_834.34       0.9094          1.0183            1.0125        11.75
IVFOPQ-m32-nl158-np8                                   3_387.83       173.40     3_561.23       0.9098          1.0175            1.0124         9.27
IVFOPQ-m64-nl158-np8                                   3_776.22       241.24     4_017.46       0.9147          1.0157            1.0108        15.74
SOAROPQ-orth1-m32-nl158-np8                            3_591.85       274.07     3_865.93       0.9095          1.0181            1.0125        11.75
IVFOPQ-m32-nl158-np12                                  3_387.83       215.01     3_602.84       0.9098          1.0175            1.0124         9.27
IVFOPQ-m64-nl158-np12                                  3_776.22       317.39     4_093.61       0.9147          1.0157            1.0108        15.74
SOAROPQ-orth1-m32-nl158-np12                           3_591.85       329.34     3_921.20       0.9098          1.0176            1.0124        11.75
IVFOPQ-m32-nl223-np1                                   3_449.58       118.81     3_568.40       0.7350          1.1743            1.0954        11.43
IVFOPQ-m64-nl223-np1                                   3_834.87       114.10     3_948.97       0.7371          1.1717            1.0929        19.93
SOAROPQ-orth1-m32-nl223-np1                            3_696.65       114.96     3_811.61       0.8488          1.0698            1.0308        13.91
IVFOPQ-m32-nl223-np2                                   3_449.58       115.69     3_565.28       0.8710          1.0451            1.0209        11.43
IVFOPQ-m64-nl223-np2                                   3_834.87       126.45     3_961.33       0.8752          1.0426            1.0184        19.93
SOAROPQ-orth1-m32-nl223-np2                            3_696.65       129.84     3_826.49       0.8987          1.0296            1.0151        13.91
IVFOPQ-m32-nl223-np4                                   3_449.58       127.80     3_577.38       0.9096          1.0183            1.0120        11.43
IVFOPQ-m64-nl223-np4                                   3_834.87       150.72     3_985.59       0.9156          1.0157            1.0102        19.93
SOAROPQ-orth1-m32-nl223-np4                            3_696.65       157.39     3_854.04       0.9099          1.0206            1.0123        13.91
IVFOPQ-m32-nl223-np8                                   3_449.58       152.52     3_602.10       0.9129          1.0166            1.0113        11.43
IVFOPQ-m64-nl223-np8                                   3_834.87       202.22     4_037.09       0.9193          1.0140            1.0094        19.93
SOAROPQ-orth1-m32-nl223-np8                            3_696.65       208.73     3_905.39       0.9126          1.0174            1.0115        13.91
IVFOPQ-m32-nl223-np11                                  3_449.58       173.85     3_623.43       0.9131          1.0165            1.0113        11.43
IVFOPQ-m64-nl223-np11                                  3_834.87       241.25     4_076.13       0.9194          1.0140            1.0093        19.93
SOAROPQ-orth1-m32-nl223-np11                           3_696.65       250.14     3_946.79       0.9130          1.0168            1.0113        13.91
IVFOPQ-m32-nl223-np14                                  3_449.58       194.21     3_643.79       0.9131          1.0165            1.0113        11.43
IVFOPQ-m64-nl223-np14                                  3_834.87       280.97     4_115.85       0.9194          1.0140            1.0093        19.93
SOAROPQ-orth1-m32-nl223-np14                           3_696.65       289.88     3_986.53       0.9131          1.0166            1.0113        13.91
IVFOPQ-m32-nl316-np1                                   3_460.38       120.00     3_580.38       0.7069          1.2052            1.1249        15.02
IVFOPQ-m64-nl316-np1                                   3_903.17       118.39     4_021.56       0.7085          1.2033            1.1226        26.42
SOAROPQ-orth1-m32-nl316-np1                            3_801.88       117.38     3_919.26       0.8365          1.0789            1.0355        17.50
IVFOPQ-m32-nl316-np2                                   3_460.38       117.09     3_577.47       0.8592          1.0540            1.0239        15.02
IVFOPQ-m64-nl316-np2                                   3_903.17       127.28     4_030.45       0.8628          1.0520            1.0215        26.42
SOAROPQ-orth1-m32-nl316-np2                            3_801.88       128.37     3_930.25       0.9007          1.0281            1.0144        17.50
IVFOPQ-m32-nl316-np4                                   3_460.38       126.55     3_586.93       0.9127          1.0170            1.0107        15.02
IVFOPQ-m64-nl316-np4                                   3_903.17       154.79     4_057.96       0.9183          1.0150            1.0089        26.42
SOAROPQ-orth1-m32-nl316-np4                            3_801.88       152.01     3_953.89       0.9147          1.0182            1.0106        17.50
IVFOPQ-m32-nl316-np8                                   3_460.38       148.26     3_608.64       0.9177          1.0143            1.0097        15.02
IVFOPQ-m64-nl316-np8                                   3_903.17       190.54     4_093.71       0.9240          1.0122            1.0078        26.42
SOAROPQ-orth1-m32-nl316-np8                            3_801.88       192.45     3_994.34       0.9175          1.0151            1.0099        17.50
IVFOPQ-m32-nl316-np15                                  3_460.38       184.00     3_644.38       0.9181          1.0142            1.0096        15.02
IVFOPQ-m64-nl316-np15                                  3_903.17       270.55     4_173.72       0.9244          1.0121            1.0077        26.42
SOAROPQ-orth1-m32-nl316-np15                           3_801.88       257.14     4_059.03       0.9180          1.0142            1.0097        17.50
IVFOPQ-m32-nl316-np17                                  3_460.38       198.78     3_659.15       0.9181          1.0141            1.0096        15.02
IVFOPQ-m64-nl316-np17                                  3_903.17       283.44     4_186.61       0.9244          1.0121            1.0077        26.42
SOAROPQ-orth1-m32-nl316-np17                           3_801.88       288.65     4_090.53       0.9180          1.0142            1.0096        17.50
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
Exhaustive (query)                                        74.37     1_408.46     1_482.83       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                       3_507.00       125.09     3_632.09       0.8640          1.0547            1.0244        11.75
SOAROPQ-near-np2                                       3_507.00       139.98     3_646.98       0.9004          1.0253            1.0151        11.75
SOAROPQ-near-np4                                       3_507.00       178.43     3_685.43       0.9084          1.0190            1.0128        11.75
SOAROPQ-near-np7                                       3_507.00       237.51     3_744.51       0.9096          1.0177            1.0124        11.75
SOAROPQ-near-np8                                       3_507.00       259.31     3_766.30       0.9097          1.0176            1.0124        11.75
SOAROPQ-near-np12                                      3_507.00       322.90     3_829.90       0.9098          1.0175            1.0124        11.75
SOAROPQ-shift0.3-np1                                   3_531.51       125.47     3_656.98       0.8627          1.0570            1.0257        11.75
SOAROPQ-shift0.3-np2                                   3_531.51       141.88     3_673.38       0.8989          1.0272            1.0155        11.75
SOAROPQ-shift0.3-np4                                   3_531.51       183.06     3_714.57       0.9075          1.0202            1.0131        11.75
SOAROPQ-shift0.3-np7                                   3_531.51       240.92     3_772.43       0.9094          1.0181            1.0125        11.75
SOAROPQ-shift0.3-np8                                   3_531.51       263.94     3_795.44       0.9095          1.0179            1.0125        11.75
SOAROPQ-shift0.3-np12                                  3_531.51       329.81     3_861.32       0.9098          1.0176            1.0124        11.75
SOAROPQ-shift0.7-np1                                   3_556.72       123.99     3_680.72       0.8573          1.0641            1.0271        11.75
SOAROPQ-shift0.7-np2                                   3_556.72       138.58     3_695.31       0.8958          1.0315            1.0162        11.75
SOAROPQ-shift0.7-np4                                   3_556.72       185.44     3_742.16       0.9060          1.0224            1.0135        11.75
SOAROPQ-shift0.7-np7                                   3_556.72       243.96     3_800.68       0.9089          1.0192            1.0127        11.75
SOAROPQ-shift0.7-np8                                   3_556.72       264.29     3_821.01       0.9092          1.0187            1.0126        11.75
SOAROPQ-shift0.7-np12                                  3_556.72       334.36     3_891.09       0.9097          1.0177            1.0124        11.75
SOAROPQ-orth1-np1                                      3_592.22       126.11     3_718.33       0.8579          1.0630            1.0259        11.75
SOAROPQ-orth1-np2                                      3_592.22       141.50     3_733.71       0.8982          1.0286            1.0155        11.75
SOAROPQ-orth1-np4                                      3_592.22       187.23     3_779.45       0.9077          1.0205            1.0131        11.75
SOAROPQ-orth1-np7                                      3_592.22       248.31     3_840.53       0.9094          1.0183            1.0125        11.75
SOAROPQ-orth1-np8                                      3_592.22       262.79     3_855.01       0.9095          1.0181            1.0125        11.75
SOAROPQ-orth1-np12                                     3_592.22       328.13     3_920.35       0.9098          1.0176            1.0124        11.75
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
