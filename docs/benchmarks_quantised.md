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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.58       631.96       643.54       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.58     6_222.50     6_234.09       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.36     1_184.91     1_197.27       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    12.36    12_107.03    12_119.40       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                               76.21        85.33       161.54       0.9793          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                               76.21        96.52       172.74       0.9827          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                               76.21       128.23       204.44       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                     76.21     1_343.02     1_419.24       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                               99.14        88.27       187.41       0.9807          1.0002            1.0000         9.20
IVF-BF16-nl387-np27 (query)                               99.14       115.41       214.55       0.9828          1.0001            1.0000         9.20
IVF-BF16-nl387 (self)                                     99.14     1_173.49     1_272.64       0.9798          1.0001            1.0000         9.20
IVF-BF16-nl547-np23 (query)                              139.78        80.21       219.98       0.9768          1.0005            1.0000         9.22
IVF-BF16-nl547-np27 (query)                              139.78        89.53       229.31       0.9812          1.0002            1.0000         9.22
IVF-BF16-nl547-np33 (query)                              139.78       111.69       251.46       0.9827          1.0001            1.0000         9.22
IVF-BF16-nl547 (self)                                    139.78     1_072.96     1_212.74       0.9798          1.0001            1.0000         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.77       721.49       733.26       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.77     7_009.35     7_021.12       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   13.16     1_219.07     1_232.23       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    13.16    12_470.54    12_483.70       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                               80.64        90.21       170.85       0.8841          1.0075            1.0021         9.48
IVF-BF16-nl273-np16 (query)                               80.64       105.43       186.06       0.8867          1.0072            1.0019         9.48
IVF-BF16-nl273-np23 (query)                               80.64       144.22       224.85       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                     80.64     1_484.77     1_565.40       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                               93.56        93.00       186.56       0.8861          1.0073            1.0020         9.49
IVF-BF16-nl387-np27 (query)                               93.56       122.50       216.06       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                     93.56     1_260.73     1_354.29       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              143.34        85.48       228.82       0.8841          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              143.34        94.23       237.57       0.8862          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              143.34       111.85       255.19       0.8871          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    143.34     1_132.78     1_276.12       0.8852          1.0073            1.0020         9.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.31       640.49       651.80       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.31     6_222.87     6_234.18       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.09     1_172.32     1_186.41       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    14.09    11_646.76    11_660.85       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                               74.73        82.24       156.98       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                               74.73        88.41       163.14       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                               74.73       116.06       190.79       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                     74.73     1_145.41     1_220.15       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                               88.07        82.60       170.67       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387-np27 (query)                               88.07       100.04       188.11       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387 (self)                                     88.07     1_004.52     1_092.59       0.9184          1.0030            1.0021         9.20
IVF-BF16-nl547-np23 (query)                              132.62        76.63       209.25       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np27 (query)                              132.62        83.13       215.75       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np33 (query)                              132.62       102.82       235.44       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547 (self)                                    132.62       943.44     1_076.05       0.9184          1.0030            1.0021         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.25       636.98       648.23       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.25     6_176.43     6_187.68       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.23     1_169.62     1_182.85       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    13.23    11_627.60    11_640.83       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                               94.09        77.89       171.98       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                               94.09        87.59       181.68       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                               94.09       114.66       208.75       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                     94.09     1_154.77     1_248.86       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                               97.80        79.36       177.16       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387-np27 (query)                               97.80       100.16       197.95       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387 (self)                                     97.80       998.13     1_095.93       0.9429          1.0017            1.0009         9.20
IVF-BF16-nl547-np23 (query)                              154.00        74.55       228.55       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np27 (query)                              154.00        82.92       236.92       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np33 (query)                              154.00        93.25       247.25       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547 (self)                                    154.00       941.50     1_095.50       0.9429          1.0017            1.0009         9.22
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        51.77     1_215.91     1_267.68       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         51.77    11_820.39    11_872.16       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   57.21     5_174.23     5_231.44       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    57.21    51_927.82    51_985.03       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              208.29       258.68       466.96       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              208.29       295.39       503.68       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              208.29       398.17       606.45       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    208.29     4_050.12     4_258.41       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              229.89       266.40       496.29       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              229.89       349.79       579.68       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    229.89     3_503.09     3_732.98       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                              330.87       253.79       584.66       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                              330.87       279.20       610.07       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                              330.87       323.37       654.24       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                    330.87     3_241.71     3_572.58       0.9679          1.0005            1.0000        36.89
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.25       631.77       643.02       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.25     6_224.27     6_235.52       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.31       964.22       981.53       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     17.31     9_728.64     9_745.95       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                                81.88        67.03       148.90       0.9227          1.0021            1.0010         5.76
IVF-SQ8-nl273-np16 (query)                                81.88        69.82       151.69       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273-np23 (query)                                81.88        90.02       171.90       0.9252          1.0018            1.0009         5.76
IVF-SQ8-nl273 (self)                                      81.88       888.32       970.20       0.9249          1.0018            1.0009         5.76
IVF-SQ8-nl387-np19 (query)                                99.43        62.58       162.01       0.9243          1.0020            1.0009         5.77
IVF-SQ8-nl387-np27 (query)                                99.43        79.81       179.24       0.9257          1.0018            1.0009         5.77
IVF-SQ8-nl387 (self)                                      99.43       791.05       890.48       0.9251          1.0018            1.0009         5.77
IVF-SQ8-nl547-np23 (query)                               153.64        56.54       210.18       0.9210          1.0023            1.0011         5.79
IVF-SQ8-nl547-np27 (query)                               153.64        63.34       216.98       0.9239          1.0020            1.0010         5.79
IVF-SQ8-nl547-np33 (query)                               153.64        74.58       228.22       0.9249          1.0018            1.0009         5.79
IVF-SQ8-nl547 (self)                                     153.64       730.25       883.89       0.9250          1.0018            1.0009         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.96       687.34       699.30       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.96     6_954.72     6_966.68       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    23.00       899.86       922.87       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     23.00     9_987.88    10_010.89       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                                96.77        59.17       155.93       0.7379          1.0360            1.0158         5.76
IVF-SQ8-nl273-np16 (query)                                96.77        66.11       162.88       0.7387          1.0359            1.0157         5.76
IVF-SQ8-nl273-np23 (query)                                96.77        92.22       188.98       0.7385          1.0360            1.0157         5.76
IVF-SQ8-nl273 (self)                                      96.77       946.32     1_043.09       0.7376          1.0361            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               108.22        60.01       168.24       0.7370          1.0360            1.0159         5.77
IVF-SQ8-nl387-np27 (query)                               108.22        77.03       185.25       0.7373          1.0359            1.0158         5.77
IVF-SQ8-nl387 (self)                                     108.22       808.51       916.73       0.7360          1.0363            1.0158         5.77
IVF-SQ8-nl547-np23 (query)                               151.37        53.93       205.29       0.7386          1.0358            1.0161         5.79
IVF-SQ8-nl547-np27 (query)                               151.37        63.55       214.91       0.7395          1.0356            1.0156         5.79
IVF-SQ8-nl547-np33 (query)                               151.37        69.73       221.10       0.7397          1.0356            1.0155         5.79
IVF-SQ8-nl547 (self)                                     151.37       739.52       890.89       0.7379          1.0361            1.0154         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.16       628.91       640.07       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.16     6_224.23     6_235.38       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.25       982.83       999.08       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     16.25     9_660.23     9_676.48       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                                80.16        59.14       139.30       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np16 (query)                                80.16        63.77       143.93       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273-np23 (query)                                80.16        80.10       160.27       0.8145          1.0166            1.0145         5.76
IVF-SQ8-nl273 (self)                                      80.16       794.00       874.16       0.8118          1.0175            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                                96.15        59.42       155.56       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387-np27 (query)                                96.15        73.52       169.67       0.8150          1.0165            1.0146         5.77
IVF-SQ8-nl387 (self)                                      96.15       693.32       789.47       0.8119          1.0175            1.0155         5.77
IVF-SQ8-nl547-np23 (query)                               141.20        55.00       196.20       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np27 (query)                               141.20        59.73       200.93       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547-np33 (query)                               141.20        66.88       208.08       0.8146          1.0165            1.0146         5.79
IVF-SQ8-nl547 (self)                                     141.20       662.59       803.79       0.8120          1.0175            1.0155         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.20       616.14       627.35       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.20     6_175.67     6_186.88       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.98       997.00     1_013.97       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     16.98     9_675.00     9_691.97       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                                79.96        58.20       138.16       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np16 (query)                                79.96        61.90       141.86       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273-np23 (query)                                79.96        79.89       159.85       0.7904          1.0265            1.0243         5.76
IVF-SQ8-nl273 (self)                                      79.96       783.47       863.43       0.7901          1.0280            1.0256         5.76
IVF-SQ8-nl387-np19 (query)                                96.21        56.23       152.43       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387-np27 (query)                                96.21        69.93       166.14       0.7899          1.0264            1.0241         5.77
IVF-SQ8-nl387 (self)                                      96.21       691.63       787.84       0.7901          1.0281            1.0256         5.77
IVF-SQ8-nl547-np23 (query)                               138.53        53.44       191.97       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np27 (query)                               138.53        57.90       196.43       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547-np33 (query)                               138.53        66.87       205.40       0.7904          1.0264            1.0242         5.79
IVF-SQ8-nl547 (self)                                     138.53       651.17       789.71       0.7903          1.0280            1.0256         5.79
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### More dimensions

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.08     1_203.74     1_252.82       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.08    11_664.83    11_713.91       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    76.77     1_164.87     1_241.64       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     76.77    11_908.94    11_985.71       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               211.99        76.81       288.80       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np16 (query)                               211.99        82.93       294.92       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273-np23 (query)                               211.99       106.05       318.04       0.8797          1.0062            1.0051        19.59
IVF-SQ8-nl273 (self)                                     211.99       856.75     1_068.74       0.8866          1.0073            1.0059        19.59
IVF-SQ8-nl387-np19 (query)                               264.30        81.37       345.67       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387-np27 (query)                               264.30        97.78       362.08       0.8803          1.0061            1.0051        19.65
IVF-SQ8-nl387 (self)                                     264.30       778.03     1_042.32       0.8866          1.0073            1.0059        19.65
IVF-SQ8-nl547-np23 (query)                               364.47        81.71       446.18       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np27 (query)                               364.47        86.84       451.31       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547-np33 (query)                               364.47        97.53       462.01       0.8796          1.0061            1.0051        19.73
IVF-SQ8-nl547 (self)                                     364.47       757.65     1_122.12       0.8867          1.0073            1.0059        19.73
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.31       648.03       659.34       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.31     6_121.82     6_133.14       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.97       970.22       986.20       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               758.07        44.92       802.99       0.9300          1.0213            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              758.07        80.26       838.33       0.9639          1.0100            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              758.07       149.47       907.54       0.9827          1.0070            1.0000        38.52
HNSW-M16-ef100 (self)                                    758.07       788.00     1_546.07       0.9640          1.0103            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_471.15        47.93     1_519.08       0.9594          1.0075            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_471.15        85.82     1_556.97       0.9832          1.0031            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_471.15       156.86     1_628.01       0.9921          1.0022            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_471.15       845.91     2_317.06       0.9834          1.0029            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_583.24        56.62     1_639.86       0.9698          1.0080            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_583.24        95.15     1_678.39       0.9879          1.0041            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_583.24       172.03     1_755.27       0.9949          1.0022            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_583.24       934.95     2_518.19       0.9878          1.0046            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_646.86        56.49     1_703.35       0.9724          1.0146            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_646.86       100.15     1_747.01       0.9894          1.0086            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_646.86       177.42     1_824.28       0.9958          1.0026            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_646.86       979.63     2_626.49       0.9901          1.0037            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          698.28        35.60       733.88       0.8772          1.0365            1.0032        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         698.28        66.08       764.36       0.9021          1.0143            1.0020        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         698.28       119.34       817.62       0.9147          1.0075            1.0014        26.32
HNSW-SQ8U-M16-ef100 (self)                               698.28       612.75     1_311.03       0.9010          1.0119            1.0020        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_348.87        36.58     1_385.45       0.8982          1.0235            1.0021        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_348.87        67.61     1_416.48       0.9147          1.0085            1.0014        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_348.87       125.70     1_474.57       0.9208          1.0038            1.0011        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_348.87       655.89     2_004.76       0.9142          1.0114            1.0014        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_446.15        39.34     1_485.48       0.9069          1.0165            1.0017        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_446.15        71.23     1_517.38       0.9183          1.0117            1.0012        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_446.15       131.84     1_577.99       0.9226          1.0065            1.0010        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_446.15       690.56     2_136.70       0.9178          1.0129            1.0012        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_527.29        42.30     1_569.59       0.9087          1.0148            1.0016        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_527.29        77.41     1_604.70       0.9196          1.0071            1.0011        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_527.29       144.39     1_671.68       0.9233          1.0036            1.0010        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_527.29       726.83     2_254.11       0.9190          1.0063            1.0012        44.63
HNSW-SQ8U-drop0 (query)                                1_335.32        65.70     1_401.02       0.8950          1.0103            1.0024        26.32
HNSW-SQ8U-drop0.001 (query)                            1_435.00        67.10     1_502.11       0.9150          1.0065            1.0014        26.32
HNSW-SQ8U-drop0.01 (query)                             1_344.57        67.82     1_412.40       0.8987          1.0084            1.0018        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.74       677.94       689.69       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.74     6_767.77     6_779.52       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    18.81       891.86       910.67       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               795.03        46.97       842.00       0.9355          1.0191            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              795.03        88.11       883.14       0.9690          1.0072            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              795.03       156.26       951.29       0.9879          1.0034            1.0000        39.09
HNSW-M16-ef100 (self)                                    795.03       835.42     1_630.45       0.9692          1.0092            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_536.64        50.89     1_587.54       0.9642          1.0086            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_536.64        90.68     1_627.33       0.9866          1.0035            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_536.64       163.70     1_700.34       0.9944          1.0018            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_536.64       885.55     2_422.20       0.9866          1.0052            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_644.41        56.47     1_700.88       0.9738          1.0022            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_644.41        99.12     1_743.53       0.9910          1.0010            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_644.41       177.51     1_821.93       0.9967          1.0007            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_644.41       988.63     2_633.04       0.9910          1.0013            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_703.36        59.32     1_762.68       0.9757          1.0249            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_703.36       103.45     1_806.80       0.9915          1.0093            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_703.36       185.19     1_888.55       0.9973          1.0006            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_703.36     1_018.70     2_722.06       0.9914          1.0213            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          706.56        34.34       740.90       0.6836          1.0614            1.0293        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         706.56        64.14       770.69       0.7087          1.0493            1.0241        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         706.56       120.08       826.64       0.7228          1.0431            1.0209        26.32
HNSW-SQ8U-M16-ef100 (self)                               706.56       607.39     1_313.95       0.7084          1.0502            1.0238        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_360.24        36.41     1_396.65       0.7065          1.0529            1.0232        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_360.24        67.17     1_427.41       0.7235          1.0420            1.0198        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_360.24       126.41     1_486.65       0.7314          1.0388            1.0185        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_360.24       647.43     2_007.68       0.7231          1.0452            1.0199        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_462.75        40.45     1_503.20       0.7184          1.0454            1.0208        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_462.75        74.57     1_537.33       0.7305          1.0385            1.0185        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_462.75       134.24     1_596.99       0.7355          1.0367            1.0175        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_462.75       703.64     2_166.39       0.7297          1.0385            1.0181        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_566.58        42.27     1_608.85       0.7216          1.0671            1.0197        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_566.58        75.95     1_642.53       0.7328          1.0443            1.0176        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_566.58       139.21     1_705.80       0.7369          1.0367            1.0168        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_566.58       740.88     2_307.46       0.7319          1.0397            1.0174        44.63
HNSW-SQ8U-drop0 (query)                                1_352.60        68.63     1_421.24       0.6649          1.0632            1.0303        26.32
HNSW-SQ8U-drop0.001 (query)                            1_365.71        66.38     1_432.09       0.7238          1.0458            1.0200        26.32
HNSW-SQ8U-drop0.01 (query)                             1_371.58        66.16     1_437.74       0.6886          1.0527            1.0260        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.39       648.07       659.46       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.39     6_272.81     6_284.20       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    15.79       964.74       980.53       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               753.69        45.85       799.54       0.9920          8.7311            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              753.69        79.87       833.56       0.9994          1.1407            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              753.69       139.05       892.74       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    753.69       768.09     1_521.77       0.9993          1.1933            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_334.74        46.08     1_380.82       0.9955          4.4608            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_334.74        79.25     1_413.99       0.9993          1.3308            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_334.74       140.00     1_474.74       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_334.74       767.99     2_102.73       0.9993          1.3837            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_377.16        47.69     1_424.85       0.9992          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_377.16        82.20     1_459.36       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_377.16       147.19     1_524.35       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_377.16       794.95     2_172.11       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_430.78        49.51     1_480.29       0.9993          1.0829            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_430.78        84.66     1_515.44       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_430.78       148.24     1_579.01       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_430.78       819.41     2_250.18       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          682.82        34.23       717.06       0.8138          1.0236            1.0149        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         682.82        63.17       745.99       0.8144          1.0166            1.0148        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         682.82       112.48       795.30       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef100 (self)                               682.82       590.47     1_273.30       0.8118          1.0176            1.0156        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_239.16        34.24     1_273.40       0.8141          1.0167            1.0148        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_239.16        62.03     1_301.19       0.8145          1.0166            1.0148        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_239.16       116.01     1_355.17       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_239.16       589.09     1_828.26       0.8119          1.0175            1.0155        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_309.60        37.27     1_346.88       0.8140          1.0691            1.0148        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_309.60        66.67     1_376.27       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_309.60       119.43     1_429.03       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_309.60       632.74     1_942.34       0.8119          1.0175            1.0155        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_361.91        39.57     1_401.47       0.8144          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_361.91        69.86     1_431.76       0.8145          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_361.91       123.85     1_485.76       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_361.91       661.90     2_023.81       0.8119          1.0175            1.0155        44.63
HNSW-SQ8U-drop0 (query)                                1_222.01        61.96     1_283.97       0.8076          1.1150            1.0158        26.32
HNSW-SQ8U-drop0.001 (query)                            1_227.46        61.58     1_289.04       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-drop0.01 (query)                             1_256.81        63.34     1_320.15       0.8044          1.0203            1.0163        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.22       629.16       640.37       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.22     6_369.03     6_380.25       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.25       973.84       991.09       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               833.97        49.68       883.64       0.9980          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              833.97        87.49       921.46       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              833.97       156.55       990.51       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    833.97       859.34     1_693.31       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_487.66        50.54     1_538.20       0.9986          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_487.66        88.16     1_575.82       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_487.66       161.74     1_649.40       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_487.66       875.41     2_363.07       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_569.83        58.60     1_628.43       0.9993          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_569.83        96.75     1_666.58       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_569.83       173.92     1_743.75       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_569.83       953.72     2_523.55       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_632.63        61.61     1_694.24       0.9994          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_632.63       100.86     1_733.49       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_632.63       176.24     1_808.87       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_632.63       988.10     2_620.72       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          758.54        35.29       793.83       0.7888          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         758.54        66.31       824.85       0.7892          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         758.54       122.90       881.44       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100 (self)                               758.54       638.68     1_397.22       0.7896          1.0282            1.0258        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_383.27        36.22     1_419.49       0.7891          1.0267            1.0244        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_383.27        67.49     1_450.76       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_383.27       124.02     1_507.29       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_383.27       642.64     2_025.91       0.7897          1.0281            1.0258        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_438.54        41.73     1_480.27       0.7891          1.0266            1.0245        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_438.54        75.64     1_514.18       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_438.54       140.98     1_579.52       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_438.54       730.74     2_169.28       0.7897          1.0281            1.0258        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_514.09        44.26     1_558.35       0.7892          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_514.09        78.96     1_593.05       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_514.09       140.20     1_654.30       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_514.09       760.92     2_275.01       0.7897          1.0281            1.0258        44.63
HNSW-SQ8U-drop0 (query)                                1_356.86        65.77     1_422.63       0.7860          1.0279            1.0254        26.32
HNSW-SQ8U-drop0.001 (query)                            1_358.46        65.71     1_424.17       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-drop0.01 (query)                             1_355.57        65.81     1_421.38       0.7830          1.0294            1.0262        26.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Euclidean (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.65     1_219.11     1_267.76       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.65    12_278.96    12_327.61       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    80.35     1_207.36     1_287.71       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_296.73        75.11     1_371.85       0.9924          1.0356            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_296.73       130.19     1_426.93       0.9951          1.0198            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_296.73       227.67     1_524.41       0.9970          1.0101            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_296.73     1_258.59     2_555.32       0.9948          1.0211            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_334.45        79.05     2_413.50       0.9953          1.0288            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_334.45       137.16     2_471.61       0.9974          1.0146            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_334.45       237.34     2_571.80       0.9985          1.0066            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_334.45     1_321.89     3_656.34       0.9976          1.0148            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_466.22        86.53     2_552.75       0.9983          1.0094            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_466.22       142.96     2_609.18       0.9991          1.0047            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_466.22       250.61     2_716.83       0.9996          1.0015            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_466.22     1_402.44     3_868.66       0.9990          1.0051            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_550.43        86.68     2_637.11       0.9989          1.0071            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_550.43       144.22     2_694.65       0.9994          1.0031            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_550.43       252.97     2_803.40       0.9998          1.0010            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_550.43     1_409.86     3_960.29       0.9995          1.0030            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          792.16        36.41       828.57       0.9289          1.0369            1.0038        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         792.16        66.30       858.46       0.9310          1.0248            1.0037        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         792.16       120.64       912.80       0.9321          1.0161            1.0037        40.05
HNSW-SQ8U-M16-ef100 (self)                               792.16       626.33     1_418.48       0.9309          1.0208            1.0038        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_436.37        38.54     1_474.91       0.9315          1.0261            1.0037        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_436.37        68.55     1_504.92       0.9326          1.0169            1.0037        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_436.37       126.04     1_562.40       0.9335          1.0099            1.0036        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_436.37       654.21     2_090.57       0.9321          1.0195            1.0037        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_554.46        39.98     1_594.45       0.9325          1.0174            1.0036        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_554.46        70.15     1_624.61       0.9332          1.0120            1.0036        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_554.46       126.21     1_680.68       0.9337          1.0086            1.0036        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_554.46       667.08     2_221.54       0.9332          1.0114            1.0036        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_610.54        40.94     1_651.48       0.9331          1.0134            1.0036        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_610.54        70.19     1_680.73       0.9334          1.0115            1.0036        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_610.54       126.18     1_736.72       0.9339          1.0081            1.0036        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_610.54       673.13     2_283.67       0.9335          1.0099            1.0036        58.36
HNSW-SQ8U-drop0 (query)                                1_461.75        65.43     1_527.18       0.8644          1.0438            1.0220        40.05
HNSW-SQ8U-drop0.001 (query)                            1_438.66        66.90     1_505.56       0.9323          1.0205            1.0036        40.05
HNSW-SQ8U-drop0.01 (query)                             1_438.04        65.08     1_503.12       0.9323          1.0404            1.0020        40.05
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>HNSW-SQ8U - Cosine (NN embeddings; more dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        51.08     1_279.06     1_330.14       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         51.08    12_915.99    12_967.07       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    87.21     1_221.06     1_308.27       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_161.12        64.82     1_225.94       0.9936          1.1067            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_161.12       114.27     1_275.39       0.9962          1.0402            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_161.12       198.83     1_359.95       0.9977          1.0167            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_161.12     1_075.14     2_236.26       0.9957          1.0491            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_172.42        71.98     2_244.40       0.9932          1.1481            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_172.42       120.48     2_292.90       0.9958          1.0811            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_172.42       220.45     2_392.87       0.9979          1.0358            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_172.42     1_145.16     3_317.58       0.9957          1.0878            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_446.06        71.54     2_517.60       0.9984          1.0178            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_446.06       121.10     2_567.17       0.9989          1.0140            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_446.06       217.45     2_663.52       0.9995          1.0069            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_446.06     1_220.10     3_666.16       0.9991          1.0117            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_403.78        70.88     2_474.66       0.9989          1.0168            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_403.78       131.83     2_535.61       0.9996          1.0044            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_403.78       216.16     2_619.93       0.9998          1.0013            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_403.78     1_180.40     3_584.17       0.9994          1.0084            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          777.58        36.92       814.51       0.6624          1.4420            1.1649        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         777.58        63.85       841.43       0.6647          1.3897            1.1631        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         777.58       121.54       899.12       0.6663          1.3656            1.1621        40.05
HNSW-SQ8U-M16-ef100 (self)                               777.58       598.22     1_375.80       0.6647          1.3898            1.1635        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_403.79        37.99     1_441.78       0.6624          1.4603            1.1647        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_403.79        66.30     1_470.09       0.6643          1.4115            1.1633        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_403.79       143.95     1_547.74       0.6660          1.3738            1.1622        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_403.79       617.03     2_020.82       0.6647          1.3999            1.1633        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_495.91        40.90     1_536.80       0.6665          1.3593            1.1619        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_495.91        67.07     1_562.98       0.6670          1.3529            1.1616        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_495.91       125.25     1_621.15       0.6673          1.3489            1.1614        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_495.91       655.05     2_150.95       0.6666          1.3609            1.1619        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_573.95        41.43     1_615.37       0.6664          1.3760            1.1619        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_573.95        71.37     1_645.32       0.6670          1.3586            1.1616        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_573.95       128.39     1_702.34       0.6672          1.3517            1.1614        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_573.95       657.80     2_231.74       0.6665          1.3685            1.1619        58.36
HNSW-SQ8U-drop0 (query)                                1_428.20        65.20     1_493.40       0.6191          1.5317            1.2372        40.05
HNSW-SQ8U-drop0.001 (query)                            1_408.90        64.15     1_473.04       0.6654          1.3910            1.1624        40.05
HNSW-SQ8U-drop0.01 (query)                             1_427.22        63.49     1_490.71       0.6779          1.3780            1.1508        40.05
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.54       683.26       715.80       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.54     2_250.35     2_282.89       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                306.72       646.84       953.56       0.2581          1.1827            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 306.72     2_123.88     2_430.59       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                                352.62     1_480.00     1_832.63       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                                 352.62     4_924.59     5_277.21       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                                425.99     3_556.55     3_982.54       0.3611          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                                 425.99    11_660.50    12_086.48       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                             420.17       196.99       617.16       0.3795          1.0927            1.0935         1.17
IVF-PQ-nl158-m16-np12 (query)                            420.17       309.03       729.21       0.3795          1.0927            1.0935         1.17
IVF-PQ-nl158-m16-np17 (query)                            420.17       427.01       847.19       0.3795          1.0927            1.0935         1.17
IVF-PQ-nl158-m16 (self)                                  420.17     1_395.73     1_815.90       0.3050          1.1270            1.1308         1.17
IVF-PQ-nl158-m32-np7 (query)                             465.59       358.12       823.71       0.4875          1.0597            1.0550         1.93
IVF-PQ-nl158-m32-np12 (query)                            465.59       564.92     1_030.51       0.4875          1.0597            1.0550         1.93
IVF-PQ-nl158-m32-np17 (query)                            465.59       772.03     1_237.61       0.4875          1.0597            1.0550         1.93
IVF-PQ-nl158-m32 (self)                                  465.59     2_550.35     3_015.94       0.4074          1.0813            1.0787         1.93
IVF-PQ-nl158-m64-np7 (query)                             546.37       633.83     1_180.20       0.6924          1.0213            1.0162         3.46
IVF-PQ-nl158-m64-np12 (query)                            546.37       998.98     1_545.35       0.6924          1.0213            1.0162         3.46
IVF-PQ-nl158-m64-np17 (query)                            546.37     1_373.21     1_919.58       0.6924          1.0213            1.0162         3.46
IVF-PQ-nl158-m64 (self)                                  546.37     4_518.35     5_064.72       0.6324          1.0288            1.0243         3.46
IVF-PQ-nl223-m16-np11 (query)                            443.57       292.05       735.62       0.3901          1.0871            1.0885         1.23
IVF-PQ-nl223-m16-np14 (query)                            443.57       354.61       798.18       0.3901          1.0871            1.0885         1.23
IVF-PQ-nl223-m16-np21 (query)                            443.57       516.35       959.92       0.3901          1.0871            1.0885         1.23
IVF-PQ-nl223-m16 (self)                                  443.57     1_727.27     2_170.84       0.3120          1.1216            1.1261         1.23
IVF-PQ-nl223-m32-np11 (query)                            472.23       518.49       990.72       0.4995          1.0556            1.0517         2.00
IVF-PQ-nl223-m32-np14 (query)                            472.23       641.60     1_113.83       0.4995          1.0556            1.0517         2.00
IVF-PQ-nl223-m32-np21 (query)                            472.23       933.65     1_405.88       0.4995          1.0556            1.0517         2.00
IVF-PQ-nl223-m32 (self)                                  472.23     3_101.06     3_573.28       0.4161          1.0769            1.0757         2.00
IVF-PQ-nl223-m64-np11 (query)                            555.17       935.68     1_490.85       0.7007          1.0188            1.0153         3.52
IVF-PQ-nl223-m64-np14 (query)                            555.17     1_357.34     1_912.51       0.7007          1.0188            1.0153         3.52
IVF-PQ-nl223-m64-np21 (query)                            555.17     1_784.08     2_339.24       0.7007          1.0188            1.0153         3.52
IVF-PQ-nl223-m64 (self)                                  555.17     5_753.19     6_308.35       0.6428          1.0259            1.0230         3.52
IVF-PQ-nl316-m16-np15 (query)                            542.55       386.14       928.68       0.4014          1.0814            1.0842         1.32
IVF-PQ-nl316-m16-np17 (query)                            542.55       420.64       963.19       0.4014          1.0814            1.0842         1.32
IVF-PQ-nl316-m16-np25 (query)                            542.55       648.60     1_191.15       0.4014          1.0814            1.0842         1.32
IVF-PQ-nl316-m16 (self)                                  542.55     1_982.91     2_525.46       0.3174          1.1174            1.1224         1.32
IVF-PQ-nl316-m32-np15 (query)                            520.91       676.49     1_197.40       0.5121          1.0503            1.0485         2.09
IVF-PQ-nl316-m32-np17 (query)                            520.91       755.81     1_276.71       0.5121          1.0503            1.0485         2.09
IVF-PQ-nl316-m32-np25 (query)                            520.91     1_080.52     1_601.43       0.5121          1.0503            1.0485         2.09
IVF-PQ-nl316-m32 (self)                                  520.91     3_573.46     4_094.37       0.4252          1.0725            1.0727         2.09
IVF-PQ-nl316-m64-np15 (query)                            591.36     1_176.89     1_768.25       0.7103          1.0164            1.0144         3.61
IVF-PQ-nl316-m64-np17 (query)                            591.36     1_321.15     1_912.51       0.7103          1.0164            1.0144         3.61
IVF-PQ-nl316-m64-np25 (query)                            591.36     1_881.45     2_472.81       0.7103          1.0164            1.0144         3.61
IVF-PQ-nl316-m64 (self)                                  591.36     6_229.41     6_820.77       0.6508          1.0241            1.0218         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.96     1_231.36     1_301.31       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.96     4_052.72     4_122.68       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                577.34       662.25     1_239.59       0.2443          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                                 577.34     2_140.05     2_717.38       0.2277          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                                600.37     1_504.45     2_104.82       0.2649          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                                 600.37     4_926.64     5_527.00       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                                683.01     3_840.02     4_523.03       0.2958          1.0991            1.1029         3.55
Exhaustive-PQ-m64 (self)                                 683.01    11_724.07    12_407.08       0.2627          1.1103            1.1143         3.55
IVF-PQ-nl158-m16-np7 (query)                             751.48       278.04     1_029.52       0.3109          1.0857            1.0894         1.57
IVF-PQ-nl158-m16-np12 (query)                            751.48       413.09     1_164.58       0.3109          1.0857            1.0894         1.57
IVF-PQ-nl158-m16-np17 (query)                            751.48       563.55     1_315.04       0.3109          1.0857            1.0894         1.57
IVF-PQ-nl158-m16 (self)                                  751.48     1_859.27     2_610.75       0.2636          1.1065            1.1127         1.57
IVF-PQ-nl158-m32-np7 (query)                             789.42       399.45     1_188.87       0.3609          1.0693            1.0692         2.34
IVF-PQ-nl158-m32-np12 (query)                            789.42       623.11     1_412.53       0.3609          1.0693            1.0692         2.34
IVF-PQ-nl158-m32-np17 (query)                            789.42       855.39     1_644.81       0.3609          1.0693            1.0692         2.34
IVF-PQ-nl158-m32 (self)                                  789.42     2_806.94     3_596.36       0.2932          1.0904            1.0937         2.34
IVF-PQ-nl158-m64-np7 (query)                             891.39       720.86     1_612.25       0.4661          1.0460            1.0408         3.86
IVF-PQ-nl158-m64-np12 (query)                            891.39     1_136.60     2_027.99       0.4661          1.0460            1.0408         3.86
IVF-PQ-nl158-m64-np17 (query)                            891.39     1_554.80     2_446.19       0.4661          1.0460            1.0408         3.86
IVF-PQ-nl158-m64 (self)                                  891.39     5_143.90     6_035.29       0.3914          1.0589            1.0567         3.86
IVF-PQ-nl223-m16-np11 (query)                            794.93       412.17     1_207.10       0.3240          1.0784            1.0829         1.70
IVF-PQ-nl223-m16-np14 (query)                            794.93       487.29     1_282.22       0.3240          1.0784            1.0829         1.70
IVF-PQ-nl223-m16-np21 (query)                            794.93       709.37     1_504.30       0.3240          1.0784            1.0829         1.70
IVF-PQ-nl223-m16 (self)                                  794.93     2_354.95     3_149.88       0.2682          1.1028            1.1098         1.70
IVF-PQ-nl223-m32-np11 (query)                            846.25       589.42     1_435.67       0.3733          1.0632            1.0645         2.46
IVF-PQ-nl223-m32-np14 (query)                            846.25       718.32     1_564.57       0.3733          1.0632            1.0645         2.46
IVF-PQ-nl223-m32-np21 (query)                            846.25     1_036.64     1_882.89       0.3733          1.0632            1.0645         2.46
IVF-PQ-nl223-m32 (self)                                  846.25     3_417.13     4_263.38       0.2958          1.0879            1.0916         2.46
IVF-PQ-nl223-m64-np11 (query)                            914.06     1_037.17     1_951.23       0.4834          1.0412            1.0372         3.99
IVF-PQ-nl223-m64-np14 (query)                            914.06     1_284.05     2_198.11       0.4834          1.0412            1.0372         3.99
IVF-PQ-nl223-m64-np21 (query)                            914.06     1_872.45     2_786.50       0.4834          1.0412            1.0372         3.99
IVF-PQ-nl223-m64 (self)                                  914.06     6_285.02     7_199.08       0.4000          1.0559            1.0544         3.99
IVF-PQ-nl316-m16-np15 (query)                            838.69       517.11     1_355.80       0.3302          1.0747            1.0799         1.88
IVF-PQ-nl316-m16-np17 (query)                            838.69       568.78     1_407.47       0.3302          1.0747            1.0799         1.88
IVF-PQ-nl316-m16-np25 (query)                            838.69       810.14     1_648.82       0.3302          1.0747            1.0799         1.88
IVF-PQ-nl316-m16 (self)                                  838.69     2_682.41     3_521.10       0.2716          1.0994            1.1064         1.88
IVF-PQ-nl316-m32-np15 (query)                            874.80       749.63     1_624.44       0.3833          1.0594            1.0614         2.65
IVF-PQ-nl316-m32-np17 (query)                            874.80       838.32     1_713.13       0.3833          1.0594            1.0614         2.65
IVF-PQ-nl316-m32-np25 (query)                            874.80     1_192.52     2_067.32       0.3833          1.0594            1.0614         2.65
IVF-PQ-nl316-m32 (self)                                  874.80     3_913.94     4_788.75       0.3010          1.0845            1.0888         2.65
IVF-PQ-nl316-m64-np15 (query)                            954.65     1_347.56     2_302.21       0.4950          1.0369            1.0355         4.17
IVF-PQ-nl316-m64-np17 (query)                            954.65     1_509.11     2_463.77       0.4950          1.0369            1.0355         4.17
IVF-PQ-nl316-m64-np25 (query)                            954.65     2_162.29     3_116.95       0.4950          1.0369            1.0355         4.17
IVF-PQ-nl316-m64 (self)                                  954.65     7_132.30     8_086.96       0.4089          1.0523            1.0526         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.50     1_740.53     1_842.03       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.50     5_801.12     5_902.62       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                835.58       674.44     1_510.02       0.2345          1.1095            1.1000         1.51
Exhaustive-PQ-m16 (self)                                 835.58     2_179.69     3_015.27       0.2206          1.1180            1.1048         1.51
Exhaustive-PQ-m32 (query)                                862.82     1_517.53     2_380.35       0.2567          1.0943            1.0974         2.28
Exhaustive-PQ-m32 (self)                                 862.82     5_001.21     5_864.03       0.2391          1.1012            1.1021         2.28
Exhaustive-PQ-m64 (query)                                921.80     3_559.12     4_480.91       0.2775          1.0855            1.0910         3.80
Exhaustive-PQ-m64 (self)                                 921.80    11_737.50    12_659.30       0.2515          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             1_092.56     7_848.37     8_940.92       0.3162          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              1_092.56    25_974.92    27_067.48       0.2755          1.0817            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_101.60       347.96     1_449.55       0.2870          1.0772            1.0818         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_101.60       532.20     1_633.80       0.2870          1.0772            1.0818         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_101.60       728.00     1_829.59       0.2870          1.0772            1.0818         1.98
IVF-PQ-nl158-m16 (self)                                1_101.60     2_405.12     3_506.71       0.2499          1.0929            1.0991         1.98
IVF-PQ-nl158-m32-np7 (query)                           1_143.96       510.05     1_654.01       0.3206          1.0654            1.0681         2.74
IVF-PQ-nl158-m32-np12 (query)                          1_143.96       813.31     1_957.27       0.3206          1.0654            1.0681         2.74
IVF-PQ-nl158-m32-np17 (query)                          1_143.96     1_110.86     2_254.82       0.3206          1.0654            1.0681         2.74
IVF-PQ-nl158-m32 (self)                                1_143.96     3_671.53     4_815.48       0.2635          1.0843            1.0893         2.74
IVF-PQ-nl158-m64-np7 (query)                           1_243.40       809.02     2_052.43       0.3802          1.0522            1.0496         4.27
IVF-PQ-nl158-m64-np12 (query)                          1_243.40     1_293.00     2_536.40       0.3802          1.0522            1.0496         4.27
IVF-PQ-nl158-m64-np17 (query)                          1_243.40     1_766.53     3_009.94       0.3802          1.0522            1.0496         4.27
IVF-PQ-nl158-m64 (self)                                1_243.40     5_851.69     7_095.09       0.3073          1.0675            1.0675         4.27
IVF-PQ-nl158-m128-np7 (query)                          1_405.13     1_629.18     3_034.31       0.5367          1.0280            1.0223         7.32
IVF-PQ-nl158-m128-np12 (query)                         1_405.13     2_575.13     3_980.26       0.5367          1.0280            1.0223         7.32
IVF-PQ-nl158-m128-np17 (query)                         1_405.13     3_535.86     4_940.99       0.5367          1.0280            1.0223         7.32
IVF-PQ-nl158-m128 (self)                               1_405.13    11_700.61    13_105.74       0.4650          1.0348            1.0315         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_136.34       501.01     1_637.35       0.2981          1.0713            1.0771         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_136.34       608.81     1_745.16       0.2981          1.0713            1.0771         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_136.34       882.72     2_019.07       0.2981          1.0713            1.0771         2.17
IVF-PQ-nl223-m16 (self)                                1_136.34     2_912.03     4_048.37       0.2560          1.0886            1.0956         2.17
IVF-PQ-nl223-m32-np11 (query)                          1_167.66       743.28     1_910.94       0.3312          1.0606            1.0639         2.93
IVF-PQ-nl223-m32-np14 (query)                          1_167.66       914.54     2_082.20       0.3312          1.0606            1.0639         2.93
IVF-PQ-nl223-m32-np21 (query)                          1_167.66     1_328.31     2_495.97       0.3312          1.0606            1.0639         2.93
IVF-PQ-nl223-m32 (self)                                1_167.66     4_360.49     5_528.15       0.2676          1.0816            1.0868         2.93
IVF-PQ-nl223-m64-np11 (query)                          1_251.15     1_165.75     2_416.90       0.3963          1.0464            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          1_251.15     1_441.98     2_693.13       0.3963          1.0464            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          1_251.15     2_228.62     3_479.77       0.3963          1.0464            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                1_251.15     6_868.85     8_120.00       0.3162          1.0635            1.0652         4.46
IVF-PQ-nl223-m128-np11 (query)                         1_404.28     2_343.40     3_747.68       0.5499          1.0236            1.0212         7.51
IVF-PQ-nl223-m128-np14 (query)                         1_404.28     2_902.70     4_306.97       0.5499          1.0236            1.0212         7.51
IVF-PQ-nl223-m128-np21 (query)                         1_404.28     4_245.38     5_649.66       0.5499          1.0236            1.0212         7.51
IVF-PQ-nl223-m128 (self)                               1_404.28    13_994.75    15_399.02       0.4736          1.0318            1.0306         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_201.78       646.01     1_847.80       0.3053          1.0677            1.0734         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_201.78       713.33     1_915.11       0.3053          1.0677            1.0734         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_201.78     1_018.73     2_220.51       0.3053          1.0677            1.0734         2.44
IVF-PQ-nl316-m16 (self)                                1_201.78     3_400.01     4_601.79       0.2586          1.0861            1.0927         2.44
IVF-PQ-nl316-m32-np15 (query)                          1_236.81       975.89     2_212.70       0.3393          1.0573            1.0609         3.20
IVF-PQ-nl316-m32-np17 (query)                          1_236.81     1_088.35     2_325.16       0.3393          1.0573            1.0609         3.20
IVF-PQ-nl316-m32-np25 (query)                          1_236.81     1_560.62     2_797.42       0.3393          1.0573            1.0609         3.20
IVF-PQ-nl316-m32 (self)                                1_236.81     5_174.10     6_410.91       0.2711          1.0790            1.0845         3.20
IVF-PQ-nl316-m64-np15 (query)                          1_349.21     1_528.61     2_877.82       0.4084          1.0427            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          1_349.21     1_708.56     3_057.78       0.4084          1.0427            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          1_349.21     2_448.95     3_798.17       0.4084          1.0427            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                1_349.21     8_084.86     9_434.07       0.3237          1.0605            1.0631         4.73
IVF-PQ-nl316-m128-np15 (query)                         1_506.84     3_076.20     4_583.04       0.5612          1.0212            1.0201         7.78
IVF-PQ-nl316-m128-np17 (query)                         1_506.84     3_441.49     4_948.32       0.5612          1.0212            1.0201         7.78
IVF-PQ-nl316-m128-np25 (query)                         1_506.84     4_921.36     6_428.19       0.5612          1.0212            1.0201         7.78
IVF-PQ-nl316-m128 (self)                               1_506.84    16_290.60    17_797.43       0.4840          1.0301            1.0294         7.78
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Data where the structure resides on a lower-dimensional manifold.

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.66       689.21       721.87       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.66     2_254.17     2_286.83       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                308.01       647.09       955.10       0.2932          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 308.01     2_141.32     2_449.32       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                                350.76     1_486.57     1_837.33       0.4008          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                                 350.76     4_946.52     5_297.28       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                                440.53     3_518.77     3_959.30       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                                 440.53    11_696.44    12_136.98       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                             415.38       195.26       610.64       0.5349          1.0887            1.0858         1.17
IVF-PQ-nl158-m16-np12 (query)                            415.38       304.98       720.36       0.5349          1.0887            1.0858         1.17
IVF-PQ-nl158-m16-np17 (query)                            415.38       419.64       835.02       0.5349          1.0887            1.0858         1.17
IVF-PQ-nl158-m16 (self)                                  415.38     1_384.78     1_800.17       0.4272          1.1661            1.1615         1.17
IVF-PQ-nl158-m32-np7 (query)                             464.39       352.09       816.49       0.6756          1.0396            1.0372         1.93
IVF-PQ-nl158-m32-np12 (query)                            464.39       565.12     1_029.51       0.6756          1.0396            1.0372         1.93
IVF-PQ-nl158-m32-np17 (query)                            464.39       791.38     1_255.78       0.6756          1.0396            1.0372         1.93
IVF-PQ-nl158-m32 (self)                                  464.39     2_602.18     3_066.57       0.6054          1.0694            1.0645         1.93
IVF-PQ-nl158-m64-np7 (query)                             548.59       620.26     1_168.85       0.8339          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                            548.59       991.05     1_539.64       0.8339          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                            548.59     1_395.14     1_943.72       0.8339          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                  548.59     4_613.99     5_162.58       0.7984          1.0164            1.0142         3.46
IVF-PQ-nl223-m16-np11 (query)                            428.72       284.64       713.36       0.5365          1.0877            1.0844         1.23
IVF-PQ-nl223-m16-np14 (query)                            428.72       358.82       787.54       0.5365          1.0877            1.0844         1.23
IVF-PQ-nl223-m16-np21 (query)                            428.72       514.76       943.48       0.5365          1.0877            1.0844         1.23
IVF-PQ-nl223-m16 (self)                                  428.72     1_695.41     2_124.13       0.4228          1.1688            1.1643         1.23
IVF-PQ-nl223-m32-np11 (query)                            476.28       512.92       989.20       0.6763          1.0394            1.0371         2.00
IVF-PQ-nl223-m32-np14 (query)                            476.28       638.42     1_114.69       0.6763          1.0394            1.0371         2.00
IVF-PQ-nl223-m32-np21 (query)                            476.28       936.91     1_413.19       0.6763          1.0394            1.0371         2.00
IVF-PQ-nl223-m32 (self)                                  476.28     3_095.88     3_572.16       0.6025          1.0704            1.0658         2.00
IVF-PQ-nl223-m64-np11 (query)                            558.25       894.59     1_452.83       0.8366          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                            558.25     1_120.46     1_678.71       0.8366          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                            558.25     1_645.34     2_203.59       0.8366          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                  558.25     5_455.54     6_013.78       0.8009          1.0158            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                            452.18       375.21       827.39       0.5357          1.0879            1.0852         1.32
IVF-PQ-nl316-m16-np17 (query)                            452.18       419.21       871.38       0.5357          1.0879            1.0852         1.32
IVF-PQ-nl316-m16-np25 (query)                            452.18       596.58     1_048.76       0.5357          1.0879            1.0852         1.32
IVF-PQ-nl316-m16 (self)                                  452.18     1_977.68     2_429.85       0.4167          1.1731            1.1687         1.32
IVF-PQ-nl316-m32-np15 (query)                            497.78       678.31     1_176.08       0.6783          1.0386            1.0366         2.09
IVF-PQ-nl316-m32-np17 (query)                            497.78       773.51     1_271.29       0.6783          1.0386            1.0366         2.09
IVF-PQ-nl316-m32-np25 (query)                            497.78     1_079.34     1_577.12       0.6783          1.0386            1.0366         2.09
IVF-PQ-nl316-m32 (self)                                  497.78     3_574.18     4_071.96       0.6014          1.0706            1.0662         2.09
IVF-PQ-nl316-m64-np15 (query)                            582.68     1_157.74     1_740.42       0.8387          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                            582.68     1_299.51     1_882.20       0.8387          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                            582.68     1_890.60     2_473.28       0.8387          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                  582.68     6_687.53     7_270.22       0.8029          1.0154            1.0136         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.70     1_228.21     1_298.91       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.70     4_014.94     4_085.64       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                559.71       661.89     1_221.60       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 559.71     2_156.02     2_715.73       0.1772          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                                609.27     1_504.22     2_113.50       0.2801          1.1736            1.1700         2.03
Exhaustive-PQ-m32 (self)                                 609.27     4_932.57     5_541.84       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                                680.73     3_535.67     4_216.40       0.3752          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                                 680.73    11_731.03    12_411.76       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                             736.51       259.60       996.11       0.3790          1.1180            1.1177         1.57
IVF-PQ-nl158-m16-np12 (query)                            736.51       403.14     1_139.65       0.3790          1.1180            1.1177         1.57
IVF-PQ-nl158-m16-np17 (query)                            736.51       547.01     1_283.51       0.3790          1.1180            1.1177         1.57
IVF-PQ-nl158-m16 (self)                                  736.51     1_777.35     2_513.86       0.2720          1.2091            1.2127         1.57
IVF-PQ-nl158-m32-np7 (query)                             780.67       391.63     1_172.30       0.4913          1.0723            1.0709         2.34
IVF-PQ-nl158-m32-np12 (query)                            780.67       654.65     1_435.32       0.4913          1.0723            1.0709         2.34
IVF-PQ-nl158-m32-np17 (query)                            780.67       840.48     1_621.15       0.4913          1.0723            1.0709         2.34
IVF-PQ-nl158-m32 (self)                                  780.67     2_804.79     3_585.45       0.3909          1.1262            1.1249         2.34
IVF-PQ-nl158-m64-np7 (query)                             875.79       713.45     1_589.25       0.6315          1.0349            1.0333         3.86
IVF-PQ-nl158-m64-np12 (query)                            875.79     1_126.29     2_002.08       0.6315          1.0349            1.0333         3.86
IVF-PQ-nl158-m64-np17 (query)                            875.79     1_542.09     2_417.89       0.6315          1.0349            1.0333         3.86
IVF-PQ-nl158-m64 (self)                                  875.79     5_089.45     5_965.25       0.5743          1.0544            1.0503         3.86
IVF-PQ-nl223-m16-np11 (query)                            776.78       388.94     1_165.72       0.3805          1.1171            1.1170         1.70
IVF-PQ-nl223-m16-np14 (query)                            776.78       472.91     1_249.69       0.3805          1.1171            1.1170         1.70
IVF-PQ-nl223-m16-np21 (query)                            776.78       687.00     1_463.78       0.3805          1.1171            1.1170         1.70
IVF-PQ-nl223-m16 (self)                                  776.78     2_336.22     3_113.00       0.2700          1.2105            1.2145         1.70
IVF-PQ-nl223-m32-np11 (query)                            817.10       578.03     1_395.13       0.4908          1.0720            1.0705         2.46
IVF-PQ-nl223-m32-np14 (query)                            817.10       705.80     1_522.91       0.4908          1.0720            1.0705         2.46
IVF-PQ-nl223-m32-np21 (query)                            817.10     1_026.11     1_843.21       0.4908          1.0720            1.0705         2.46
IVF-PQ-nl223-m32 (self)                                  817.10     3_391.27     4_208.37       0.3857          1.1288            1.1269         2.46
IVF-PQ-nl223-m64-np11 (query)                            906.29     1_027.75     1_934.04       0.6333          1.0343            1.0327         3.99
IVF-PQ-nl223-m64-np14 (query)                            906.29     1_273.59     2_179.88       0.6333          1.0343            1.0327         3.99
IVF-PQ-nl223-m64-np21 (query)                            906.29     1_847.86     2_754.15       0.6333          1.0343            1.0327         3.99
IVF-PQ-nl223-m64 (self)                                  906.29     6_115.57     7_021.87       0.5729          1.0547            1.0507         3.99
IVF-PQ-nl316-m16-np15 (query)                            850.51       524.84     1_375.34       0.3767          1.1183            1.1183         1.88
IVF-PQ-nl316-m16-np17 (query)                            850.51       579.26     1_429.77       0.3767          1.1183            1.1183         1.88
IVF-PQ-nl316-m16-np25 (query)                            850.51       823.89     1_674.40       0.3767          1.1183            1.1183         1.88
IVF-PQ-nl316-m16 (self)                                  850.51     2_724.04     3_574.55       0.2616          1.2171            1.2216         1.88
IVF-PQ-nl316-m32-np15 (query)                            868.92       751.70     1_620.62       0.4874          1.0732            1.0720         2.65
IVF-PQ-nl316-m32-np17 (query)                            868.92       899.09     1_768.01       0.4874          1.0732            1.0720         2.65
IVF-PQ-nl316-m32-np25 (query)                            868.92     1_193.72     2_062.64       0.4874          1.0732            1.0720         2.65
IVF-PQ-nl316-m32 (self)                                  868.92     3_926.07     4_795.00       0.3754          1.1341            1.1326         2.65
IVF-PQ-nl316-m64-np15 (query)                            956.31     1_338.46     2_294.77       0.6347          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np17 (query)                            956.31     1_499.73     2_456.03       0.6347          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np25 (query)                            956.31     2_150.60     3_106.90       0.6347          1.0339            1.0324         4.17
IVF-PQ-nl316-m64 (self)                                  956.31     7_097.79     8_054.10       0.5696          1.0555            1.0517         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.81     1_768.18     1_870.99       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.81     5_830.54     5_933.35       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                849.44       675.00     1_524.44       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                                 849.44     2_179.73     3_029.17       0.1758          1.3086            1.3089         1.51
Exhaustive-PQ-m32 (query)                                865.58     1_518.70     2_384.28       0.2712          1.1686            1.1636         2.28
Exhaustive-PQ-m32 (self)                                 865.58     5_051.42     5_917.00       0.2191          1.2527            1.2506         2.28
Exhaustive-PQ-m64 (query)                                948.89     3_544.58     4_493.46       0.3546          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                                 948.89    11_788.12    12_737.00       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             1_104.70     7_837.44     8_942.14       0.4597          1.0781            1.0752         6.86
Exhaustive-PQ-m128 (self)                              1_104.70    25_991.47    27_096.17       0.3908          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_108.39       350.31     1_458.70       0.3623          1.1176            1.1166         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_108.39       537.28     1_645.67       0.3623          1.1176            1.1166         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_108.39       742.24     1_850.63       0.3623          1.1176            1.1166         1.98
IVF-PQ-nl158-m16 (self)                                1_108.39     2_467.83     3_576.22       0.2562          1.2222            1.2264         1.98
IVF-PQ-nl158-m32-np7 (query)                           1_145.20       509.61     1_654.81       0.4635          1.0762            1.0746         2.74
IVF-PQ-nl158-m32-np12 (query)                          1_145.20       811.54     1_956.74       0.4635          1.0762            1.0746         2.74
IVF-PQ-nl158-m32-np17 (query)                          1_145.20     1_122.96     2_268.16       0.4635          1.0762            1.0746         2.74
IVF-PQ-nl158-m32 (self)                                1_145.20     3_741.50     4_886.70       0.3615          1.1421            1.1411         2.74
IVF-PQ-nl158-m64-np7 (query)                           1_248.30       804.53     2_052.83       0.5756          1.0442            1.0427         4.27
IVF-PQ-nl158-m64-np12 (query)                          1_248.30     1_288.92     2_537.22       0.5756          1.0442            1.0427         4.27
IVF-PQ-nl158-m64-np17 (query)                          1_248.30     1_804.22     3_052.52       0.5756          1.0442            1.0427         4.27
IVF-PQ-nl158-m64 (self)                                1_248.30     6_437.03     7_685.33       0.5122          1.0734            1.0691         4.27
IVF-PQ-nl158-m128-np7 (query)                          1_644.53     1_684.39     3_328.92       0.7393          1.0157            1.0139         7.32
IVF-PQ-nl158-m128-np12 (query)                         1_644.53     2_584.87     4_229.41       0.7393          1.0157            1.0139         7.32
IVF-PQ-nl158-m128-np17 (query)                         1_644.53     3_559.12     5_203.65       0.7393          1.0157            1.0139         7.32
IVF-PQ-nl158-m128 (self)                               1_644.53    11_732.91    13_377.44       0.7137          1.0239            1.0196         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_134.91       499.80     1_634.70       0.3623          1.1170            1.1169         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_134.91       604.13     1_739.04       0.3623          1.1170            1.1169         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_134.91       885.17     2_020.07       0.3623          1.1170            1.1169         2.17
IVF-PQ-nl223-m16 (self)                                1_134.91     2_895.78     4_030.69       0.2535          1.2252            1.2304         2.17
IVF-PQ-nl223-m32-np11 (query)                          1_159.10       739.06     1_898.16       0.4608          1.0765            1.0751         2.93
IVF-PQ-nl223-m32-np14 (query)                          1_159.10       910.78     2_069.88       0.4608          1.0765            1.0751         2.93
IVF-PQ-nl223-m32-np21 (query)                          1_159.10     1_322.89     2_481.99       0.4608          1.0765            1.0751         2.93
IVF-PQ-nl223-m32 (self)                                1_159.10     4_363.54     5_522.63       0.3520          1.1464            1.1458         2.93
IVF-PQ-nl223-m64-np11 (query)                          1_242.73     1_159.64     2_402.37       0.5740          1.0445            1.0427         4.46
IVF-PQ-nl223-m64-np14 (query)                          1_242.73     1_445.84     2_688.57       0.5740          1.0445            1.0427         4.46
IVF-PQ-nl223-m64-np21 (query)                          1_242.73     2_087.74     3_330.47       0.5740          1.0445            1.0427         4.46
IVF-PQ-nl223-m64 (self)                                1_242.73     6_911.10     8_153.83       0.5029          1.0763            1.0720         4.46
IVF-PQ-nl223-m128-np11 (query)                         1_399.53     2_357.15     3_756.68       0.7396          1.0156            1.0140         7.51
IVF-PQ-nl223-m128-np14 (query)                         1_399.53     2_899.53     4_299.06       0.7396          1.0156            1.0140         7.51
IVF-PQ-nl223-m128-np21 (query)                         1_399.53     4_223.57     5_623.10       0.7396          1.0156            1.0140         7.51
IVF-PQ-nl223-m128 (self)                               1_399.53    14_022.13    15_421.66       0.7120          1.0240            1.0200         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_195.89       643.81     1_839.70       0.3583          1.1192            1.1195         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_195.89       733.45     1_929.34       0.3583          1.1192            1.1195         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_195.89     1_010.13     2_206.02       0.3583          1.1192            1.1195         2.44
IVF-PQ-nl316-m16 (self)                                1_195.89     3_355.28     4_551.17       0.2455          1.2315            1.2371         2.44
IVF-PQ-nl316-m32-np15 (query)                          1_232.70       972.08     2_204.78       0.4537          1.0788            1.0776         3.20
IVF-PQ-nl316-m32-np17 (query)                          1_232.70     1_087.70     2_320.40       0.4537          1.0788            1.0776         3.20
IVF-PQ-nl316-m32-np25 (query)                          1_232.70     1_554.70     2_787.40       0.4537          1.0788            1.0776         3.20
IVF-PQ-nl316-m32 (self)                                1_232.70     5_172.28     6_404.98       0.3334          1.1569            1.1576         3.20
IVF-PQ-nl316-m64-np15 (query)                          1_353.39     1_523.88     2_877.27       0.5722          1.0450            1.0435         4.73
IVF-PQ-nl316-m64-np17 (query)                          1_353.39     1_706.91     3_060.30       0.5722          1.0450            1.0435         4.73
IVF-PQ-nl316-m64-np25 (query)                          1_353.39     2_451.65     3_805.05       0.5722          1.0450            1.0435         4.73
IVF-PQ-nl316-m64 (self)                                1_353.39     8_134.34     9_487.74       0.4877          1.0813            1.0776         4.73
IVF-PQ-nl316-m128-np15 (query)                         1_498.94     3_053.49     4_552.42       0.7406          1.0155            1.0139         7.78
IVF-PQ-nl316-m128-np17 (query)                         1_498.94     3_415.00     4_913.94       0.7406          1.0155            1.0139         7.78
IVF-PQ-nl316-m128-np25 (query)                         1_498.94     4_975.16     6_474.10       0.7406          1.0155            1.0139         7.78
IVF-PQ-nl316-m128 (self)                               1_498.94    16_326.73    17_825.67       0.7100          1.0240            1.0206         7.78
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.13       689.41       722.54       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.13     2_267.24     2_300.37       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                308.16       646.15       954.31       0.7118          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 308.16     2_139.39     2_447.56       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                                369.43     1_602.91     1_972.34       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                                 369.43     4_934.13     5_303.55       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                                427.74     3_514.98     3_942.72       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                                 427.74    11_691.84    12_119.57       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                             421.24       202.57       623.81       0.8335          1.0486            1.0415         1.17
IVF-PQ-nl158-m16-np12 (query)                            421.24       329.02       750.26       0.8341          1.0482            1.0411         1.17
IVF-PQ-nl158-m16-np17 (query)                            421.24       471.72       892.97       0.8341          1.0482            1.0411         1.17
IVF-PQ-nl158-m16 (self)                                  421.24     1_516.56     1_937.80       0.7729          1.0936            1.0788         1.17
IVF-PQ-nl158-m32-np7 (query)                             464.28       378.74       843.01       0.8776          1.0251            1.0205         1.93
IVF-PQ-nl158-m32-np12 (query)                            464.28       628.28     1_092.56       0.8785          1.0246            1.0202         1.93
IVF-PQ-nl158-m32-np17 (query)                            464.28       882.04     1_346.32       0.8785          1.0246            1.0202         1.93
IVF-PQ-nl158-m32 (self)                                  464.28     2_907.98     3_372.26       0.8331          1.0481            1.0397         1.93
IVF-PQ-nl158-m64-np7 (query)                             541.50       674.50     1_215.99       0.9059          1.0148            1.0115         3.46
IVF-PQ-nl158-m64-np12 (query)                            541.50     1_134.66     1_676.16       0.9070          1.0143            1.0112         3.46
IVF-PQ-nl158-m64-np17 (query)                            541.50     1_591.57     2_133.07       0.9070          1.0143            1.0112         3.46
IVF-PQ-nl158-m64 (self)                                  541.50     5_288.62     5_830.12       0.8726          1.0277            1.0217         3.46
IVF-PQ-nl223-m16-np11 (query)                            433.55       298.03       731.57       0.8437          1.0424            1.0363         1.23
IVF-PQ-nl223-m16-np14 (query)                            433.55       368.83       802.38       0.8438          1.0424            1.0363         1.23
IVF-PQ-nl223-m16-np21 (query)                            433.55       545.41       978.95       0.8438          1.0424            1.0363         1.23
IVF-PQ-nl223-m16 (self)                                  433.55     1_795.33     2_228.87       0.7859          1.0824            1.0687         1.23
IVF-PQ-nl223-m32-np11 (query)                            466.69       549.61     1_016.30       0.8841          1.0224            1.0184         2.00
IVF-PQ-nl223-m32-np14 (query)                            466.69       680.31     1_147.00       0.8842          1.0224            1.0183         2.00
IVF-PQ-nl223-m32-np21 (query)                            466.69     1_023.75     1_490.44       0.8842          1.0224            1.0183         2.00
IVF-PQ-nl223-m32 (self)                                  466.69     3_334.33     3_801.02       0.8399          1.0442            1.0358         2.00
IVF-PQ-nl223-m64-np11 (query)                            550.64       956.12     1_506.76       0.9100          1.0133            1.0101         3.52
IVF-PQ-nl223-m64-np14 (query)                            550.64     1_214.30     1_764.94       0.9102          1.0132            1.0101         3.52
IVF-PQ-nl223-m64-np21 (query)                            550.64     1_796.99     2_347.63       0.9102          1.0132            1.0101         3.52
IVF-PQ-nl223-m64 (self)                                  550.64     6_010.01     6_560.65       0.8764          1.0258            1.0201         3.52
IVF-PQ-nl316-m16-np15 (query)                            459.34       383.34       842.68       0.8515          1.0384            1.0325         1.32
IVF-PQ-nl316-m16-np17 (query)                            459.34       426.58       885.92       0.8516          1.0384            1.0325         1.32
IVF-PQ-nl316-m16-np25 (query)                            459.34       616.21     1_075.55       0.8516          1.0384            1.0324         1.32
IVF-PQ-nl316-m16 (self)                                  459.34     2_043.26     2_502.60       0.7932          1.0774            1.0628         1.32
IVF-PQ-nl316-m32-np15 (query)                            505.09       691.00     1_196.09       0.8886          1.0207            1.0168         2.09
IVF-PQ-nl316-m32-np17 (query)                            505.09       779.36     1_284.45       0.8886          1.0207            1.0168         2.09
IVF-PQ-nl316-m32-np25 (query)                            505.09     1_132.14     1_637.23       0.8887          1.0207            1.0168         2.09
IVF-PQ-nl316-m32 (self)                                  505.09     3_751.92     4_257.00       0.8445          1.0420            1.0330         2.09
IVF-PQ-nl316-m64-np15 (query)                            572.45     1_206.58     1_779.03       0.9133          1.0124            1.0096         3.61
IVF-PQ-nl316-m64-np17 (query)                            572.45     1_367.52     1_939.97       0.9133          1.0124            1.0096         3.61
IVF-PQ-nl316-m64-np25 (query)                            572.45     2_002.45     2_574.90       0.9134          1.0124            1.0096         3.61
IVF-PQ-nl316-m64 (self)                                  572.45     6_617.89     7_190.34       0.8800          1.0242            1.0186         3.61
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.84     1_227.01     1_295.85       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.84     4_033.61     4_102.45       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                568.48       664.52     1_233.00       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 568.48     2_152.32     2_720.80       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                                604.07     1_497.38     2_101.45       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                                 604.07     5_009.24     5_613.31       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                                682.43     3_530.78     4_213.21       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                                 682.43    11_689.74    12_372.18       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                             728.90       269.73       998.63       0.8452          1.0451            1.0357         1.57
IVF-PQ-nl158-m16-np12 (query)                            728.90       429.54     1_158.45       0.8456          1.0450            1.0356         1.57
IVF-PQ-nl158-m16-np17 (query)                            728.90       614.78     1_343.68       0.8456          1.0450            1.0356         1.57
IVF-PQ-nl158-m16 (self)                                  728.90     1_930.48     2_659.38       0.7847          1.0913            1.0644         1.57
IVF-PQ-nl158-m32-np7 (query)                             799.22       418.06     1_217.27       0.8722          1.0301            1.0236         2.34
IVF-PQ-nl158-m32-np12 (query)                            799.22       687.63     1_486.84       0.8725          1.0300            1.0235         2.34
IVF-PQ-nl158-m32-np17 (query)                            799.22       954.01     1_753.23       0.8725          1.0300            1.0235         2.34
IVF-PQ-nl158-m32 (self)                                  799.22     3_103.20     3_902.42       0.8197          1.0621            1.0435         2.34
IVF-PQ-nl158-m64-np7 (query)                             869.58       767.44     1_637.02       0.8927          1.0210            1.0155         3.86
IVF-PQ-nl158-m64-np12 (query)                            869.58     1_281.97     2_151.55       0.8931          1.0208            1.0153         3.86
IVF-PQ-nl158-m64-np17 (query)                            869.58     1_790.95     2_660.52       0.8931          1.0208            1.0153         3.86
IVF-PQ-nl158-m64 (self)                                  869.58     5_893.21     6_762.79       0.8473          1.0436            1.0300         3.86
IVF-PQ-nl223-m16-np11 (query)                            783.19       407.42     1_190.61       0.8603          1.0367            1.0290         1.70
IVF-PQ-nl223-m16-np14 (query)                            783.19       495.24     1_278.43       0.8603          1.0367            1.0289         1.70
IVF-PQ-nl223-m16-np21 (query)                            783.19       730.09     1_513.28       0.8603          1.0367            1.0289         1.70
IVF-PQ-nl223-m16 (self)                                  783.19     2_454.45     3_237.64       0.8010          1.0770            1.0539         1.70
IVF-PQ-nl223-m32-np11 (query)                            836.29       607.60     1_443.89       0.8835          1.0251            1.0189         2.46
IVF-PQ-nl223-m32-np14 (query)                            836.29       757.99     1_594.27       0.8836          1.0251            1.0188         2.46
IVF-PQ-nl223-m32-np21 (query)                            836.29     1_111.25     1_947.54       0.8836          1.0251            1.0188         2.46
IVF-PQ-nl223-m32 (self)                                  836.29     3_687.72     4_524.01       0.8325          1.0539            1.0361         2.46
IVF-PQ-nl223-m64-np11 (query)                            918.69     1_097.84     2_016.52       0.9025          1.0170            1.0127         3.99
IVF-PQ-nl223-m64-np14 (query)                            918.69     1_381.55     2_300.24       0.9026          1.0170            1.0126         3.99
IVF-PQ-nl223-m64-np21 (query)                            918.69     2_042.99     2_961.68       0.9026          1.0170            1.0126         3.99
IVF-PQ-nl223-m64 (self)                                  918.69     6_743.05     7_661.74       0.8592          1.0367            1.0244         3.99
IVF-PQ-nl316-m16-np15 (query)                            837.41       532.13     1_369.54       0.8711          1.0314            1.0245         1.88
IVF-PQ-nl316-m16-np17 (query)                            837.41       587.94     1_425.35       0.8711          1.0314            1.0245         1.88
IVF-PQ-nl316-m16-np25 (query)                            837.41       844.42     1_681.84       0.8711          1.0314            1.0245         1.88
IVF-PQ-nl316-m16 (self)                                  837.41     2_778.21     3_615.63       0.8168          1.0641            1.0447         1.88
IVF-PQ-nl316-m32-np15 (query)                            878.15       776.97     1_655.11       0.8924          1.0213            1.0158         2.65
IVF-PQ-nl316-m32-np17 (query)                            878.15       869.47     1_747.62       0.8924          1.0213            1.0158         2.65
IVF-PQ-nl316-m32-np25 (query)                            878.15     1_315.16     2_193.31       0.8924          1.0213            1.0158         2.65
IVF-PQ-nl316-m32 (self)                                  878.15     4_120.24     4_998.38       0.8456          1.0448            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                            961.19     1_398.46     2_359.66       0.9077          1.0150            1.0109         4.17
IVF-PQ-nl316-m64-np17 (query)                            961.19     1_577.69     2_538.89       0.9077          1.0150            1.0109         4.17
IVF-PQ-nl316-m64-np25 (query)                            961.19     2_292.37     3_253.57       0.9077          1.0150            1.0108         4.17
IVF-PQ-nl316-m64 (self)                                  961.19     7_569.90     8_531.10       0.8670          1.0322            1.0213         4.17
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.83     1_753.26     1_855.09       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.83     5_800.38     5_902.21       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                                835.62       683.66     1_519.28       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                                 835.62     2_235.75     3_071.37       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                                876.21     1_517.16     2_393.36       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                                 876.21     4_969.19     5_845.39       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                                933.10     3_599.29     4_532.39       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                                 933.10    11_764.63    12_697.73       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             1_099.12     7_807.65     8_906.77       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              1_099.12    25_893.08    26_992.21       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_112.36       360.78     1_473.14       0.8545          1.0410            1.0324         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_112.36       567.97     1_680.33       0.8545          1.0410            1.0323         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_112.36       787.74     1_900.10       0.8545          1.0410            1.0323         1.98
IVF-PQ-nl158-m16 (self)                                1_112.36     2_623.49     3_735.85       0.7949          1.0798            1.0583         1.98
IVF-PQ-nl158-m32-np7 (query)                           1_146.98       540.48     1_687.46       0.9035          1.0193            1.0118         2.74
IVF-PQ-nl158-m32-np12 (query)                          1_146.98       886.14     2_033.12       0.9036          1.0192            1.0118         2.74
IVF-PQ-nl158-m32-np17 (query)                          1_146.98     1_237.91     2_384.89       0.9036          1.0192            1.0118         2.74
IVF-PQ-nl158-m32 (self)                                1_146.98     4_106.92     5_253.90       0.8603          1.0400            1.0217         2.74
IVF-PQ-nl158-m64-np7 (query)                           1_257.25       862.97     2_120.22       0.9235          1.0121            1.0064         4.27
IVF-PQ-nl158-m64-np12 (query)                          1_257.25     1_436.76     2_694.01       0.9236          1.0120            1.0064         4.27
IVF-PQ-nl158-m64-np17 (query)                          1_257.25     2_006.64     3_263.89       0.9236          1.0120            1.0064         4.27
IVF-PQ-nl158-m64 (self)                                1_257.25     6_630.96     7_888.21       0.8868          1.0265            1.0124         4.27
IVF-PQ-nl158-m128-np7 (query)                          1_406.57     1_720.04     3_126.61       0.9410          1.0066            1.0028         7.32
IVF-PQ-nl158-m128-np12 (query)                         1_406.57     2_902.49     4_309.06       0.9412          1.0066            1.0028         7.32
IVF-PQ-nl158-m128-np17 (query)                         1_406.57     4_068.19     5_474.76       0.9412          1.0066            1.0028         7.32
IVF-PQ-nl158-m128 (self)                               1_406.57    13_446.96    14_853.53       0.9098          1.0161            1.0065         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_134.31       540.03     1_674.34       0.8629          1.0361            1.0286         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_134.31       636.23     1_770.54       0.8629          1.0361            1.0286         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_134.31       929.16     2_063.47       0.8629          1.0361            1.0286         2.17
IVF-PQ-nl223-m16 (self)                                1_134.31     3_080.50     4_214.81       0.8071          1.0694            1.0515         2.17
IVF-PQ-nl223-m32-np11 (query)                          1_157.57       780.13     1_937.70       0.9087          1.0175            1.0106         2.93
IVF-PQ-nl223-m32-np14 (query)                          1_157.57       972.85     2_130.42       0.9087          1.0175            1.0106         2.93
IVF-PQ-nl223-m32-np21 (query)                          1_157.57     1_435.48     2_593.05       0.9087          1.0175            1.0106         2.93
IVF-PQ-nl223-m32 (self)                                1_157.57     4_787.24     5_944.81       0.8679          1.0350            1.0196         2.93
IVF-PQ-nl223-m64-np11 (query)                          1_263.90     1_231.55     2_495.45       0.9273          1.0109            1.0055         4.46
IVF-PQ-nl223-m64-np14 (query)                          1_263.90     1_550.79     2_814.69       0.9273          1.0109            1.0055         4.46
IVF-PQ-nl223-m64-np21 (query)                          1_263.90     2_289.92     3_553.82       0.9273          1.0109            1.0055         4.46
IVF-PQ-nl223-m64 (self)                                1_263.90     7_550.36     8_814.26       0.8928          1.0233            1.0110         4.46
IVF-PQ-nl223-m128-np11 (query)                         1_408.71     2_472.84     3_881.55       0.9442          1.0059            1.0024         7.51
IVF-PQ-nl223-m128-np14 (query)                         1_408.71     3_121.48     4_530.19       0.9443          1.0059            1.0024         7.51
IVF-PQ-nl223-m128-np21 (query)                         1_408.71     4_653.97     6_062.68       0.9443          1.0059            1.0024         7.51
IVF-PQ-nl223-m128 (self)                               1_408.71    15_446.37    16_855.08       0.9138          1.0145            1.0057         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_202.24       660.25     1_862.48       0.8688          1.0324            1.0254         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_202.24       731.34     1_933.58       0.8688          1.0324            1.0254         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_202.24     1_054.74     2_256.98       0.8688          1.0324            1.0254         2.44
IVF-PQ-nl316-m16 (self)                                1_202.24     3_494.39     4_696.62       0.8156          1.0629            1.0467         2.44
IVF-PQ-nl316-m32-np15 (query)                          1_245.45     1_004.79     2_250.23       0.9116          1.0160            1.0097         3.20
IVF-PQ-nl316-m32-np17 (query)                          1_245.45     1_127.07     2_372.52       0.9116          1.0160            1.0096         3.20
IVF-PQ-nl316-m32-np25 (query)                          1_245.45     1_632.22     2_877.67       0.9116          1.0160            1.0096         3.20
IVF-PQ-nl316-m32 (self)                                1_245.45     5_440.63     6_686.08       0.8732          1.0323            1.0180         3.20
IVF-PQ-nl316-m64-np15 (query)                          1_360.94     1_588.80     2_949.74       0.9296          1.0100            1.0049         4.73
IVF-PQ-nl316-m64-np17 (query)                          1_360.94     1_786.18     3_147.12       0.9296          1.0100            1.0049         4.73
IVF-PQ-nl316-m64-np25 (query)                          1_360.94     2_589.46     3_950.41       0.9296          1.0100            1.0049         4.73
IVF-PQ-nl316-m64 (self)                                1_360.94     8_617.57     9_978.52       0.8978          1.0211            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         1_500.93     3_167.93     4_668.87       0.9458          1.0054            1.0021         7.78
IVF-PQ-nl316-m128-np17 (query)                         1_500.93     3_595.85     5_096.78       0.9458          1.0054            1.0021         7.78
IVF-PQ-nl316-m128-np25 (query)                         1_500.93     5_267.06     6_768.00       0.9458          1.0054            1.0021         7.78
IVF-PQ-nl316-m128 (self)                               1_500.93    17_305.31    18_806.25       0.9179          1.0132            1.0050         7.78
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.43       670.82       703.26       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.43     2_182.97     2_215.40       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_352.23       705.38     2_057.61       0.2858          1.1535            1.1327         1.26
Exhaustive-OPQ-m16 (self)                              1_352.23     2_647.84     4_000.07       0.2593          1.1721            1.1492         1.26
Exhaustive-OPQ-m32 (query)                             1_514.46     1_537.60     3_052.06       0.3254          1.1214            1.1169         2.03
Exhaustive-OPQ-m32 (self)                              1_514.46     5_438.62     6_953.08       0.2827          1.1442            1.1387         2.03
Exhaustive-OPQ-m64 (query)                             1_859.51     3_579.75     5_439.26       0.3791          1.0985            1.0952         3.55
Exhaustive-OPQ-m64 (self)                              1_859.51    12_232.77    14_092.28       0.3227          1.1203            1.1168         3.55
IVF-OPQ-nl158-m16-np7 (query)                          1_405.14       259.51     1_664.65       0.3911          1.0868            1.0878         1.67
IVF-OPQ-nl158-m16-np12 (query)                         1_405.14       373.17     1_778.31       0.3911          1.0868            1.0878         1.67
IVF-OPQ-nl158-m16-np17 (query)                         1_405.14       481.62     1_886.75       0.3911          1.0868            1.0878         1.67
IVF-OPQ-nl158-m16 (self)                               1_405.14     1_917.37     3_322.51       0.3171          1.1192            1.1229         1.67
IVF-OPQ-nl158-m32-np7 (query)                          1_566.12       424.66     1_990.77       0.4970          1.0554            1.0528         2.43
IVF-OPQ-nl158-m32-np12 (query)                         1_566.12       637.69     2_203.80       0.4970          1.0554            1.0528         2.43
IVF-OPQ-nl158-m32-np17 (query)                         1_566.12       848.43     2_414.54       0.4970          1.0554            1.0528         2.43
IVF-OPQ-nl158-m32 (self)                               1_566.12     3_178.41     4_744.52       0.4155          1.0770            1.0762         2.43
IVF-OPQ-nl158-m64-np7 (query)                          1_919.56       743.81     2_663.37       0.6953          1.0193            1.0160         3.96
IVF-OPQ-nl158-m64-np12 (query)                         1_919.56     1_073.25     2_992.81       0.6953          1.0193            1.0160         3.96
IVF-OPQ-nl158-m64-np17 (query)                         1_919.56     1_448.42     3_367.98       0.6953          1.0193            1.0160         3.96
IVF-OPQ-nl158-m64 (self)                               1_919.56     5_148.65     7_068.21       0.6368          1.0268            1.0238         3.96
IVF-OPQ-nl223-m16-np11 (query)                         1_523.24       347.79     1_871.02       0.3980          1.0832            1.0845         1.73
IVF-OPQ-nl223-m16-np14 (query)                         1_523.24       417.46     1_940.70       0.3980          1.0832            1.0845         1.73
IVF-OPQ-nl223-m16-np21 (query)                         1_523.24       573.44     2_096.67       0.3980          1.0832            1.0845         1.73
IVF-OPQ-nl223-m16 (self)                               1_523.24     2_232.52     3_755.76       0.3219          1.1154            1.1195         1.73
IVF-OPQ-nl223-m32-np11 (query)                         1_675.12       587.32     2_262.43       0.5079          1.0516            1.0497         2.50
IVF-OPQ-nl223-m32-np14 (query)                         1_675.12       715.28     2_390.40       0.5079          1.0516            1.0497         2.50
IVF-OPQ-nl223-m32-np21 (query)                         1_675.12     1_005.19     2_680.31       0.5079          1.0516            1.0497         2.50
IVF-OPQ-nl223-m32 (self)                               1_675.12     3_677.35     5_352.47       0.4258          1.0726            1.0728         2.50
IVF-OPQ-nl223-m64-np11 (query)                         2_004.02       978.54     2_982.55       0.7048          1.0174            1.0150         4.02
IVF-OPQ-nl223-m64-np14 (query)                         2_004.02     1_201.90     3_205.92       0.7048          1.0174            1.0150         4.02
IVF-OPQ-nl223-m64-np21 (query)                         2_004.02     1_708.62     3_712.64       0.7048          1.0174            1.0150         4.02
IVF-OPQ-nl223-m64 (self)                               2_004.02     6_013.30     8_017.31       0.6468          1.0247            1.0224         4.02
IVF-OPQ-nl316-m16-np15 (query)                         1_516.27       436.46     1_952.73       0.4097          1.0777            1.0806         2.07
IVF-OPQ-nl316-m16-np17 (query)                         1_516.27       476.36     1_992.63       0.4097          1.0777            1.0806         2.07
IVF-OPQ-nl316-m16-np25 (query)                         1_516.27       654.26     2_170.53       0.4097          1.0777            1.0806         2.07
IVF-OPQ-nl316-m16 (self)                               1_516.27     2_498.10     4_014.36       0.3273          1.1114            1.1168         2.07
IVF-OPQ-nl316-m32-np15 (query)                         1_700.07       746.44     2_446.51       0.5184          1.0477            1.0472         2.84
IVF-OPQ-nl316-m32-np17 (query)                         1_700.07       833.08     2_533.15       0.5184          1.0477            1.0472         2.84
IVF-OPQ-nl316-m32-np25 (query)                         1_700.07     1_163.54     2_863.61       0.5184          1.0477            1.0472         2.84
IVF-OPQ-nl316-m32 (self)                               1_700.07     4_197.43     5_897.50       0.4330          1.0696            1.0707         2.84
IVF-OPQ-nl316-m64-np15 (query)                         2_023.33     1_245.81     3_269.14       0.7140          1.0156            1.0141         4.36
IVF-OPQ-nl316-m64-np17 (query)                         2_023.33     1_397.94     3_421.27       0.7140          1.0156            1.0141         4.36
IVF-OPQ-nl316-m64-np25 (query)                         2_023.33     1_967.34     3_990.67       0.7140          1.0156            1.0141         4.36
IVF-OPQ-nl316-m64 (self)                               2_023.33     6_899.13     8_922.47       0.6548          1.0232            1.0214         4.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.31     1_251.73     1_320.04       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.31     3_988.88     4_057.20       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_747.30       988.88     3_736.18       0.2661          1.1132            1.1009         2.26
Exhaustive-OPQ-m16 (self)                              2_747.30     4_644.34     7_391.64       0.2457          1.1248            1.1094         2.26
Exhaustive-OPQ-m32 (query)                             2_841.85     1_810.35     4_652.19       0.2884          1.0947            1.0956         3.03
Exhaustive-OPQ-m32 (self)                              2_841.85     7_429.57    10_271.42       0.2623          1.1062            1.1056         3.03
Exhaustive-OPQ-m64 (query)                             3_241.67     3_857.29     7_098.96       0.3208          1.0819            1.0844         4.55
Exhaustive-OPQ-m64 (self)                              3_241.67    14_142.51    17_384.17       0.2784          1.0968            1.0989         4.55
Exhaustive-OPQ-m128 (query)                            3_892.36     8_073.78    11_966.14       0.3695          1.0683            1.0688         7.61
Exhaustive-OPQ-m128 (self)                             3_892.36    28_256.32    32_148.68       0.3153          1.0826            1.0834         7.61
IVF-OPQ-nl158-m16-np7 (query)                          2_865.51       597.96     3_463.47       0.3257          1.0783            1.0815         3.07
IVF-OPQ-nl158-m16-np12 (query)                         2_865.51       743.48     3_608.99       0.3257          1.0783            1.0815         3.07
IVF-OPQ-nl158-m16-np17 (query)                         2_865.51       896.15     3_761.66       0.3257          1.0783            1.0815         3.07
IVF-OPQ-nl158-m16 (self)                               2_865.51     4_353.98     7_219.48       0.2760          1.0981            1.1036         3.07
IVF-OPQ-nl158-m32-np7 (query)                          3_073.23       731.31     3_804.54       0.3714          1.0642            1.0651         3.84
IVF-OPQ-nl158-m32-np12 (query)                         3_073.23       960.46     4_033.69       0.3714          1.0642            1.0651         3.84
IVF-OPQ-nl158-m32-np17 (query)                         3_073.23     1_190.85     4_264.08       0.3714          1.0642            1.0651         3.84
IVF-OPQ-nl158-m32 (self)                               3_073.23     5_352.61     8_425.84       0.3023          1.0855            1.0891         3.84
IVF-OPQ-nl158-m64-np7 (query)                          3_406.09     1_060.37     4_466.46       0.4765          1.0410            1.0394         5.36
IVF-OPQ-nl158-m64-np12 (query)                         3_406.09     1_499.79     4_905.89       0.4765          1.0410            1.0394         5.36
IVF-OPQ-nl158-m64-np17 (query)                         3_406.09     2_220.33     5_626.43       0.4765          1.0410            1.0394         5.36
IVF-OPQ-nl158-m64 (self)                               3_406.09     8_363.77    11_769.86       0.4011          1.0548            1.0550         5.36
IVF-OPQ-nl158-m128-np7 (query)                         4_188.29     1_614.71     5_803.00       0.6832          1.0143            1.0117         8.42
IVF-OPQ-nl158-m128-np12 (query)                        4_188.29     2_374.12     6_562.41       0.6832          1.0143            1.0117         8.42
IVF-OPQ-nl158-m128-np17 (query)                        4_188.29     3_070.48     7_258.77       0.6832          1.0143            1.0117         8.42
IVF-OPQ-nl158-m128 (self)                              4_188.29    11_750.53    15_938.82       0.6268          1.0192            1.0169         8.42
IVF-OPQ-nl223-m16-np11 (query)                         3_044.28       718.24     3_762.52       0.3319          1.0744            1.0789         3.20
IVF-OPQ-nl223-m16-np14 (query)                         3_044.28       806.49     3_850.78       0.3319          1.0744            1.0789         3.20
IVF-OPQ-nl223-m16-np21 (query)                         3_044.28     1_016.71     4_060.99       0.3319          1.0744            1.0789         3.20
IVF-OPQ-nl223-m16 (self)                               3_044.28     4_747.89     7_792.17       0.2771          1.0966            1.1032         3.20
IVF-OPQ-nl223-m32-np11 (query)                         3_222.37       924.92     4_147.29       0.3815          1.0601            1.0619         3.96
IVF-OPQ-nl223-m32-np14 (query)                         3_222.37     1_051.57     4_273.94       0.3815          1.0601            1.0619         3.96
IVF-OPQ-nl223-m32-np21 (query)                         3_222.37     1_366.70     4_589.07       0.3815          1.0601            1.0619         3.96
IVF-OPQ-nl223-m32 (self)                               3_222.37     5_917.82     9_140.19       0.3046          1.0836            1.0876         3.96
IVF-OPQ-nl223-m64-np11 (query)                         3_552.31     1_382.33     4_934.64       0.4927          1.0377            1.0365         5.49
IVF-OPQ-nl223-m64-np14 (query)                         3_552.31     1_642.53     5_194.84       0.4927          1.0377            1.0365         5.49
IVF-OPQ-nl223-m64-np21 (query)                         3_552.31     2_227.86     5_780.17       0.4927          1.0377            1.0365         5.49
IVF-OPQ-nl223-m64 (self)                               3_552.31     8_826.52    12_378.83       0.4096          1.0526            1.0530         5.49
IVF-OPQ-nl223-m128-np11 (query)                        4_175.26     2_156.42     6_331.69       0.6925          1.0129            1.0110         8.54
IVF-OPQ-nl223-m128-np14 (query)                        4_175.26     2_588.64     6_763.90       0.6925          1.0129            1.0110         8.54
IVF-OPQ-nl223-m128-np21 (query)                        4_175.26     3_586.96     7_762.22       0.6925          1.0129            1.0110         8.54
IVF-OPQ-nl223-m128 (self)                              4_175.26    13_374.77    17_550.03       0.6353          1.0180            1.0161         8.54
IVF-OPQ-nl316-m16-np15 (query)                         3_093.15       838.87     3_932.02       0.3389          1.0714            1.0763         3.88
IVF-OPQ-nl316-m16-np17 (query)                         3_093.15       895.29     3_988.44       0.3389          1.0714            1.0763         3.88
IVF-OPQ-nl316-m16-np25 (query)                         3_093.15     1_133.17     4_226.32       0.3389          1.0714            1.0763         3.88
IVF-OPQ-nl316-m16 (self)                               3_093.15     5_135.31     8_228.46       0.2802          1.0943            1.1011         3.88
IVF-OPQ-nl316-m32-np15 (query)                         3_254.05     1_091.45     4_345.50       0.3910          1.0568            1.0594         4.65
IVF-OPQ-nl316-m32-np17 (query)                         3_254.05     1_199.86     4_453.91       0.3910          1.0568            1.0594         4.65
IVF-OPQ-nl316-m32-np25 (query)                         3_254.05     1_537.12     4_791.17       0.3910          1.0568            1.0594         4.65
IVF-OPQ-nl316-m32 (self)                               3_254.05     6_500.37     9_754.43       0.3099          1.0809            1.0857         4.65
IVF-OPQ-nl316-m64-np15 (query)                         3_555.81     1_705.65     5_261.46       0.5031          1.0346            1.0346         6.17
IVF-OPQ-nl316-m64-np17 (query)                         3_555.81     1_871.45     5_427.26       0.5031          1.0346            1.0346         6.17
IVF-OPQ-nl316-m64-np25 (query)                         3_555.81     2_558.16     6_113.98       0.5031          1.0346            1.0346         6.17
IVF-OPQ-nl316-m64 (self)                               3_555.81     9_970.07    13_525.88       0.4170          1.0500            1.0512         6.17
IVF-OPQ-nl316-m128-np15 (query)                        4_191.14     2_677.76     6_868.90       0.7007          1.0115            1.0104         9.23
IVF-OPQ-nl316-m128-np17 (query)                        4_191.14     2_963.81     7_154.95       0.7007          1.0115            1.0104         9.23
IVF-OPQ-nl316-m128-np25 (query)                        4_191.14     4_083.08     8_274.22       0.7007          1.0115            1.0104         9.23
IVF-OPQ-nl316-m128 (self)                              4_191.14    15_123.15    19_314.29       0.6420          1.0168            1.0155         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.61     1_778.50     1_880.12       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.61     5_889.99     5_991.61       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_522.44     1_509.15     6_031.58       0.2572          1.0953            1.0838         3.76
Exhaustive-OPQ-m16 (self)                              4_522.44     8_100.97    12_623.41       0.2411          1.1050            1.0895         3.76
Exhaustive-OPQ-m32 (query)                             4_512.92     2_321.96     6_834.88       0.2801          1.0776            1.0800         4.53
Exhaustive-OPQ-m32 (self)                              4_512.92    10_919.43    15_432.34       0.2567          1.0868            1.0870         4.53
Exhaustive-OPQ-m64 (query)                             5_050.20     4_414.01     9_464.21       0.2974          1.0722            1.0753         6.05
Exhaustive-OPQ-m64 (self)                              5_050.20    17_681.55    22_731.75       0.2647          1.0826            1.0853         6.05
Exhaustive-OPQ-m128 (query)                            5_575.90     8_630.34    14_206.24       0.3296          1.0636            1.0665         9.11
Exhaustive-OPQ-m128 (self)                             5_575.90    33_032.84    38_608.74       0.2830          1.0757            1.0788         9.11
IVF-OPQ-nl158-m16-np7 (query)                          4_823.04     1_145.44     5_968.48       0.3037          1.0689            1.0729         4.98
IVF-OPQ-nl158-m16-np12 (query)                         4_823.04     1_334.81     6_157.85       0.3037          1.0689            1.0729         4.98
IVF-OPQ-nl158-m16-np17 (query)                         4_823.04     1_508.12     6_331.15       0.3037          1.0689            1.0729         4.98
IVF-OPQ-nl158-m16 (self)                               4_823.04     8_209.70    13_032.74       0.2644          1.0834            1.0883         4.98
IVF-OPQ-nl158-m32-np7 (query)                          4_730.37     1_332.96     6_063.34       0.3322          1.0607            1.0630         5.74
IVF-OPQ-nl158-m32-np12 (query)                         4_730.37     1_635.76     6_366.13       0.3322          1.0607            1.0630         5.74
IVF-OPQ-nl158-m32-np17 (query)                         4_730.37     1_909.21     6_639.58       0.3322          1.0607            1.0630         5.74
IVF-OPQ-nl158-m32 (self)                               4_730.37     9_559.97    14_290.34       0.2735          1.0790            1.0832         5.74
IVF-OPQ-nl158-m64-np7 (query)                          5_087.59     1_655.83     6_743.42       0.3933          1.0474            1.0468         7.27
IVF-OPQ-nl158-m64-np12 (query)                         5_087.59     2_145.34     7_232.93       0.3933          1.0474            1.0468         7.27
IVF-OPQ-nl158-m64-np17 (query)                         5_087.59     2_662.32     7_749.91       0.3933          1.0474            1.0468         7.27
IVF-OPQ-nl158-m64 (self)                               5_087.59    13_047.54    18_135.13       0.3185          1.0632            1.0646         7.27
IVF-OPQ-nl158-m128-np7 (query)                         5_559.11     2_464.46     8_023.57       0.5427          1.0245            1.0221        10.32
IVF-OPQ-nl158-m128-np12 (query)                        5_559.11     3_432.26     8_991.38       0.5427          1.0245            1.0221        10.32
IVF-OPQ-nl158-m128-np17 (query)                        5_559.11     4_404.22     9_963.33       0.5427          1.0245            1.0221        10.32
IVF-OPQ-nl158-m128 (self)                              5_559.11    17_933.82    23_492.93       0.4724          1.0322            1.0309        10.32
IVF-OPQ-nl223-m16-np11 (query)                         4_870.03     1_292.61     6_162.64       0.3097          1.0660            1.0711         5.17
IVF-OPQ-nl223-m16-np14 (query)                         4_870.03     1_396.84     6_266.87       0.3097          1.0660            1.0711         5.17
IVF-OPQ-nl223-m16-np21 (query)                         4_870.03     1_668.62     6_538.66       0.3097          1.0660            1.0711         5.17
IVF-OPQ-nl223-m16 (self)                               4_870.03     8_697.96    13_568.00       0.2676          1.0815            1.0878         5.17
IVF-OPQ-nl223-m32-np11 (query)                         4_987.29     1_559.59     6_546.88       0.3403          1.0574            1.0606         5.93
IVF-OPQ-nl223-m32-np14 (query)                         4_987.29     1_730.68     6_717.97       0.3403          1.0574            1.0606         5.93
IVF-OPQ-nl223-m32-np21 (query)                         4_987.29     2_139.93     7_127.22       0.3403          1.0574            1.0606         5.93
IVF-OPQ-nl223-m32 (self)                               4_987.29    10_323.72    15_311.01       0.2769          1.0770            1.0821         5.93
IVF-OPQ-nl223-m64-np11 (query)                         5_455.28     2_036.37     7_491.65       0.4056          1.0433            1.0443         7.46
IVF-OPQ-nl223-m64-np14 (query)                         5_455.28     2_329.18     7_784.46       0.4056          1.0433            1.0443         7.46
IVF-OPQ-nl223-m64-np21 (query)                         5_455.28     3_005.83     8_461.10       0.4056          1.0433            1.0443         7.46
IVF-OPQ-nl223-m64 (self)                               5_455.28    13_212.62    18_667.90       0.3253          1.0604            1.0630         7.46
IVF-OPQ-nl223-m128-np11 (query)                        6_030.11     3_195.43     9_225.53       0.5564          1.0216            1.0207        10.51
IVF-OPQ-nl223-m128-np14 (query)                        6_030.11     3_768.71     9_798.82       0.5564          1.0216            1.0207        10.51
IVF-OPQ-nl223-m128-np21 (query)                        6_030.11     5_118.97    11_149.08       0.5564          1.0216            1.0207        10.51
IVF-OPQ-nl223-m128 (self)                              6_030.11    20_292.96    26_323.07       0.4809          1.0301            1.0299        10.51
IVF-OPQ-nl316-m16-np15 (query)                         4_936.72     1_436.92     6_373.64       0.3145          1.0637            1.0691         6.19
IVF-OPQ-nl316-m16-np17 (query)                         4_936.72     1_513.48     6_450.19       0.3145          1.0637            1.0691         6.19
IVF-OPQ-nl316-m16-np25 (query)                         4_936.72     1_798.80     6_735.52       0.3145          1.0637            1.0691         6.19
IVF-OPQ-nl316-m16 (self)                               4_936.72     9_162.19    14_098.91       0.2683          1.0802            1.0863         6.19
IVF-OPQ-nl316-m32-np15 (query)                         5_136.49     1_791.90     6_928.39       0.3455          1.0554            1.0591         6.95
IVF-OPQ-nl316-m32-np17 (query)                         5_136.49     1_907.21     7_043.70       0.3455          1.0554            1.0591         6.95
IVF-OPQ-nl316-m32-np25 (query)                         5_136.49     2_370.89     7_507.38       0.3455          1.0554            1.0591         6.95
IVF-OPQ-nl316-m32 (self)                               5_136.49    11_088.53    16_225.01       0.2785          1.0757            1.0812         6.95
IVF-OPQ-nl316-m64-np15 (query)                         5_532.94     2_411.88     7_944.81       0.4164          1.0405            1.0421         8.48
IVF-OPQ-nl316-m64-np17 (query)                         5_532.94     2_615.19     8_148.13       0.4164          1.0405            1.0421         8.48
IVF-OPQ-nl316-m64-np25 (query)                         5_532.94     3_371.18     8_904.11       0.4164          1.0405            1.0421         8.48
IVF-OPQ-nl316-m64 (self)                               5_532.94    14_423.61    19_956.54       0.3306          1.0583            1.0617         8.48
IVF-OPQ-nl316-m128-np15 (query)                        6_026.73     3_938.91     9_965.64       0.5665          1.0199            1.0196        11.53
IVF-OPQ-nl316-m128-np17 (query)                        6_026.73     4_307.93    10_334.66       0.5665          1.0199            1.0196        11.53
IVF-OPQ-nl316-m128-np25 (query)                        6_026.73     5_811.61    11_838.34       0.5665          1.0199            1.0196        11.53
IVF-OPQ-nl316-m128 (self)                              6_026.73    22_626.01    28_652.74       0.4890          1.0288            1.0288        11.53
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

##### Lowrank data

Let's test the manifold data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.47       668.84       701.31       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.47     2_204.62     2_237.09       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_378.21       700.50     2_078.71       0.3003          1.2501            1.2435         1.26
Exhaustive-OPQ-m16 (self)                              1_378.21     2_648.54     4_026.74       0.2369          1.3780            1.3719         1.26
Exhaustive-OPQ-m32 (query)                             1_508.42     1_532.58     3_041.01       0.4212          1.1523            1.1471         2.03
Exhaustive-OPQ-m32 (self)                              1_508.42     5_405.16     6_913.58       0.3382          1.2478            1.2414         2.03
Exhaustive-OPQ-m64 (query)                             1_857.14     3_565.50     5_422.64       0.5666          1.0762            1.0730         3.55
Exhaustive-OPQ-m64 (self)                              1_857.14    12_259.70    14_116.83       0.4882          1.1282            1.1236         3.55
IVF-OPQ-nl158-m16-np7 (query)                          1_403.24       261.34     1_664.58       0.6991          1.0329            1.0312         1.67
IVF-OPQ-nl158-m16-np12 (query)                         1_403.24       369.38     1_772.62       0.6991          1.0329            1.0312         1.67
IVF-OPQ-nl158-m16-np17 (query)                         1_403.24       505.86     1_909.09       0.6991          1.0329            1.0312         1.67
IVF-OPQ-nl158-m16 (self)                               1_403.24     1_947.37     3_350.60       0.6204          1.0628            1.0600         1.67
IVF-OPQ-nl158-m32-np7 (query)                          1_576.13       421.72     1_997.84       0.8013          1.0136            1.0124         2.43
IVF-OPQ-nl158-m32-np12 (query)                         1_576.13       634.35     2_210.47       0.8013          1.0136            1.0124         2.43
IVF-OPQ-nl158-m32-np17 (query)                         1_576.13       859.05     2_435.18       0.8013          1.0136            1.0124         2.43
IVF-OPQ-nl158-m32 (self)                               1_576.13     3_205.41     4_781.53       0.7513          1.0252            1.0232         2.43
IVF-OPQ-nl158-m64-np7 (query)                          1_929.80       692.30     2_622.10       0.8619          1.0064            1.0054         3.96
IVF-OPQ-nl158-m64-np12 (query)                         1_929.80     1_069.72     2_999.52       0.8619          1.0064            1.0054         3.96
IVF-OPQ-nl158-m64-np17 (query)                         1_929.80     1_476.88     3_406.68       0.8619          1.0064            1.0054         3.96
IVF-OPQ-nl158-m64 (self)                               1_929.80     5_262.34     7_192.13       0.8315          1.0111            1.0096         3.96
IVF-OPQ-nl223-m16-np11 (query)                         1_517.84       349.85     1_867.68       0.7077          1.0308            1.0292         1.73
IVF-OPQ-nl223-m16-np14 (query)                         1_517.84       417.42     1_935.25       0.7077          1.0308            1.0292         1.73
IVF-OPQ-nl223-m16-np21 (query)                         1_517.84       583.91     2_101.74       0.7077          1.0308            1.0292         1.73
IVF-OPQ-nl223-m16 (self)                               1_517.84     2_266.53     3_784.37       0.6294          1.0591            1.0566         1.73
IVF-OPQ-nl223-m32-np11 (query)                         1_697.40       581.06     2_278.46       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32-np14 (query)                         1_697.40       706.43     2_403.83       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32-np21 (query)                         1_697.40     1_013.21     2_710.61       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32 (self)                               1_697.40     3_697.26     5_394.66       0.7555          1.0241            1.0224         2.50
IVF-OPQ-nl223-m64-np11 (query)                         2_040.80       971.97     3_012.76       0.8648          1.0060            1.0051         4.02
IVF-OPQ-nl223-m64-np14 (query)                         2_040.80     1_194.03     3_234.83       0.8648          1.0060            1.0051         4.02
IVF-OPQ-nl223-m64-np21 (query)                         2_040.80     1_740.65     3_781.45       0.8648          1.0060            1.0051         4.02
IVF-OPQ-nl223-m64 (self)                               2_040.80     6_102.41     8_143.21       0.8351          1.0105            1.0093         4.02
IVF-OPQ-nl316-m16-np15 (query)                         1_547.19       433.09     1_980.27       0.7116          1.0299            1.0284         2.07
IVF-OPQ-nl316-m16-np17 (query)                         1_547.19       474.61     2_021.79       0.7116          1.0299            1.0284         2.07
IVF-OPQ-nl316-m16-np25 (query)                         1_547.19       656.42     2_203.61       0.7116          1.0299            1.0284         2.07
IVF-OPQ-nl316-m16 (self)                               1_547.19     2_494.07     4_041.26       0.6361          1.0569            1.0546         2.07
IVF-OPQ-nl316-m32-np15 (query)                         1_718.96       740.43     2_459.40       0.8070          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np17 (query)                         1_718.96       825.87     2_544.84       0.8070          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np25 (query)                         1_718.96     1_161.39     2_880.35       0.8070          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32 (self)                               1_718.96     4_191.75     5_910.71       0.7593          1.0233            1.0217         2.84
IVF-OPQ-nl316-m64-np15 (query)                         2_006.00     1_239.63     3_245.62       0.8655          1.0060            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         2_006.00     1_385.57     3_391.57       0.8655          1.0060            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         2_006.00     1_978.67     3_984.66       0.8655          1.0060            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               2_006.00     6_897.65     8_903.65       0.8374          1.0102            1.0089         4.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.89     1_235.73     1_304.62       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.89     4_079.26     4_148.14       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_801.22       995.11     3_796.33       0.2324          1.2133            1.2098         2.26
Exhaustive-OPQ-m16 (self)                              2_801.22     4_666.53     7_467.75       0.1872          1.2980            1.2978         2.26
Exhaustive-OPQ-m32 (query)                             2_938.30     1_817.44     4_755.74       0.3204          1.1503            1.1475         3.03
Exhaustive-OPQ-m32 (self)                              2_938.30     7_383.59    10_321.89       0.2587          1.2172            1.2150         3.03
Exhaustive-OPQ-m64 (query)                             3_263.64     3_880.04     7_143.67       0.4335          1.0938            1.0916         4.55
Exhaustive-OPQ-m64 (self)                              3_263.64    14_156.02    17_419.66       0.3626          1.1418            1.1394         4.55
Exhaustive-OPQ-m128 (query)                            3_937.80     8_097.58    12_035.38       0.5701          1.0489            1.0475         7.61
Exhaustive-OPQ-m128 (self)                             3_937.80    28_322.67    32_260.47       0.5007          1.0773            1.0754         7.61
IVF-OPQ-nl158-m16-np7 (query)                          2_888.02       592.49     3_480.51       0.5402          1.0563            1.0552         3.07
IVF-OPQ-nl158-m16-np12 (query)                         2_888.02       739.15     3_627.17       0.5402          1.0563            1.0552         3.07
IVF-OPQ-nl158-m16-np17 (query)                         2_888.02       892.02     3_780.04       0.5402          1.0563            1.0552         3.07
IVF-OPQ-nl158-m16 (self)                               2_888.02     4_364.90     7_252.92       0.4397          1.1007            1.0987         3.07
IVF-OPQ-nl158-m32-np7 (query)                          3_061.76       726.07     3_787.83       0.6862          1.0241            1.0229         3.84
IVF-OPQ-nl158-m32-np12 (query)                         3_061.76       954.80     4_016.55       0.6862          1.0241            1.0229         3.84
IVF-OPQ-nl158-m32-np17 (query)                         3_061.76     1_185.61     4_247.36       0.6862          1.0241            1.0229         3.84
IVF-OPQ-nl158-m32 (self)                               3_061.76     5_343.48     8_405.24       0.6115          1.0433            1.0412         3.84
IVF-OPQ-nl158-m64-np7 (query)                          3_424.39     1_052.65     4_477.04       0.7800          1.0115            1.0104         5.36
IVF-OPQ-nl158-m64-np12 (query)                         3_424.39     1_481.07     4_905.47       0.7800          1.0115            1.0104         5.36
IVF-OPQ-nl158-m64-np17 (query)                         3_424.39     1_915.95     5_340.34       0.7800          1.0115            1.0104         5.36
IVF-OPQ-nl158-m64 (self)                               3_424.39     7_776.65    11_201.05       0.7320          1.0197            1.0177         5.36
IVF-OPQ-nl158-m128-np7 (query)                         3_955.41     1_587.23     5_542.65       0.8382          1.0061            1.0051         8.42
IVF-OPQ-nl158-m128-np12 (query)                        3_955.41     2_318.89     6_274.30       0.8382          1.0061            1.0051         8.42
IVF-OPQ-nl158-m128-np17 (query)                        3_955.41     3_057.48     7_012.89       0.8382          1.0061            1.0051         8.42
IVF-OPQ-nl158-m128 (self)                              3_955.41    11_786.48    15_741.90       0.8109          1.0097            1.0080         8.42
IVF-OPQ-nl223-m16-np11 (query)                         3_015.46       718.11     3_733.57       0.5464          1.0543            1.0531         3.20
IVF-OPQ-nl223-m16-np14 (query)                         3_015.46       809.60     3_825.06       0.5464          1.0543            1.0531         3.20
IVF-OPQ-nl223-m16-np21 (query)                         3_015.46     1_019.75     4_035.21       0.5464          1.0543            1.0531         3.20
IVF-OPQ-nl223-m16 (self)                               3_015.46     4_750.95     7_766.41       0.4469          1.0974            1.0955         3.20
IVF-OPQ-nl223-m32-np11 (query)                         3_183.78       907.56     4_091.34       0.6906          1.0234            1.0224         3.96
IVF-OPQ-nl223-m32-np14 (query)                         3_183.78     1_103.94     4_287.72       0.6906          1.0234            1.0224         3.96
IVF-OPQ-nl223-m32-np21 (query)                         3_183.78     1_362.22     4_546.00       0.6906          1.0234            1.0224         3.96
IVF-OPQ-nl223-m32 (self)                               3_183.78     5_911.17     9_094.95       0.6182          1.0416            1.0399         3.96
IVF-OPQ-nl223-m64-np11 (query)                         3_561.25     1_381.41     4_942.66       0.7831          1.0111            1.0101         5.49
IVF-OPQ-nl223-m64-np14 (query)                         3_561.25     1_629.36     5_190.61       0.7831          1.0111            1.0101         5.49
IVF-OPQ-nl223-m64-np21 (query)                         3_561.25     2_214.56     5_775.81       0.7831          1.0111            1.0101         5.49
IVF-OPQ-nl223-m64 (self)                               3_561.25     8_909.44    12_470.68       0.7362          1.0189            1.0173         5.49
IVF-OPQ-nl223-m128-np11 (query)                        4_177.46     2_138.12     6_315.58       0.8403          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128-np14 (query)                        4_177.46     2_550.88     6_728.34       0.8403          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128-np21 (query)                        4_177.46     3_605.28     7_782.74       0.8403          1.0059            1.0050         8.54
IVF-OPQ-nl223-m128 (self)                              4_177.46    13_379.87    17_557.33       0.8131          1.0093            1.0080         8.54
IVF-OPQ-nl316-m16-np15 (query)                         3_083.98       837.11     3_921.09       0.5522          1.0526            1.0515         3.88
IVF-OPQ-nl316-m16-np17 (query)                         3_083.98       893.39     3_977.36       0.5522          1.0526            1.0515         3.88
IVF-OPQ-nl316-m16-np25 (query)                         3_083.98     1_133.06     4_217.04       0.5522          1.0526            1.0515         3.88
IVF-OPQ-nl316-m16 (self)                               3_083.98     5_140.92     8_224.90       0.4548          1.0943            1.0926         3.88
IVF-OPQ-nl316-m32-np15 (query)                         3_260.46     1_086.12     4_346.59       0.6949          1.0228            1.0215         4.65
IVF-OPQ-nl316-m32-np17 (query)                         3_260.46     1_177.42     4_437.89       0.6949          1.0228            1.0215         4.65
IVF-OPQ-nl316-m32-np25 (query)                         3_260.46     1_537.45     4_797.92       0.6949          1.0228            1.0215         4.65
IVF-OPQ-nl316-m32 (self)                               3_260.46     6_498.01     9_758.48       0.6243          1.0401            1.0384         4.65
IVF-OPQ-nl316-m64-np15 (query)                         3_580.52     1_695.03     5_275.55       0.7844          1.0109            1.0100         6.17
IVF-OPQ-nl316-m64-np17 (query)                         3_580.52     1_859.45     5_439.97       0.7844          1.0109            1.0100         6.17
IVF-OPQ-nl316-m64-np25 (query)                         3_580.52     2_525.92     6_106.44       0.7844          1.0109            1.0100         6.17
IVF-OPQ-nl316-m64 (self)                               3_580.52     9_841.21    13_421.72       0.7395          1.0183            1.0168         6.17
IVF-OPQ-nl316-m128-np15 (query)                        4_214.95     2_664.33     6_879.28       0.8407          1.0059            1.0050         9.23
IVF-OPQ-nl316-m128-np17 (query)                        4_214.95     2_940.93     7_155.89       0.8407          1.0059            1.0050         9.23
IVF-OPQ-nl316-m128-np25 (query)                        4_214.95     4_071.43     8_286.39       0.8407          1.0059            1.0050         9.23
IVF-OPQ-nl316-m128 (self)                              4_214.95    14_965.34    19_180.29       0.8145          1.0092            1.0078         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.11     1_766.41     1_867.52       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.11     5_883.45     5_984.56       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_421.95     1_501.03     5_922.98       0.2278          1.2029            1.1998         3.76
Exhaustive-OPQ-m16 (self)                              4_421.95     8_123.43    12_545.38       0.1860          1.2978            1.2978         3.76
Exhaustive-OPQ-m32 (query)                             4_462.21     2_329.67     6_791.88       0.3126          1.1446            1.1410         4.53
Exhaustive-OPQ-m32 (self)                              4_462.21    11_125.21    15_587.42       0.2577          1.2169            1.2141         4.53
Exhaustive-OPQ-m64 (query)                             4_886.07     4_398.85     9_284.91       0.4083          1.0976            1.0948         6.05
Exhaustive-OPQ-m64 (self)                              4_886.07    17_652.34    22_538.40       0.3475          1.1498            1.1473         6.05
Exhaustive-OPQ-m128 (query)                            5_588.92     8_668.94    14_257.87       0.5288          1.0566            1.0549         9.11
Exhaustive-OPQ-m128 (self)                             5_588.92    31_998.32    37_587.25       0.4657          1.0908            1.0889         9.11
IVF-OPQ-nl158-m16-np7 (query)                          4_551.97     1_241.00     5_792.96       0.5358          1.0535            1.0524         4.98
IVF-OPQ-nl158-m16-np12 (query)                         4_551.97     1_416.65     5_968.62       0.5358          1.0535            1.0524         4.98
IVF-OPQ-nl158-m16-np17 (query)                         4_551.97     1_508.36     6_060.33       0.5358          1.0535            1.0524         4.98
IVF-OPQ-nl158-m16 (self)                               4_551.97     8_201.69    12_753.65       0.4333          1.1029            1.1012         4.98
IVF-OPQ-nl158-m32-np7 (query)                          4_677.52     1_316.77     5_994.29       0.6775          1.0239            1.0228         5.74
IVF-OPQ-nl158-m32-np12 (query)                         4_677.52     1_612.16     6_289.68       0.6775          1.0239            1.0228         5.74
IVF-OPQ-nl158-m32-np17 (query)                         4_677.52     1_932.74     6_610.26       0.6775          1.0239            1.0228         5.74
IVF-OPQ-nl158-m32 (self)                               4_677.52     9_544.79    14_222.31       0.6035          1.0449            1.0430         5.74
IVF-OPQ-nl158-m64-np7 (query)                          5_453.27     1_661.02     7_114.30       0.7767          1.0111            1.0100         7.27
IVF-OPQ-nl158-m64-np12 (query)                         5_453.27     2_161.52     7_614.80       0.7767          1.0111            1.0100         7.27
IVF-OPQ-nl158-m64-np17 (query)                         5_453.27     2_659.69     8_112.96       0.7767          1.0111            1.0100         7.27
IVF-OPQ-nl158-m64 (self)                               5_453.27    12_028.46    17_481.74       0.7276          1.0203            1.0182         7.27
IVF-OPQ-nl158-m128-np7 (query)                         5_976.79     2_443.17     8_419.95       0.8339          1.0061            1.0050        10.32
IVF-OPQ-nl158-m128-np12 (query)                        5_976.79     3_431.79     9_408.58       0.8339          1.0061            1.0050        10.32
IVF-OPQ-nl158-m128-np17 (query)                        5_976.79     4_431.10    10_407.89       0.8339          1.0061            1.0050        10.32
IVF-OPQ-nl158-m128 (self)                              5_976.79    17_975.46    23_952.24       0.8053          1.0103            1.0084        10.32
IVF-OPQ-nl223-m16-np11 (query)                         4_919.76     1_295.59     6_215.34       0.5404          1.0522            1.0508         5.17
IVF-OPQ-nl223-m16-np14 (query)                         4_919.76     1_404.19     6_323.95       0.5404          1.0522            1.0508         5.17
IVF-OPQ-nl223-m16-np21 (query)                         4_919.76     1_654.16     6_573.92       0.5404          1.0522            1.0508         5.17
IVF-OPQ-nl223-m16 (self)                               4_919.76     8_677.75    13_597.51       0.4416          1.0993            1.0976         5.17
IVF-OPQ-nl223-m32-np11 (query)                         5_188.17     1_573.79     6_761.96       0.6813          1.0233            1.0221         5.93
IVF-OPQ-nl223-m32-np14 (query)                         5_188.17     1_732.65     6_920.82       0.6813          1.0233            1.0221         5.93
IVF-OPQ-nl223-m32-np21 (query)                         5_188.17     2_159.19     7_347.36       0.6813          1.0233            1.0221         5.93
IVF-OPQ-nl223-m32 (self)                               5_188.17    10_337.07    15_525.24       0.6076          1.0439            1.0421         5.93
IVF-OPQ-nl223-m64-np11 (query)                         5_604.92     2_064.50     7_669.42       0.7776          1.0109            1.0099         7.46
IVF-OPQ-nl223-m64-np14 (query)                         5_604.92     2_338.86     7_943.78       0.7776          1.0109            1.0099         7.46
IVF-OPQ-nl223-m64-np21 (query)                         5_604.92     3_013.84     8_618.76       0.7776          1.0109            1.0099         7.46
IVF-OPQ-nl223-m64 (self)                               5_604.92    13_209.71    18_814.63       0.7310          1.0196            1.0179         7.46
IVF-OPQ-nl223-m128-np11 (query)                        5_965.82     3_215.81     9_181.64       0.8350          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np14 (query)                        5_965.82     3_762.76     9_728.58       0.8350          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np21 (query)                        5_965.82     5_140.18    11_106.00       0.8350          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128 (self)                              5_965.82    20_236.34    26_202.16       0.8079          1.0099            1.0083        10.51
IVF-OPQ-nl316-m16-np15 (query)                         4_900.76     1_434.94     6_335.70       0.5469          1.0509            1.0498         6.19
IVF-OPQ-nl316-m16-np17 (query)                         4_900.76     1_504.90     6_405.66       0.5469          1.0509            1.0498         6.19
IVF-OPQ-nl316-m16-np25 (query)                         4_900.76     1_874.67     6_775.42       0.5469          1.0509            1.0498         6.19
IVF-OPQ-nl316-m16 (self)                               4_900.76     9_164.39    14_065.15       0.4474          1.0969            1.0954         6.19
IVF-OPQ-nl316-m32-np15 (query)                         5_170.83     1_807.47     6_978.30       0.6834          1.0230            1.0219         6.95
IVF-OPQ-nl316-m32-np17 (query)                         5_170.83     1_901.42     7_072.25       0.6834          1.0230            1.0219         6.95
IVF-OPQ-nl316-m32-np25 (query)                         5_170.83     2_408.84     7_579.67       0.6834          1.0230            1.0219         6.95
IVF-OPQ-nl316-m32 (self)                               5_170.83    11_112.28    16_283.11       0.6111          1.0430            1.0413         6.95
IVF-OPQ-nl316-m64-np15 (query)                         5_537.84     2_406.64     7_944.48       0.7798          1.0106            1.0097         8.48
IVF-OPQ-nl316-m64-np17 (query)                         5_537.84     2_597.98     8_135.82       0.7798          1.0106            1.0097         8.48
IVF-OPQ-nl316-m64-np25 (query)                         5_537.84     3_370.03     8_907.87       0.7798          1.0106            1.0097         8.48
IVF-OPQ-nl316-m64 (self)                               5_537.84    14_421.30    19_959.15       0.7344          1.0190            1.0175         8.48
IVF-OPQ-nl316-m128-np15 (query)                        6_012.14     3_911.10     9_923.24       0.8359          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128-np17 (query)                        6_012.14     4_290.68    10_302.81       0.8359          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128-np25 (query)                        6_012.14     5_818.44    11_830.58       0.8359          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128 (self)                              6_012.14    22_572.23    28_584.37       0.8096          1.0096            1.0082        11.53
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.44       677.50       709.94       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.44     2_181.51     2_213.96       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             1_370.21       703.06     2_073.27       0.7915          1.0818            1.0683         1.26
Exhaustive-OPQ-m16 (self)                              1_370.21     2_639.71     4_009.92       0.7235          1.1500            1.1252         1.26
Exhaustive-OPQ-m32 (query)                             1_502.80     1_527.98     3_030.78       0.8299          1.0537            1.0423         2.03
Exhaustive-OPQ-m32 (self)                              1_502.80     5_393.33     6_896.12       0.7758          1.0978            1.0769         2.03
Exhaustive-OPQ-m64 (query)                             1_853.02     3_573.63     5_426.65       0.8560          1.0398            1.0294         3.55
Exhaustive-OPQ-m64 (self)                              1_853.02    12_231.99    14_085.01       0.8090          1.0724            1.0538         3.55
IVF-OPQ-nl158-m16-np7 (query)                          1_406.16       269.44     1_675.60       0.8916          1.0201            1.0155         1.67
IVF-OPQ-nl158-m16-np12 (query)                         1_406.16       396.97     1_803.13       0.8925          1.0196            1.0152         1.67
IVF-OPQ-nl158-m16-np17 (query)                         1_406.16       542.55     1_948.71       0.8925          1.0196            1.0152         1.67
IVF-OPQ-nl158-m16 (self)                               1_406.16     2_067.34     3_473.50       0.8514          1.0384            1.0303         1.67
IVF-OPQ-nl158-m32-np7 (query)                          1_580.42       443.19     2_023.61       0.9124          1.0130            1.0095         2.43
IVF-OPQ-nl158-m32-np12 (query)                         1_580.42       697.52     2_277.94       0.9134          1.0125            1.0093         2.43
IVF-OPQ-nl158-m32-np17 (query)                         1_580.42       987.49     2_567.91       0.9134          1.0125            1.0093         2.43
IVF-OPQ-nl158-m32 (self)                               1_580.42     3_512.49     5_092.92       0.8792          1.0247            1.0190         2.43
IVF-OPQ-nl158-m64-np7 (query)                          1_908.57       743.15     2_651.72       0.9245          1.0098            1.0065         3.96
IVF-OPQ-nl158-m64-np12 (query)                         1_908.57     1_209.27     3_117.84       0.9257          1.0093            1.0063         3.96
IVF-OPQ-nl158-m64-np17 (query)                         1_908.57     1_755.65     3_664.22       0.9257          1.0093            1.0063         3.96
IVF-OPQ-nl158-m64 (self)                               1_908.57     5_906.16     7_814.73       0.8968          1.0183            1.0129         3.96
IVF-OPQ-nl223-m16-np11 (query)                         1_530.26       360.12     1_890.39       0.8969          1.0180            1.0137         1.73
IVF-OPQ-nl223-m16-np14 (query)                         1_530.26       432.86     1_963.12       0.8970          1.0180            1.0137         1.73
IVF-OPQ-nl223-m16-np21 (query)                         1_530.26       609.70     2_139.96       0.8970          1.0180            1.0137         1.73
IVF-OPQ-nl223-m16 (self)                               1_530.26     2_369.43     3_899.69       0.8574          1.0356            1.0273         1.73
IVF-OPQ-nl223-m32-np11 (query)                         1_684.65       610.64     2_295.29       0.9154          1.0118            1.0085         2.50
IVF-OPQ-nl223-m32-np14 (query)                         1_684.65       752.81     2_437.46       0.9155          1.0118            1.0085         2.50
IVF-OPQ-nl223-m32-np21 (query)                         1_684.65     1_083.55     2_768.20       0.9156          1.0118            1.0085         2.50
IVF-OPQ-nl223-m32 (self)                               1_684.65     3_960.90     5_645.55       0.8825          1.0234            1.0176         2.50
IVF-OPQ-nl223-m64-np11 (query)                         2_002.71     1_028.19     3_030.90       0.9266          1.0091            1.0060         4.02
IVF-OPQ-nl223-m64-np14 (query)                         2_002.71     1_304.23     3_306.94       0.9268          1.0090            1.0060         4.02
IVF-OPQ-nl223-m64-np21 (query)                         2_002.71     1_884.08     3_886.80       0.9268          1.0090            1.0060         4.02
IVF-OPQ-nl223-m64 (self)                               2_002.71     6_613.73     8_616.44       0.8988          1.0176            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         1_494.49       440.18     1_934.67       0.9020          1.0163            1.0125         2.07
IVF-OPQ-nl316-m16-np17 (query)                         1_494.49       487.76     1_982.25       0.9021          1.0163            1.0124         2.07
IVF-OPQ-nl316-m16-np25 (query)                         1_494.49       677.11     2_171.60       0.9021          1.0163            1.0124         2.07
IVF-OPQ-nl316-m16 (self)                               1_494.49     2_570.67     4_065.16       0.8638          1.0327            1.0243         2.07
IVF-OPQ-nl316-m32-np15 (query)                         1_692.38       765.73     2_458.11       0.9182          1.0111            1.0079         2.84
IVF-OPQ-nl316-m32-np17 (query)                         1_692.38       854.80     2_547.18       0.9182          1.0111            1.0079         2.84
IVF-OPQ-nl316-m32-np25 (query)                         1_692.38     1_214.78     2_907.16       0.9183          1.0111            1.0079         2.84
IVF-OPQ-nl316-m32 (self)                               1_692.38     4_374.22     6_066.60       0.8852          1.0224            1.0166         2.84
IVF-OPQ-nl316-m64-np15 (query)                         2_010.48     1_401.32     3_411.80       0.9288          1.0085            1.0055         4.36
IVF-OPQ-nl316-m64-np17 (query)                         2_010.48     1_558.92     3_569.40       0.9289          1.0085            1.0055         4.36
IVF-OPQ-nl316-m64-np25 (query)                         2_010.48     2_098.92     4_109.40       0.9289          1.0085            1.0055         4.36
IVF-OPQ-nl316-m64 (self)                               2_010.48     7_473.66     9_484.14       0.9011          1.0166            1.0116         4.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.73     1_233.11     1_301.83       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.73     4_071.29     4_140.02       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             2_806.54       989.65     3_796.19       0.7548          1.1139            1.0983         2.26
Exhaustive-OPQ-m16 (self)                              2_806.54     4_632.86     7_439.40       0.6783          1.2043            1.1743         2.26
Exhaustive-OPQ-m32 (query)                             2_887.58     1_811.56     4_699.14       0.8059          1.0693            1.0573         3.03
Exhaustive-OPQ-m32 (self)                              2_887.58     7_425.03    10_312.61       0.7451          1.1246            1.1019         3.03
Exhaustive-OPQ-m64 (query)                             3_237.53     3_870.47     7_108.00       0.8410          1.0458            1.0366         4.55
Exhaustive-OPQ-m64 (self)                              3_237.53    14_174.54    17_412.08       0.7916          1.0819            1.0657         4.55
Exhaustive-OPQ-m128 (query)                            3_915.53     8_069.54    11_985.06       0.9200          1.0107            1.0070         7.61
Exhaustive-OPQ-m128 (self)                             3_915.53    28_212.36    32_127.89       0.8931          1.0194            1.0139         7.61
IVF-OPQ-nl158-m16-np7 (query)                          2_882.61       609.86     3_492.47       0.8920          1.0215            1.0161         3.07
IVF-OPQ-nl158-m16-np12 (query)                         2_882.61       787.68     3_670.28       0.8924          1.0213            1.0159         3.07
IVF-OPQ-nl158-m16-np17 (query)                         2_882.61       933.01     3_815.62       0.8924          1.0213            1.0159         3.07
IVF-OPQ-nl158-m16 (self)                               2_882.61     4_490.27     7_372.88       0.8483          1.0437            1.0305         3.07
IVF-OPQ-nl158-m32-np7 (query)                          3_082.99       753.77     3_836.76       0.9017          1.0173            1.0127         3.84
IVF-OPQ-nl158-m32-np12 (query)                         3_082.99     1_025.34     4_108.33       0.9021          1.0171            1.0126         3.84
IVF-OPQ-nl158-m32-np17 (query)                         3_082.99     1_285.37     4_368.36       0.9021          1.0171            1.0126         3.84
IVF-OPQ-nl158-m32 (self)                               3_082.99     5_667.61     8_750.61       0.8626          1.0348            1.0243         3.84
IVF-OPQ-nl158-m64-np7 (query)                          3_384.31     1_103.43     4_487.74       0.9105          1.0146            1.0099         5.36
IVF-OPQ-nl158-m64-np12 (query)                         3_384.31     1_641.77     5_026.08       0.9110          1.0144            1.0098         5.36
IVF-OPQ-nl158-m64-np17 (query)                         3_384.31     2_133.94     5_518.25       0.9110          1.0144            1.0098         5.36
IVF-OPQ-nl158-m64 (self)                               3_384.31     8_508.80    11_893.11       0.8752          1.0288            1.0194         5.36
IVF-OPQ-nl158-m128-np7 (query)                         3_949.18     1_705.82     5_655.00       0.9627          1.0027            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                        3_949.18     2_656.89     6_606.07       0.9634          1.0024            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                        3_949.18     3_581.74     7_530.93       0.9635          1.0024            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                              3_949.18    13_335.77    17_284.95       0.9463          1.0054            1.0012         8.42
IVF-OPQ-nl223-m16-np11 (query)                         3_092.80       719.67     3_812.47       0.9001          1.0182            1.0133         3.20
IVF-OPQ-nl223-m16-np14 (query)                         3_092.80       816.05     3_908.84       0.9002          1.0182            1.0133         3.20
IVF-OPQ-nl223-m16-np21 (query)                         3_092.80     1_031.24     4_124.03       0.9002          1.0182            1.0133         3.20
IVF-OPQ-nl223-m16 (self)                               3_092.80     4_824.52     7_917.32       0.8566          1.0385            1.0263         3.20
IVF-OPQ-nl223-m32-np11 (query)                         3_224.64       929.49     4_154.12       0.9080          1.0153            1.0109         3.96
IVF-OPQ-nl223-m32-np14 (query)                         3_224.64     1_082.10     4_306.74       0.9081          1.0152            1.0109         3.96
IVF-OPQ-nl223-m32-np21 (query)                         3_224.64     1_437.11     4_661.75       0.9081          1.0152            1.0109         3.96
IVF-OPQ-nl223-m32 (self)                               3_224.64     6_603.14     9_827.78       0.8697          1.0314            1.0212         3.96
IVF-OPQ-nl223-m64-np11 (query)                         3_532.74     1_442.24     4_974.98       0.9179          1.0117            1.0080         5.49
IVF-OPQ-nl223-m64-np14 (query)                         3_532.74     1_720.78     5_253.52       0.9180          1.0116            1.0080         5.49
IVF-OPQ-nl223-m64-np21 (query)                         3_532.74     2_389.22     5_921.96       0.9180          1.0116            1.0080         5.49
IVF-OPQ-nl223-m64 (self)                               3_532.74     9_438.50    12_971.24       0.8844          1.0239            1.0161         5.49
IVF-OPQ-nl223-m128-np11 (query)                        4_207.53     2_276.78     6_484.31       0.9664          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                        4_207.53     2_789.02     6_996.55       0.9665          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                        4_207.53     3_981.53     8_189.06       0.9666          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                              4_207.53    14_739.76    18_947.29       0.9487          1.0052            1.0008         8.54
IVF-OPQ-nl316-m16-np15 (query)                         3_094.50       835.94     3_930.44       0.9059          1.0164            1.0118         3.88
IVF-OPQ-nl316-m16-np17 (query)                         3_094.50       893.91     3_988.41       0.9060          1.0164            1.0118         3.88
IVF-OPQ-nl316-m16-np25 (query)                         3_094.50     1_149.33     4_243.82       0.9060          1.0164            1.0118         3.88
IVF-OPQ-nl316-m16 (self)                               3_094.50     5_203.46     8_297.96       0.8666          1.0330            1.0223         3.88
IVF-OPQ-nl316-m32-np15 (query)                         3_257.31     1_102.71     4_360.02       0.9150          1.0128            1.0089         4.65
IVF-OPQ-nl316-m32-np17 (query)                         3_257.31     1_201.25     4_458.57       0.9150          1.0128            1.0089         4.65
IVF-OPQ-nl316-m32-np25 (query)                         3_257.31     1_582.10     4_839.41       0.9150          1.0128            1.0089         4.65
IVF-OPQ-nl316-m32 (self)                               3_257.31     6_656.39     9_913.70       0.8798          1.0262            1.0176         4.65
IVF-OPQ-nl316-m64-np15 (query)                         3_568.17     1_744.63     5_312.80       0.9219          1.0105            1.0071         6.17
IVF-OPQ-nl316-m64-np17 (query)                         3_568.17     1_923.63     5_491.80       0.9219          1.0105            1.0071         6.17
IVF-OPQ-nl316-m64-np25 (query)                         3_568.17     2_646.26     6_214.43       0.9219          1.0105            1.0071         6.17
IVF-OPQ-nl316-m64 (self)                               3_568.17    10_224.83    13_793.01       0.8895          1.0216            1.0145         6.17
IVF-OPQ-nl316-m128-np15 (query)                        4_218.82     2_774.90     6_993.72       0.9684          1.0019            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                        4_218.82     3_090.88     7_309.70       0.9685          1.0019            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                        4_218.82     4_359.80     8_578.62       0.9685          1.0019            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                              4_218.82    15_997.54    20_216.36       0.9511          1.0049            1.0005         9.23
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.12     1_790.28     1_893.40       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.12     5_887.66     5_990.78       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             4_409.35     1_488.07     5_897.42       0.7383          1.1310            1.1123         3.76
Exhaustive-OPQ-m16 (self)                              4_409.35     8_158.35    12_567.70       0.6589          1.2299            1.1973         3.76
Exhaustive-OPQ-m32 (query)                             4_539.36     2_323.88     6_863.24       0.8490          1.0412            1.0319         4.53
Exhaustive-OPQ-m32 (self)                              4_539.36    10_928.75    15_468.11       0.8007          1.0713            1.0579         4.53
Exhaustive-OPQ-m64 (query)                             4_938.68     4_367.90     9_306.59       0.8797          1.0255            1.0187         6.05
Exhaustive-OPQ-m64 (self)                              4_938.68    17_658.45    22_597.13       0.8412          1.0441            1.0343         6.05
Exhaustive-OPQ-m128 (query)                            5_633.41     8_617.11    14_250.52       0.9050          1.0147            1.0105         9.11
Exhaustive-OPQ-m128 (self)                             5_633.41    31_982.90    37_616.30       0.8745          1.0261            1.0199         9.11
IVF-OPQ-nl158-m16-np7 (query)                          5_364.73     1_192.93     6_557.66       0.8938          1.0214            1.0155         4.98
IVF-OPQ-nl158-m16-np12 (query)                         5_364.73     1_368.21     6_732.93       0.8939          1.0214            1.0155         4.98
IVF-OPQ-nl158-m16-np17 (query)                         5_364.73     1_567.97     6_932.70       0.8939          1.0214            1.0155         4.98
IVF-OPQ-nl158-m16 (self)                               5_364.73     8_415.71    13_780.44       0.8497          1.0422            1.0294         4.98
IVF-OPQ-nl158-m32-np7 (query)                          4_796.48     1_341.54     6_138.02       0.9395          1.0078            1.0031         5.74
IVF-OPQ-nl158-m32-np12 (query)                         4_796.48     1_758.78     6_555.26       0.9396          1.0077            1.0030         5.74
IVF-OPQ-nl158-m32-np17 (query)                         4_796.48     2_037.45     6_833.93       0.9396          1.0077            1.0030         5.74
IVF-OPQ-nl158-m32 (self)                               4_796.48     9_913.11    14_709.59       0.9115          1.0166            1.0066         5.74
IVF-OPQ-nl158-m64-np7 (query)                          5_077.22     1_711.63     6_788.85       0.9532          1.0047            1.0009         7.27
IVF-OPQ-nl158-m64-np12 (query)                         5_077.22     2_293.51     7_370.73       0.9534          1.0047            1.0009         7.27
IVF-OPQ-nl158-m64-np17 (query)                         5_077.22     2_885.43     7_962.65       0.9534          1.0047            1.0009         7.27
IVF-OPQ-nl158-m64 (self)                               5_077.22    12_800.40    17_877.62       0.9301          1.0107            1.0030         7.27
IVF-OPQ-nl158-m128-np7 (query)                         5_576.96     2_562.40     8_139.36       0.9617          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128-np12 (query)                        5_576.96     3_742.52     9_319.48       0.9619          1.0031            1.0000        10.32
IVF-OPQ-nl158-m128-np17 (query)                        5_576.96     4_930.54    10_507.50       0.9619          1.0031            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                              5_576.96    19_593.42    25_170.38       0.9406          1.0075            1.0015        10.32
IVF-OPQ-nl223-m16-np11 (query)                         4_821.01     1_310.79     6_131.80       0.8998          1.0192            1.0137         5.17
IVF-OPQ-nl223-m16-np14 (query)                         4_821.01     1_411.12     6_232.14       0.8998          1.0192            1.0137         5.17
IVF-OPQ-nl223-m16-np21 (query)                         4_821.01     1_687.76     6_508.78       0.8998          1.0192            1.0137         5.17
IVF-OPQ-nl223-m16 (self)                               4_821.01     8_782.85    13_603.87       0.8589          1.0368            1.0255         5.17
IVF-OPQ-nl223-m32-np11 (query)                         5_028.71     1_596.51     6_625.22       0.9411          1.0074            1.0028         5.93
IVF-OPQ-nl223-m32-np14 (query)                         5_028.71     1_774.14     6_802.85       0.9411          1.0074            1.0028         5.93
IVF-OPQ-nl223-m32-np21 (query)                         5_028.71     2_224.42     7_253.12       0.9411          1.0074            1.0028         5.93
IVF-OPQ-nl223-m32 (self)                               5_028.71    10_605.08    15_633.79       0.9147          1.0151            1.0061         5.93
IVF-OPQ-nl223-m64-np11 (query)                         5_439.35     2_096.47     7_535.82       0.9547          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                         5_439.35     2_434.21     7_873.56       0.9548          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np21 (query)                         5_439.35     3_192.94     8_632.29       0.9548          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64 (self)                               5_439.35    13_829.13    19_268.48       0.9334          1.0097            1.0026         7.46
IVF-OPQ-nl223-m128-np11 (query)                        5_960.87     3_321.40     9_282.27       0.9642          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                        5_960.87     4_007.31     9_968.18       0.9642          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                        5_960.87     5_518.23    11_479.10       0.9642          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                              5_960.87    21_660.98    27_621.85       0.9436          1.0066            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                         4_930.36     1_458.13     6_388.49       0.9043          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16-np17 (query)                         4_930.36     1_523.14     6_453.49       0.9044          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16-np25 (query)                         4_930.36     1_831.19     6_761.55       0.9044          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16 (self)                               4_930.36     9_235.49    14_165.85       0.8654          1.0330            1.0227         6.19
IVF-OPQ-nl316-m32-np15 (query)                         5_112.34     1_823.12     6_935.46       0.9435          1.0068            1.0024         6.95
IVF-OPQ-nl316-m32-np17 (query)                         5_112.34     1_963.01     7_075.35       0.9435          1.0068            1.0024         6.95
IVF-OPQ-nl316-m32-np25 (query)                         5_112.34     2_437.85     7_550.19       0.9435          1.0068            1.0024         6.95
IVF-OPQ-nl316-m32 (self)                               5_112.34    11_317.96    16_430.30       0.9189          1.0138            1.0053         6.95
IVF-OPQ-nl316-m64-np15 (query)                         5_506.09     2_463.50     7_969.58       0.9560          1.0042            1.0006         8.48
IVF-OPQ-nl316-m64-np17 (query)                         5_506.09     2_668.16     8_174.25       0.9560          1.0042            1.0006         8.48
IVF-OPQ-nl316-m64-np25 (query)                         5_506.09     3_517.44     9_023.52       0.9560          1.0042            1.0006         8.48
IVF-OPQ-nl316-m64 (self)                               5_506.09    14_916.56    20_422.65       0.9358          1.0090            1.0023         8.48
IVF-OPQ-nl316-m128-np15 (query)                        6_170.17     4_052.52    10_222.70       0.9650          1.0025            1.0000        11.53
IVF-OPQ-nl316-m128-np17 (query)                        6_170.17     4_489.32    10_659.49       0.9650          1.0025            1.0000        11.53
IVF-OPQ-nl316-m128-np25 (query)                        6_170.17     6_570.72    12_740.90       0.9650          1.0025            1.0000        11.53
IVF-OPQ-nl316-m128 (self)                              6_170.17    23_611.59    29_781.76       0.9466          1.0060            1.0008        11.53
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.08     1_222.46     1_291.53       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      701.30       117.88       819.17       0.3452          1.0753            1.0741         2.24
IVFPQ-m64-nl111-np1                                      786.59       191.07       977.66       0.4425          1.0520            1.0449         3.77
SOARPQ-shift0.5-m32-nl111-np1                            804.54       128.87       933.41       0.3215          1.2782            1.0791         4.72
IVFPQ-m32-nl111-np2                                      701.30       172.61       873.91       0.3486          1.0742            1.0731         2.24
IVFPQ-m64-nl111-np2                                      786.59       301.98     1_088.57       0.4516          1.0500            1.0434         3.77
SOARPQ-shift0.5-m32-nl111-np2                            804.54       185.88       990.43       0.3475          1.0767            1.0735         4.72
IVFPQ-m32-nl111-np4                                      701.30       267.75       969.05       0.3488          1.0741            1.0730         2.24
IVFPQ-m64-nl111-np4                                      786.59       482.99     1_269.57       0.4521          1.0499            1.0433         3.77
SOARPQ-shift0.5-m32-nl111-np4                            804.54       287.77     1_092.31       0.3486          1.0743            1.0731         4.72
IVFPQ-m32-nl111-np5                                      701.30       332.99     1_034.29       0.3488          1.0741            1.0730         2.24
IVFPQ-m64-nl111-np5                                      786.59       570.17     1_356.76       0.4522          1.0499            1.0433         3.77
SOARPQ-shift0.5-m32-nl111-np5                            804.54       335.51     1_140.05       0.3488          1.0741            1.0730         4.72
IVFPQ-m32-nl111-np8                                      701.30       451.77     1_153.06       0.3488          1.0741            1.0730         2.24
IVFPQ-m64-nl111-np8                                      786.59       821.73     1_608.32       0.4522          1.0499            1.0433         3.77
SOARPQ-shift0.5-m32-nl111-np8                            804.54       501.07     1_305.61       0.3488          1.0741            1.0730         4.72
IVFPQ-m32-nl111-np10                                     701.30       538.22     1_239.52       0.3488          1.0741            1.0730         2.24
IVFPQ-m64-nl111-np10                                     786.59       995.75     1_782.34       0.4522          1.0499            1.0433         3.77
SOARPQ-shift0.5-m32-nl111-np10                           804.54       585.82     1_390.36       0.3488          1.0741            1.0730         4.72
IVFPQ-m32-nl158-np1                                      789.55       105.74       895.29       0.3529          1.0723            1.0713         2.34
IVFPQ-m64-nl158-np1                                      870.35       175.78     1_046.13       0.4474          1.0505            1.0438         3.86
SOARPQ-shift0.5-m32-nl158-np1                            919.37       123.05     1_042.42       0.3229          1.2066            1.0771         4.82
IVFPQ-m32-nl158-np2                                      789.55       161.06       950.62       0.3600          1.0695            1.0694         2.34
IVFPQ-m64-nl158-np2                                      870.35       281.42     1_151.77       0.4636          1.0464            1.0412         3.86
SOARPQ-shift0.5-m32-nl158-np2                            919.37       183.99     1_103.36       0.3579          1.0719            1.0709         4.82
IVFPQ-m32-nl158-np4                                      789.55       258.96     1_048.51       0.3608          1.0693            1.0692         2.34
IVFPQ-m64-nl158-np4                                      870.35       467.49     1_337.84       0.4661          1.0460            1.0408         3.86
SOARPQ-shift0.5-m32-nl158-np4                            919.37       280.11     1_199.48       0.3605          1.0697            1.0693         4.82
IVFPQ-m32-nl158-np7                                      789.55       396.52     1_186.08       0.3609          1.0693            1.0692         2.34
IVFPQ-m64-nl158-np7                                      870.35       719.63     1_589.98       0.4661          1.0460            1.0408         3.86
SOARPQ-shift0.5-m32-nl158-np7                            919.37       424.43     1_343.80       0.3609          1.0693            1.0692         4.82
IVFPQ-m32-nl158-np8                                      789.55       443.53     1_233.08       0.3609          1.0693            1.0692         2.34
IVFPQ-m64-nl158-np8                                      870.35       803.86     1_674.21       0.4661          1.0460            1.0408         3.86
SOARPQ-shift0.5-m32-nl158-np8                            919.37       469.78     1_389.14       0.3609          1.0693            1.0692         4.82
IVFPQ-m32-nl158-np12                                     789.55       624.73     1_414.29       0.3609          1.0693            1.0692         2.34
IVFPQ-m64-nl158-np12                                     870.35     1_137.32     2_007.67       0.4661          1.0460            1.0408         3.86
SOARPQ-shift0.5-m32-nl158-np12                           919.37       659.75     1_579.11       0.3609          1.0693            1.0692         4.82
IVFPQ-m32-nl223-np1                                      826.20        98.29       924.49       0.3569          1.0685            1.0684         2.46
IVFPQ-m64-nl223-np1                                      900.29       150.81     1_051.10       0.4445          1.0490            1.0434         3.99
SOARPQ-shift0.5-m32-nl223-np1                            969.54       120.69     1_090.23       0.3207          1.1636            1.0760         4.94
IVFPQ-m32-nl223-np2                                      826.20       153.61       979.82       0.3711          1.0638            1.0651         2.46
IVFPQ-m64-nl223-np2                                      900.29       257.72     1_158.02       0.4763          1.0424            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np2                            969.54       177.78     1_147.32       0.3662          1.0680            1.0675         4.94
IVFPQ-m32-nl223-np4                                      826.20       256.29     1_082.49       0.3732          1.0633            1.0645         2.46
IVFPQ-m64-nl223-np4                                      900.29       450.26     1_350.56       0.4831          1.0413            1.0373         3.99
SOARPQ-shift0.5-m32-nl223-np4                            969.54       278.89     1_248.43       0.3720          1.0645            1.0650         4.94
IVFPQ-m32-nl223-np8                                      826.20       439.30     1_265.50       0.3733          1.0632            1.0645         2.46
IVFPQ-m64-nl223-np8                                      900.29       788.50     1_688.80       0.4834          1.0412            1.0372         3.99
SOARPQ-shift0.5-m32-nl223-np8                            969.54       464.73     1_434.27       0.3733          1.0632            1.0645         4.94
IVFPQ-m32-nl223-np11                                     826.20       574.52     1_400.73       0.3733          1.0632            1.0645         2.46
IVFPQ-m64-nl223-np11                                     900.29     1_036.60     1_936.89       0.4834          1.0412            1.0372         3.99
SOARPQ-shift0.5-m32-nl223-np11                           969.54       608.08     1_577.62       0.3733          1.0632            1.0645         4.94
IVFPQ-m32-nl223-np14                                     826.20       717.44     1_543.64       0.3733          1.0632            1.0645         2.46
IVFPQ-m64-nl223-np14                                     900.29     1_281.36     2_181.65       0.4834          1.0412            1.0372         3.99
SOARPQ-shift0.5-m32-nl223-np14                           969.54       750.85     1_720.39       0.3733          1.0632            1.0645         4.94
IVFPQ-m32-nl316-np1                                      899.10        99.85       998.94       0.3578          1.0669            1.0670         2.65
IVFPQ-m64-nl316-np1                                      981.34       154.05     1_135.38       0.4396          1.0476            1.0440         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_066.88       112.51     1_179.39       0.3204          1.1512            1.0722         5.13
IVFPQ-m32-nl316-np2                                      899.10       149.08     1_048.18       0.3781          1.0605            1.0626         2.65
IVFPQ-m64-nl316-np2                                      981.34       253.68     1_235.02       0.4802          1.0391            1.0376         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_066.88       168.36     1_235.24       0.3719          1.0646            1.0654         5.13
IVFPQ-m32-nl316-np4                                      899.10       247.78     1_146.88       0.3827          1.0595            1.0615         2.65
IVFPQ-m64-nl316-np4                                      981.34       425.53     1_406.86       0.4931          1.0373            1.0357         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_066.88       271.85     1_338.73       0.3789          1.0620            1.0627         5.13
IVFPQ-m32-nl316-np8                                      899.10       440.42     1_339.51       0.3833          1.0594            1.0614         2.65
IVFPQ-m64-nl316-np8                                      981.34       774.91     1_756.25       0.4950          1.0369            1.0355         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_066.88       464.02     1_530.91       0.3830          1.0597            1.0615         5.13
IVFPQ-m32-nl316-np15                                     899.10       760.01     1_659.11       0.3833          1.0594            1.0614         2.65
IVFPQ-m64-nl316-np15                                     981.34     1_352.96     2_334.30       0.4950          1.0369            1.0355         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_066.88       797.99     1_864.88       0.3833          1.0594            1.0614         5.13
IVFPQ-m32-nl316-np17                                     899.10       911.40     1_810.50       0.3833          1.0594            1.0614         2.65
IVFPQ-m64-nl316-np17                                     981.34     1_527.20     2_508.54       0.4950          1.0369            1.0355         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_066.88       878.67     1_945.55       0.3833          1.0594            1.0614         5.13
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
Exhaustive (query)                                        69.08     1_222.46     1_291.53       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                          960.50       121.44     1_081.94       0.3242          1.2022            1.0770         4.82
SOARPQ-near-np2                                          960.50       180.65     1_141.15       0.3579          1.0718            1.0709         4.82
SOARPQ-near-np4                                          960.50       274.94     1_235.44       0.3605          1.0697            1.0693         4.82
SOARPQ-near-np7                                          960.50       418.36     1_378.85       0.3609          1.0693            1.0692         4.82
SOARPQ-near-np8                                          960.50       467.75     1_428.24       0.3609          1.0693            1.0692         4.82
SOARPQ-near-np12                                         960.50       659.42     1_619.92       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.3-np1                                      955.04       121.30     1_076.33       0.3234          1.2047            1.0771         4.82
SOARPQ-shift0.3-np2                                      955.04       180.49     1_135.53       0.3579          1.0719            1.0709         4.82
SOARPQ-shift0.3-np4                                      955.04       276.27     1_231.31       0.3605          1.0697            1.0693         4.82
SOARPQ-shift0.3-np7                                      955.04       418.89     1_373.93       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.3-np8                                      955.04       466.41     1_421.44       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.3-np12                                     955.04       660.72     1_615.76       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.7-np1                                      968.50       121.25     1_089.75       0.3227          1.2078            1.0771         4.82
SOARPQ-shift0.7-np2                                      968.50       180.59     1_149.09       0.3578          1.0724            1.0710         4.82
SOARPQ-shift0.7-np4                                      968.50       275.73     1_244.23       0.3604          1.0699            1.0693         4.82
SOARPQ-shift0.7-np7                                      968.50       418.77     1_387.27       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.7-np8                                      968.50       470.33     1_438.83       0.3609          1.0693            1.0692         4.82
SOARPQ-shift0.7-np12                                     968.50       660.78     1_629.28       0.3609          1.0693            1.0692         4.82
SOARPQ-orth1-np1                                         968.11       122.22     1_090.33       0.3239          1.2036            1.0770         4.82
SOARPQ-orth1-np2                                         968.11       180.79     1_148.90       0.3578          1.0714            1.0709         4.82
SOARPQ-orth1-np4                                         968.11       275.26     1_243.37       0.3605          1.0697            1.0693         4.82
SOARPQ-orth1-np7                                         968.11       420.51     1_388.62       0.3609          1.0693            1.0692         4.82
SOARPQ-orth1-np8                                         968.11       468.14     1_436.25       0.3609          1.0693            1.0692         4.82
SOARPQ-orth1-np12                                        968.11       695.99     1_664.11       0.3609          1.0693            1.0692         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.60     1_279.15     1_347.75       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      755.17       118.76       873.93       0.4655          1.0826            1.0773         2.24
IVFPQ-m64-nl111-np1                                      828.12       200.02     1_028.14       0.5931          1.0452            1.0385         3.77
SOARPQ-shift0.5-m32-nl111-np1                            872.46       126.64       999.11       0.4680          1.0948            1.0764         4.72
IVFPQ-m32-nl111-np2                                      755.17       165.83       921.00       0.4844          1.0755            1.0726         2.24
IVFPQ-m64-nl111-np2                                      828.12       295.72     1_123.83       0.6257          1.0364            1.0344         3.77
SOARPQ-shift0.5-m32-nl111-np2                            872.46       179.28     1_051.74       0.4834          1.0773            1.0730         4.72
IVFPQ-m32-nl111-np4                                      755.17       258.33     1_013.51       0.4845          1.0754            1.0726         2.24
IVFPQ-m64-nl111-np4                                      828.12       469.42     1_297.54       0.6259          1.0363            1.0344         3.77
SOARPQ-shift0.5-m32-nl111-np4                            872.46       279.42     1_151.89       0.4845          1.0754            1.0726         4.72
IVFPQ-m32-nl111-np5                                      755.17       306.53     1_061.71       0.4845          1.0754            1.0726         2.24
IVFPQ-m64-nl111-np5                                      828.12       556.00     1_384.12       0.6259          1.0363            1.0344         3.77
SOARPQ-shift0.5-m32-nl111-np5                            872.46       324.59     1_197.05       0.4845          1.0754            1.0726         4.72
IVFPQ-m32-nl111-np8                                      755.17       449.93     1_205.10       0.4845          1.0754            1.0726         2.24
IVFPQ-m64-nl111-np8                                      828.12       814.88     1_643.00       0.6259          1.0363            1.0344         3.77
SOARPQ-shift0.5-m32-nl111-np8                            872.46       488.91     1_361.38       0.4845          1.0754            1.0726         4.72
IVFPQ-m32-nl111-np10                                     755.17       546.09     1_301.27       0.4845          1.0754            1.0726         2.24
IVFPQ-m64-nl111-np10                                     828.12       988.82     1_816.94       0.6259          1.0363            1.0344         3.77
SOARPQ-shift0.5-m32-nl111-np10                           872.46       600.88     1_473.34       0.4845          1.0754            1.0726         4.72
IVFPQ-m32-nl158-np1                                      825.95       110.64       936.59       0.4507          1.0867            1.0810         2.34
IVFPQ-m64-nl158-np1                                      906.59       185.16     1_091.76       0.5617          1.0524            1.0436         3.86
SOARPQ-shift0.5-m32-nl158-np1                            960.11       124.93     1_085.04       0.4807          1.0792            1.0747         4.82
IVFPQ-m32-nl158-np2                                      825.95       166.30       992.25       0.4881          1.0733            1.0718         2.34
IVFPQ-m64-nl158-np2                                      906.59       288.67     1_195.27       0.6249          1.0362            1.0343         3.86
SOARPQ-shift0.5-m32-nl158-np2                            960.11       177.99     1_138.09       0.4903          1.0732            1.0716         4.82
IVFPQ-m32-nl158-np4                                      825.95       259.19     1_085.14       0.4913          1.0723            1.0709         2.34
IVFPQ-m64-nl158-np4                                      906.59       461.64     1_368.23       0.6315          1.0349            1.0333         3.86
SOARPQ-shift0.5-m32-nl158-np4                            960.11       272.36     1_232.47       0.4913          1.0723            1.0709         4.82
IVFPQ-m32-nl158-np7                                      825.95       396.17     1_222.12       0.4913          1.0723            1.0709         2.34
IVFPQ-m64-nl158-np7                                      906.59       713.41     1_620.00       0.6315          1.0349            1.0333         3.86
SOARPQ-shift0.5-m32-nl158-np7                            960.11       417.73     1_377.84       0.4913          1.0723            1.0709         4.82
IVFPQ-m32-nl158-np8                                      825.95       442.70     1_268.65       0.4913          1.0723            1.0709         2.34
IVFPQ-m64-nl158-np8                                      906.59       797.07     1_703.66       0.6315          1.0349            1.0333         3.86
SOARPQ-shift0.5-m32-nl158-np8                            960.11       465.96     1_426.07       0.4913          1.0723            1.0709         4.82
IVFPQ-m32-nl158-np12                                     825.95       632.87     1_458.83       0.4913          1.0723            1.0709         2.34
IVFPQ-m64-nl158-np12                                     906.59     1_142.12     2_048.71       0.6315          1.0349            1.0333         3.86
SOARPQ-shift0.5-m32-nl158-np12                           960.11       664.32     1_624.42       0.4913          1.0723            1.0709         4.82
IVFPQ-m32-nl223-np1                                      862.88       106.19       969.07       0.4066          1.1013            1.0959         2.46
IVFPQ-m64-nl223-np1                                      935.61       162.68     1_098.29       0.4893          1.0696            1.0618         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_005.96       119.58     1_125.54       0.4648          1.0813            1.0790         4.94
IVFPQ-m32-nl223-np2                                      862.88       160.43     1_023.32       0.4684          1.0788            1.0771         2.46
IVFPQ-m64-nl223-np2                                      935.61       266.76     1_202.37       0.5916          1.0430            1.0398         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_005.96       176.57     1_182.53       0.4845          1.0744            1.0728         4.94
IVFPQ-m32-nl223-np4                                      862.88       260.43     1_123.32       0.4891          1.0725            1.0711         2.46
IVFPQ-m64-nl223-np4                                      935.61       448.14     1_383.74       0.6300          1.0350            1.0334         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_005.96       274.53     1_280.49       0.4906          1.0722            1.0705         4.94
IVFPQ-m32-nl223-np8                                      862.88       444.51     1_307.39       0.4908          1.0720            1.0705         2.46
IVFPQ-m64-nl223-np8                                      935.61       786.57     1_722.18       0.6333          1.0343            1.0327         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_005.96       464.64     1_470.60       0.4908          1.0720            1.0705         4.94
IVFPQ-m32-nl223-np11                                     862.88       583.17     1_446.05       0.4908          1.0720            1.0705         2.46
IVFPQ-m64-nl223-np11                                     935.61     1_040.20     1_975.81       0.6333          1.0343            1.0327         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_005.96       617.09     1_623.05       0.4908          1.0720            1.0705         4.94
IVFPQ-m32-nl223-np14                                     862.88       727.85     1_590.74       0.4908          1.0720            1.0705         2.46
IVFPQ-m64-nl223-np14                                     935.61     1_297.51     2_233.12       0.6333          1.0343            1.0327         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_005.96       744.71     1_750.67       0.4908          1.0720            1.0705         4.94
IVFPQ-m32-nl316-np1                                      896.26       104.12     1_000.38       0.3502          1.1213            1.1184         2.65
IVFPQ-m64-nl316-np1                                      995.41       144.56     1_139.96       0.3988          1.0925            1.0871         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_079.75       108.55     1_188.30       0.4262          1.0932            1.0924         5.13
IVFPQ-m32-nl316-np2                                      896.26       151.41     1_047.67       0.4262          1.0916            1.0907         2.65
IVFPQ-m64-nl316-np2                                      995.41       239.76     1_235.17       0.5213          1.0576            1.0551         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_079.75       165.27     1_245.03       0.4657          1.0802            1.0791         5.13
IVFPQ-m32-nl316-np4                                      896.26       252.44     1_148.70       0.4737          1.0772            1.0760         2.65
IVFPQ-m64-nl316-np4                                      995.41       428.53     1_423.93       0.6052          1.0397            1.0378         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_079.75       274.06     1_353.81       0.4837          1.0745            1.0734         5.13
IVFPQ-m32-nl316-np8                                      896.26       460.26     1_356.52       0.4869          1.0734            1.0721         2.65
IVFPQ-m64-nl316-np8                                      995.41       777.00     1_772.41       0.6333          1.0342            1.0327         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_079.75       457.90     1_537.66       0.4871          1.0733            1.0721         5.13
IVFPQ-m32-nl316-np15                                     896.26       779.65     1_675.91       0.4874          1.0732            1.0720         2.65
IVFPQ-m64-nl316-np15                                     995.41     1_365.68     2_361.08       0.6347          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_079.75       800.76     1_880.52       0.4874          1.0732            1.0720         5.13
IVFPQ-m32-nl316-np17                                     896.26       874.25     1_770.51       0.4874          1.0732            1.0720         2.65
IVFPQ-m64-nl316-np17                                     995.41     1_538.13     2_533.53       0.6347          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_079.75       890.53     1_970.28       0.4874          1.0732            1.0720         5.13
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
Exhaustive (query)                                        68.60     1_279.15     1_347.75       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                          987.81       123.73     1_111.54       0.4793          1.0798            1.0749         4.82
SOARPQ-near-np2                                          987.81       176.09     1_163.90       0.4903          1.0732            1.0716         4.82
SOARPQ-near-np4                                          987.81       269.21     1_257.02       0.4913          1.0723            1.0709         4.82
SOARPQ-near-np7                                          987.81       412.15     1_399.96       0.4913          1.0723            1.0709         4.82
SOARPQ-near-np8                                          987.81       460.39     1_448.20       0.4913          1.0723            1.0709         4.82
SOARPQ-near-np12                                         987.81       657.37     1_645.18       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.3-np1                                      967.82       125.02     1_092.84       0.4804          1.0793            1.0747         4.82
SOARPQ-shift0.3-np2                                      967.82       176.07     1_143.89       0.4903          1.0732            1.0716         4.82
SOARPQ-shift0.3-np4                                      967.82       270.34     1_238.17       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.3-np7                                      967.82       411.27     1_379.09       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.3-np8                                      967.82       461.12     1_428.94       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.3-np12                                     967.82       663.82     1_631.64       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.7-np1                                      968.90       127.23     1_096.13       0.4806          1.0792            1.0747         4.82
SOARPQ-shift0.7-np2                                      968.90       176.23     1_145.13       0.4903          1.0732            1.0717         4.82
SOARPQ-shift0.7-np4                                      968.90       269.72     1_238.62       0.4913          1.0724            1.0709         4.82
SOARPQ-shift0.7-np7                                      968.90       411.53     1_380.43       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.7-np8                                      968.90       463.88     1_432.78       0.4913          1.0723            1.0709         4.82
SOARPQ-shift0.7-np12                                     968.90       663.94     1_632.84       0.4913          1.0723            1.0709         4.82
SOARPQ-orth1-np1                                         986.59       126.51     1_113.10       0.4805          1.0793            1.0747         4.82
SOARPQ-orth1-np2                                         986.59       175.12     1_161.71       0.4903          1.0731            1.0716         4.82
SOARPQ-orth1-np4                                         986.59       268.47     1_255.06       0.4913          1.0723            1.0709         4.82
SOARPQ-orth1-np7                                         986.59       414.64     1_401.23       0.4913          1.0723            1.0709         4.82
SOARPQ-orth1-np8                                         986.59       460.48     1_447.07       0.4913          1.0723            1.0709         4.82
SOARPQ-orth1-np12                                        986.59       661.65     1_648.24       0.4913          1.0723            1.0709         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.69     1_302.94     1_372.63       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                      762.54        98.06       860.60       0.7152          1.1719            1.0908         2.24
IVFPQ-m64-nl111-np1                                      846.73       153.26     1_000.00       0.7260          1.1621            1.0808         3.77
SOARPQ-shift0.5-m32-nl111-np1                            854.32       115.95       970.27       0.8206          1.0707            1.0427         4.72
IVFPQ-m32-nl111-np2                                      762.54       154.53       917.07       0.8290          1.0585            1.0378         2.24
IVFPQ-m64-nl111-np2                                      846.73       269.27     1_116.00       0.8496          1.0474            1.0254         3.77
SOARPQ-shift0.5-m32-nl111-np2                            854.32       193.55     1_047.87       0.8529          1.0442            1.0319         4.72
IVFPQ-m32-nl111-np4                                      762.54       270.95     1_033.49       0.8589          1.0366            1.0290         2.24
IVFPQ-m64-nl111-np4                                      846.73       500.11     1_346.84       0.8828          1.0249            1.0183         3.77
SOARPQ-shift0.5-m32-nl111-np4                            854.32       341.38     1_195.70       0.8602          1.0382            1.0291         4.72
IVFPQ-m32-nl111-np5                                      762.54       330.86     1_093.40       0.8607          1.0355            1.0284         2.24
IVFPQ-m64-nl111-np5                                      846.73       625.44     1_472.17       0.8848          1.0238            1.0177         3.77
SOARPQ-shift0.5-m32-nl111-np5                            854.32       413.33     1_267.65       0.8608          1.0373            1.0288         4.72
IVFPQ-m32-nl111-np8                                      762.54       515.21     1_277.75       0.8618          1.0350            1.0280         2.24
IVFPQ-m64-nl111-np8                                      846.73     1_017.43     1_864.16       0.8860          1.0232            1.0175         3.77
SOARPQ-shift0.5-m32-nl111-np8                            854.32       619.72     1_474.04       0.8617          1.0357            1.0282         4.72
IVFPQ-m32-nl111-np10                                     762.54       658.37     1_420.91       0.8619          1.0349            1.0280         2.24
IVFPQ-m64-nl111-np10                                     846.73     1_263.59     2_110.32       0.8861          1.0232            1.0174         3.77
SOARPQ-shift0.5-m32-nl111-np10                           854.32       752.94     1_607.26       0.8619          1.0352            1.0280         4.72
IVFPQ-m32-nl158-np1                                      820.41        97.07       917.48       0.7053          1.1799            1.1035         2.34
IVFPQ-m64-nl158-np1                                      917.47       143.44     1_060.91       0.7138          1.1722            1.0956         3.86
SOARPQ-shift0.5-m32-nl158-np1                            952.55       107.93     1_060.48       0.8204          1.0729            1.0427         4.82
IVFPQ-m32-nl158-np2                                      820.41       146.20       966.61       0.8300          1.0593            1.0354         2.34
IVFPQ-m64-nl158-np2                                      917.47       252.98     1_170.45       0.8470          1.0506            1.0263         3.86
SOARPQ-shift0.5-m32-nl158-np2                            952.55       174.57     1_127.12       0.8609          1.0408            1.0280         4.82
IVFPQ-m32-nl158-np4                                      820.41       255.50     1_075.90       0.8678          1.0324            1.0246         2.34
IVFPQ-m64-nl158-np4                                      917.47       456.81     1_374.28       0.8879          1.0234            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np4                            952.55       305.11     1_257.66       0.8701          1.0335            1.0248         4.82
IVFPQ-m32-nl158-np7                                      820.41       426.25     1_246.66       0.8722          1.0301            1.0236         2.34
IVFPQ-m64-nl158-np7                                      917.47       774.55     1_692.02       0.8927          1.0210            1.0155         3.86
SOARPQ-shift0.5-m32-nl158-np7                            952.55       499.12     1_451.67       0.8720          1.0311            1.0239         4.82
IVFPQ-m32-nl158-np8                                      820.41       486.13     1_306.54       0.8724          1.0300            1.0235         2.34
IVFPQ-m64-nl158-np8                                      917.47       883.60     1_801.07       0.8929          1.0209            1.0154         3.86
SOARPQ-shift0.5-m32-nl158-np8                            952.55       563.16     1_515.71       0.8721          1.0308            1.0238         4.82
IVFPQ-m32-nl158-np12                                     820.41       707.26     1_527.67       0.8725          1.0300            1.0235         2.34
IVFPQ-m64-nl158-np12                                     917.47     1_298.30     2_215.77       0.8931          1.0208            1.0153         3.86
SOARPQ-shift0.5-m32-nl158-np12                           952.55       810.52     1_763.07       0.8724          1.0302            1.0235         4.82
IVFPQ-m32-nl223-np1                                      867.51        97.80       965.30       0.6921          1.1905            1.1182         2.46
IVFPQ-m64-nl223-np1                                      957.34       138.54     1_095.88       0.6987          1.1833            1.1111         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_012.69       104.07     1_116.76       0.8163          1.0795            1.0457         4.94
IVFPQ-m32-nl223-np2                                      867.51       143.00     1_010.51       0.8302          1.0608            1.0354         2.46
IVFPQ-m64-nl223-np2                                      957.34       234.80     1_192.14       0.8444          1.0531            1.0266         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_012.69       163.04     1_175.73       0.8673          1.0408            1.0255         4.94
IVFPQ-m32-nl223-np4                                      867.51       246.01     1_113.52       0.8770          1.0286            1.0206         2.46
IVFPQ-m64-nl223-np4                                      957.34       429.32     1_386.66       0.8951          1.0206            1.0142         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_012.69       281.40     1_294.09       0.8788          1.0317            1.0210         4.94
IVFPQ-m32-nl223-np8                                      867.51       458.49     1_326.00       0.8831          1.0253            1.0190         2.46
IVFPQ-m64-nl223-np8                                      957.34       820.67     1_778.01       0.9021          1.0172            1.0128         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_012.69       545.83     1_558.51       0.8825          1.0270            1.0194         4.94
IVFPQ-m32-nl223-np11                                     867.51       618.12     1_485.63       0.8835          1.0251            1.0189         2.46
IVFPQ-m64-nl223-np11                                     957.34     1_114.76     2_072.10       0.9025          1.0170            1.0127         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_012.69       694.36     1_707.05       0.8832          1.0259            1.0191         4.94
IVFPQ-m32-nl223-np14                                     867.51       779.35     1_646.86       0.8836          1.0251            1.0188         2.46
IVFPQ-m64-nl223-np14                                     957.34     1_406.42     2_363.76       0.9026          1.0170            1.0126         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_012.69       869.24     1_881.93       0.8835          1.0254            1.0189         4.94
IVFPQ-m32-nl316-np1                                      905.03       102.36     1_007.39       0.6743          1.2059            1.1402         2.65
IVFPQ-m64-nl316-np1                                    1_016.89       139.27     1_156.16       0.6787          1.2003            1.1353         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_074.32       104.63     1_178.95       0.8095          1.0848            1.0492         5.13
IVFPQ-m32-nl316-np2                                      905.03       143.42     1_048.45       0.8249          1.0652            1.0373         2.65
IVFPQ-m64-nl316-np2                                    1_016.89       230.59     1_247.48       0.8350          1.0594            1.0302         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_074.32       158.53     1_232.85       0.8720          1.0397            1.0237         5.13
IVFPQ-m32-nl316-np4                                      905.03       243.01     1_148.04       0.8833          1.0261            1.0182         2.65
IVFPQ-m64-nl316-np4                                    1_016.89       415.59     1_432.48       0.8979          1.0200            1.0129         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_074.32       269.95     1_344.27       0.8862          1.0293            1.0184         5.13
IVFPQ-m32-nl316-np8                                      905.03       443.21     1_348.24       0.8918          1.0216            1.0159         2.65
IVFPQ-m64-nl316-np8                                    1_016.89       779.01     1_795.90       0.9071          1.0153            1.0110         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_074.32       487.36     1_561.68       0.8908          1.0239            1.0166         5.13
IVFPQ-m32-nl316-np15                                     905.03       804.38     1_709.41       0.8924          1.0213            1.0158         2.65
IVFPQ-m64-nl316-np15                                   1_016.89     1_429.58     2_446.47       0.9077          1.0150            1.0109         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_074.32       873.70     1_948.02       0.8921          1.0217            1.0159         5.13
IVFPQ-m32-nl316-np17                                     905.03       910.83     1_815.86       0.8924          1.0213            1.0158         2.65
IVFPQ-m64-nl316-np17                                   1_016.89     1_615.29     2_632.18       0.9077          1.0150            1.0109         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_074.32       987.60     2_061.93       0.8922          1.0215            1.0158         5.13
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
Exhaustive (query)                                        69.69     1_302.94     1_372.63       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                          981.46       111.32     1_092.78       0.8202          1.0728            1.0407         4.82
SOARPQ-near-np2                                          981.46       175.05     1_156.50       0.8617          1.0389            1.0269         4.82
SOARPQ-near-np4                                          981.46       307.13     1_288.58       0.8709          1.0322            1.0242         4.82
SOARPQ-near-np7                                          981.46       499.84     1_481.29       0.8722          1.0305            1.0236         4.82
SOARPQ-near-np8                                          981.46       561.23     1_542.68       0.8723          1.0304            1.0236         4.82
SOARPQ-near-np12                                         981.46       807.10     1_788.56       0.8725          1.0300            1.0235         4.82
SOARPQ-shift0.3-np1                                      948.08       110.19     1_058.27       0.8225          1.0707            1.0413         4.82
SOARPQ-shift0.3-np2                                      948.08       174.76     1_122.84       0.8618          1.0398            1.0274         4.82
SOARPQ-shift0.3-np4                                      948.08       308.20     1_256.28       0.8706          1.0330            1.0246         4.82
SOARPQ-shift0.3-np7                                      948.08       500.21     1_448.29       0.8721          1.0309            1.0238         4.82
SOARPQ-shift0.3-np8                                      948.08       563.88     1_511.96       0.8722          1.0306            1.0237         4.82
SOARPQ-shift0.3-np12                                     948.08       805.55     1_753.64       0.8725          1.0302            1.0235         4.82
SOARPQ-shift0.7-np1                                      949.76       110.34     1_060.11       0.8173          1.0760            1.0442         4.82
SOARPQ-shift0.7-np2                                      949.76       174.17     1_123.94       0.8596          1.0422            1.0284         4.82
SOARPQ-shift0.7-np4                                      949.76       307.48     1_257.24       0.8697          1.0341            1.0249         4.82
SOARPQ-shift0.7-np7                                      949.76       499.29     1_449.05       0.8719          1.0314            1.0240         4.82
SOARPQ-shift0.7-np8                                      949.76       564.04     1_513.81       0.8720          1.0310            1.0239         4.82
SOARPQ-shift0.7-np12                                     949.76       805.88     1_755.64       0.8724          1.0302            1.0235         4.82
SOARPQ-orth1-np1                                         987.07       111.34     1_098.41       0.8200          1.0737            1.0415         4.82
SOARPQ-orth1-np2                                         987.07       175.18     1_162.25       0.8611          1.0404            1.0276         4.82
SOARPQ-orth1-np4                                         987.07       306.27     1_293.34       0.8704          1.0331            1.0245         4.82
SOARPQ-orth1-np7                                         987.07       500.68     1_487.75       0.8721          1.0309            1.0238         4.82
SOARPQ-orth1-np8                                         987.07       598.90     1_585.98       0.8722          1.0307            1.0237         4.82
SOARPQ-orth1-np12                                        987.07       806.92     1_793.99       0.8725          1.0302            1.0235         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-PQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-PQ vs IVF-PQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        72.28     1_335.33     1_407.61       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                      776.30        93.89       870.19       0.7692          1.1385            1.0628         2.24
IVFPQ-m64-nl111-np1                                      855.28       144.44       999.72       0.7772          1.1328            1.0551         3.77
SOARPQ-orth1-m32-nl111-np1                               901.71       108.42     1_010.13       0.8481          1.0720            1.0349         4.72
IVFPQ-m32-nl111-np2                                      776.30       144.77       921.07       0.8683          1.0418            1.0256         2.24
IVFPQ-m64-nl111-np2                                      855.28       254.05     1_109.33       0.8804          1.0361            1.0196         3.77
SOARPQ-orth1-m32-nl111-np2                               901.71       176.07     1_077.78       0.8782          1.0426            1.0246         4.72
IVFPQ-m32-nl111-np4                                      776.30       256.25     1_032.55       0.8865          1.0284            1.0211         2.24
IVFPQ-m64-nl111-np4                                      855.28       465.63     1_320.91       0.8995          1.0228            1.0160         3.77
SOARPQ-orth1-m32-nl111-np4                               901.71       314.45     1_216.17       0.8856          1.0325            1.0219         4.72
IVFPQ-m32-nl111-np5                                      776.30       313.08     1_089.38       0.8876          1.0278            1.0207         2.24
IVFPQ-m64-nl111-np5                                      855.28       573.07     1_428.35       0.9005          1.0223            1.0157         3.77
SOARPQ-orth1-m32-nl111-np5                               901.71       377.34     1_279.05       0.8867          1.0305            1.0213         4.72
IVFPQ-m32-nl111-np8                                      776.30       486.93     1_263.23       0.8882          1.0275            1.0206         2.24
IVFPQ-m64-nl111-np8                                      855.28       918.23     1_773.51       0.9012          1.0219            1.0156         3.77
SOARPQ-orth1-m32-nl111-np8                               901.71       582.20     1_483.92       0.8879          1.0282            1.0207         4.72
IVFPQ-m32-nl111-np10                                     776.30       604.37     1_380.67       0.8882          1.0274            1.0205         2.24
IVFPQ-m64-nl111-np10                                     855.28     1_122.18     1_977.46       0.9012          1.0219            1.0156         3.77
SOARPQ-orth1-m32-nl111-np10                              901.71       714.15     1_615.86       0.8881          1.0277            1.0206         4.72
IVFPQ-m32-nl158-np1                                      855.53        93.27       948.80       0.7498          1.1591            1.0787         2.34
IVFPQ-m64-nl158-np1                                      948.49       137.20     1_085.69       0.7561          1.1537            1.0723         3.86
SOARPQ-orth1-m32-nl158-np1                             1_015.72       102.12     1_117.84       0.8431          1.0769            1.0377         4.82
IVFPQ-m32-nl158-np2                                      855.53       140.41       995.94       0.8641          1.0468            1.0263         2.34
IVFPQ-m64-nl158-np2                                      948.49       236.25     1_184.74       0.8743          1.0415            1.0210         3.86
SOARPQ-orth1-m32-nl158-np2                             1_015.72       166.28     1_182.00       0.8819          1.0412            1.0233         4.82
IVFPQ-m32-nl158-np4                                      855.53       245.73     1_101.26       0.8909          1.0263            1.0194         2.34
IVFPQ-m64-nl158-np4                                      948.49       434.64     1_383.13       0.9031          1.0208            1.0145         3.86
SOARPQ-orth1-m32-nl158-np4                             1_015.72       286.37     1_302.09       0.8903          1.0310            1.0202         4.82
IVFPQ-m32-nl158-np7                                      855.53       407.62     1_263.15       0.8930          1.0252            1.0189         2.34
IVFPQ-m64-nl158-np7                                      948.49       737.34     1_685.83       0.9054          1.0197            1.0140         3.86
SOARPQ-orth1-m32-nl158-np7                             1_015.72       471.59     1_487.31       0.8925          1.0268            1.0192         4.82
IVFPQ-m32-nl158-np8                                      855.53       465.41     1_320.94       0.8931          1.0252            1.0189         2.34
IVFPQ-m64-nl158-np8                                      948.49       842.45     1_790.95       0.9056          1.0197            1.0139         3.86
SOARPQ-orth1-m32-nl158-np8                             1_015.72       532.22     1_547.94       0.8927          1.0263            1.0191         4.82
IVFPQ-m32-nl158-np12                                     855.53       688.98     1_544.51       0.8931          1.0251            1.0188         2.34
IVFPQ-m64-nl158-np12                                     948.49     1_251.24     2_199.73       0.9057          1.0196            1.0139         3.86
SOARPQ-orth1-m32-nl158-np12                            1_015.72       784.47     1_800.19       0.8931          1.0253            1.0189         4.82
IVFPQ-m32-nl223-np1                                      898.22        95.50       993.72       0.7294          1.1793            1.1017         2.46
IVFPQ-m64-nl223-np1                                      968.01       134.07     1_102.08       0.7345          1.1745            1.0950         3.99
SOARPQ-orth1-m32-nl223-np1                             1_079.95        99.86     1_179.81       0.8375          1.0792            1.0390         4.94
IVFPQ-m32-nl223-np2                                      898.22       138.54     1_036.76       0.8598          1.0505            1.0273         2.46
IVFPQ-m64-nl223-np2                                      968.01       227.14     1_195.15       0.8698          1.0453            1.0215         3.99
SOARPQ-orth1-m32-nl223-np2                             1_079.95       156.41     1_236.36       0.8844          1.0391            1.0216         4.94
IVFPQ-m32-nl223-np4                                      898.22       239.42     1_137.64       0.8955          1.0240            1.0173         2.46
IVFPQ-m64-nl223-np4                                      968.01       415.06     1_383.07       0.9083          1.0185            1.0126         3.99
SOARPQ-orth1-m32-nl223-np4                             1_079.95       279.03     1_358.98       0.8948          1.0289            1.0180         4.94
IVFPQ-m32-nl223-np8                                      898.22       444.67     1_342.89       0.8984          1.0224            1.0166         2.46
IVFPQ-m64-nl223-np8                                      968.01       791.83     1_759.84       0.9116          1.0168            1.0119         3.99
SOARPQ-orth1-m32-nl223-np8                             1_079.95       495.21     1_575.16       0.8980          1.0237            1.0169         4.94
IVFPQ-m32-nl223-np11                                     898.22       605.53     1_503.75       0.8986          1.0223            1.0166         2.46
IVFPQ-m64-nl223-np11                                     968.01     1_081.00     2_049.01       0.9118          1.0168            1.0119         3.99
SOARPQ-orth1-m32-nl223-np11                            1_079.95       671.99     1_751.94       0.8984          1.0228            1.0167         4.94
IVFPQ-m32-nl223-np14                                     898.22       764.85     1_663.07       0.8986          1.0223            1.0166         2.46
IVFPQ-m64-nl223-np14                                     968.01     1_367.00     2_335.01       0.9118          1.0168            1.0119         3.99
SOARPQ-orth1-m32-nl223-np14                            1_079.95       848.06     1_928.01       0.8986          1.0225            1.0166         4.94
IVFPQ-m32-nl316-np1                                      969.49       101.23     1_070.72       0.7028          1.2100            1.1294         2.65
IVFPQ-m64-nl316-np1                                    1_032.52       134.88     1_167.39       0.7064          1.2057            1.1253         4.17
SOARPQ-orth1-m32-nl316-np1                             1_188.92       101.85     1_290.77       0.8276          1.0861            1.0428         5.13
IVFPQ-m32-nl316-np2                                      969.49       140.48     1_109.97       0.8497          1.0588            1.0299         2.65
IVFPQ-m64-nl316-np2                                    1_032.52       224.63     1_257.15       0.8576          1.0543            1.0245         4.17
SOARPQ-orth1-m32-nl316-np2                             1_188.92       153.14     1_342.06       0.8875          1.0362            1.0203         5.13
IVFPQ-m32-nl316-np4                                      969.49       237.16     1_206.65       0.8998          1.0219            1.0153         2.65
IVFPQ-m64-nl316-np4                                    1_032.52       403.94     1_436.46       0.9104          1.0175            1.0115         4.17
SOARPQ-orth1-m32-nl316-np4                             1_188.92       259.09     1_448.02       0.9009          1.0258            1.0159         5.13
IVFPQ-m32-nl316-np8                                      969.49       443.73     1_413.22       0.9045          1.0193            1.0143         2.65
IVFPQ-m64-nl316-np8                                    1_032.52       764.61     1_797.13       0.9154          1.0150            1.0105         4.17
SOARPQ-orth1-m32-nl316-np8                             1_188.92       472.13     1_661.05       0.9040          1.0211            1.0146         5.13
IVFPQ-m32-nl316-np15                                     969.49       820.70     1_790.19       0.9047          1.0192            1.0143         2.65
IVFPQ-m64-nl316-np15                                   1_032.52     1_418.98     2_451.49       0.9156          1.0149            1.0105         4.17
SOARPQ-orth1-m32-nl316-np15                            1_188.92       853.32     2_042.25       0.9046          1.0195            1.0143         5.13
IVFPQ-m32-nl316-np17                                     969.49       897.35     1_866.84       0.9047          1.0192            1.0143         2.65
IVFPQ-m64-nl316-np17                                   1_032.52     1_587.06     2_619.58       0.9156          1.0149            1.0105         4.17
SOARPQ-orth1-m32-nl316-np17                            1_188.92       964.70     2_153.62       0.9047          1.0193            1.0143         5.13
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
Exhaustive (query)                                        72.28     1_335.33     1_407.61       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        1_011.32       107.36     1_118.69       0.8492          1.0661            1.0349         4.82
SOARPQ-near-np2                                        1_011.32       165.72     1_177.04       0.8840          1.0355            1.0225         4.82
SOARPQ-near-np4                                        1_011.32       285.05     1_296.37       0.8911          1.0283            1.0198         4.82
SOARPQ-near-np7                                        1_011.32       470.61     1_481.93       0.8928          1.0258            1.0190         4.82
SOARPQ-near-np8                                        1_011.32       531.23     1_542.55       0.8929          1.0256            1.0190         4.82
SOARPQ-near-np12                                       1_011.32       783.41     1_794.73       0.8931          1.0252            1.0189         4.82
SOARPQ-shift0.3-np1                                      985.85       105.30     1_091.15       0.8481          1.0701            1.0364         4.82
SOARPQ-shift0.3-np2                                      985.85       163.87     1_149.72       0.8823          1.0393            1.0233         4.82
SOARPQ-shift0.3-np4                                      985.85       284.26     1_270.11       0.8900          1.0303            1.0203         4.82
SOARPQ-shift0.3-np7                                      985.85       467.88     1_453.73       0.8924          1.0264            1.0192         4.82
SOARPQ-shift0.3-np8                                      985.85       531.10     1_516.95       0.8927          1.0259            1.0191         4.82
SOARPQ-shift0.3-np12                                     985.85       776.85     1_762.69       0.8931          1.0252            1.0189         4.82
SOARPQ-shift0.7-np1                                      990.14       103.53     1_093.66       0.8431          1.0777            1.0385         4.82
SOARPQ-shift0.7-np2                                      990.14       163.51     1_153.64       0.8794          1.0443            1.0240         4.82
SOARPQ-shift0.7-np4                                      990.14       285.41     1_275.55       0.8887          1.0331            1.0207         4.82
SOARPQ-shift0.7-np7                                      990.14       468.42     1_458.56       0.8920          1.0275            1.0193         4.82
SOARPQ-shift0.7-np8                                      990.14       530.69     1_520.83       0.8924          1.0267            1.0192         4.82
SOARPQ-shift0.7-np12                                     990.14       777.49     1_767.63       0.8930          1.0254            1.0189         4.82
SOARPQ-orth1-np1                                       1_021.72       105.17     1_126.89       0.8431          1.0769            1.0377         4.82
SOARPQ-orth1-np2                                       1_021.72       164.06     1_185.77       0.8819          1.0412            1.0233         4.82
SOARPQ-orth1-np4                                       1_021.72       285.37     1_307.08       0.8903          1.0310            1.0202         4.82
SOARPQ-orth1-np7                                       1_021.72       469.52     1_491.24       0.8925          1.0268            1.0192         4.82
SOARPQ-orth1-np8                                       1_021.72       533.02     1_554.74       0.8927          1.0263            1.0191         4.82
SOARPQ-orth1-np12                                      1_021.72       779.31     1_801.03       0.8931          1.0253            1.0189         4.82
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### SOAR-OPQ

<details>
<summary><b>SOAR-OPQ - Euclidean (Correlated, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.21     1_215.61     1_283.82       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_720.82       439.08     3_159.90       0.3567          1.0704            1.0691         3.49
IVFOPQ-m64-nl111-np1                                   3_042.85       557.44     3_600.29       0.4538          1.0470            1.0431         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         2_892.50       456.65     3_349.15       0.3382          1.2162            1.0723         5.97
IVFOPQ-m32-nl111-np2                                   2_720.82       495.40     3_216.22       0.3608          1.0690            1.0680         3.49
IVFOPQ-m64-nl111-np2                                   3_042.85       631.31     3_674.16       0.4632          1.0448            1.0415         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         2_892.50       511.82     3_404.32       0.3598          1.0711            1.0683         5.97
IVFOPQ-m32-nl111-np4                                   2_720.82       594.44     3_315.26       0.3610          1.0689            1.0680         3.49
IVFOPQ-m64-nl111-np4                                   3_042.85       816.48     3_859.33       0.4638          1.0447            1.0414         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         2_892.50       615.87     3_508.38       0.3609          1.0690            1.0680         5.97
IVFOPQ-m32-nl111-np5                                   2_720.82       641.60     3_362.41       0.3610          1.0689            1.0680         3.49
IVFOPQ-m64-nl111-np5                                   3_042.85       905.14     3_947.99       0.4638          1.0447            1.0414         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         2_892.50       675.62     3_568.12       0.3610          1.0689            1.0680         5.97
IVFOPQ-m32-nl111-np8                                   2_720.82       788.56     3_509.38       0.3610          1.0689            1.0680         3.49
IVFOPQ-m64-nl111-np8                                   3_042.85     1_172.83     4_215.68       0.4638          1.0447            1.0414         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         2_892.50       818.89     3_711.39       0.3610          1.0689            1.0680         5.97
IVFOPQ-m32-nl111-np10                                  2_720.82       880.80     3_601.62       0.3610          1.0689            1.0680         3.49
IVFOPQ-m64-nl111-np10                                  3_042.85     1_347.49     4_390.34       0.4638          1.0447            1.0414         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        2_892.50       924.60     3_817.10       0.3610          1.0689            1.0680         5.97
IVFOPQ-m32-nl158-np1                                   3_078.74       432.30     3_511.04       0.3623          1.0676            1.0672         3.84
IVFOPQ-m64-nl158-np1                                   3_393.36       504.17     3_897.53       0.4572          1.0458            1.0422         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         3_243.32       449.45     3_692.77       0.3401          1.1445            1.0712         6.32
IVFOPQ-m32-nl158-np2                                   3_078.74       489.49     3_568.23       0.3706          1.0645            1.0653         3.84
IVFOPQ-m64-nl158-np2                                   3_393.36       615.89     4_009.25       0.4742          1.0414            1.0397         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         3_243.32       512.02     3_755.35       0.3690          1.0659            1.0662         6.32
IVFOPQ-m32-nl158-np4                                   3_078.74       590.83     3_669.57       0.3714          1.0643            1.0651         3.84
IVFOPQ-m64-nl158-np4                                   3_393.36       806.60     4_199.97       0.4765          1.0410            1.0394         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         3_243.32       610.24     3_853.57       0.3712          1.0644            1.0651         6.32
IVFOPQ-m32-nl158-np7                                   3_078.74       732.63     3_811.36       0.3714          1.0642            1.0651         3.84
IVFOPQ-m64-nl158-np7                                   3_393.36     1_069.56     4_462.92       0.4765          1.0410            1.0394         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         3_243.32       757.61     4_000.94       0.3714          1.0642            1.0651         6.32
IVFOPQ-m32-nl158-np8                                   3_078.74       782.99     3_861.72       0.3714          1.0642            1.0651         3.84
IVFOPQ-m64-nl158-np8                                   3_393.36     1_160.78     4_554.14       0.4765          1.0410            1.0394         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         3_243.32       801.05     4_044.37       0.3714          1.0642            1.0651         6.32
IVFOPQ-m32-nl158-np12                                  3_078.74       973.75     4_052.49       0.3714          1.0642            1.0651         3.84
IVFOPQ-m64-nl158-np12                                  3_393.36     1_504.13     4_897.49       0.4765          1.0410            1.0394         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        3_243.32       994.20     4_237.52       0.3714          1.0642            1.0651         6.32
IVFOPQ-m32-nl223-np1                                   3_316.92       426.97     3_743.89       0.3637          1.0657            1.0659         3.96
IVFOPQ-m64-nl223-np1                                   3_675.66       481.96     4_157.61       0.4528          1.0457            1.0425         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         3_476.77       442.06     3_918.83       0.3376          1.1180            1.0711         6.44
IVFOPQ-m32-nl223-np2                                   3_316.92       483.05     3_799.97       0.3792          1.0607            1.0625         3.96
IVFOPQ-m64-nl223-np2                                   3_675.66       591.13     4_266.79       0.4857          1.0388            1.0377         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         3_476.77       503.93     3_980.70       0.3752          1.0635            1.0642         6.44
IVFOPQ-m32-nl223-np4                                   3_316.92       586.57     3_903.49       0.3813          1.0601            1.0619         3.96
IVFOPQ-m64-nl223-np4                                   3_675.66       786.92     4_462.58       0.4923          1.0378            1.0366         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         3_476.77       605.96     4_082.73       0.3804          1.0609            1.0622         6.44
IVFOPQ-m32-nl223-np8                                   3_316.92       776.69     4_093.61       0.3815          1.0601            1.0619         3.96
IVFOPQ-m64-nl223-np8                                   3_675.66     1_144.36     4_820.02       0.4927          1.0377            1.0365         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         3_476.77       809.37     4_286.14       0.3815          1.0601            1.0619         6.44
IVFOPQ-m32-nl223-np11                                  3_316.92       917.27     4_234.19       0.3815          1.0601            1.0619         3.96
IVFOPQ-m64-nl223-np11                                  3_675.66     1_396.46     5_072.12       0.4927          1.0377            1.0365         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        3_476.77       933.54     4_410.31       0.3815          1.0601            1.0619         6.44
IVFOPQ-m32-nl223-np14                                  3_316.92     1_057.82     4_374.73       0.3815          1.0601            1.0619         3.96
IVFOPQ-m64-nl223-np14                                  3_675.66     1_651.54     5_327.20       0.4927          1.0377            1.0365         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        3_476.77     1_074.63     4_551.40       0.3815          1.0601            1.0619         6.44
IVFOPQ-m32-nl316-np1                                   3_384.89       425.90     3_810.79       0.3642          1.0646            1.0652         4.65
IVFOPQ-m64-nl316-np1                                   3_775.02       476.09     4_251.11       0.4453          1.0454            1.0429         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         3_590.60       437.60     4_028.20       0.3360          1.1165            1.0691         7.13
IVFOPQ-m32-nl316-np2                                   3_384.89       478.06     3_862.95       0.3854          1.0581            1.0608         4.65
IVFOPQ-m64-nl316-np2                                   3_775.02       574.10     4_349.13       0.4877          1.0369            1.0368         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         3_590.60       496.16     4_086.76       0.3812          1.0609            1.0625         7.13
IVFOPQ-m32-nl316-np4                                   3_384.89       579.12     3_964.01       0.3905          1.0570            1.0595         4.65
IVFOPQ-m64-nl316-np4                                   3_775.02       764.36     4_539.39       0.5013          1.0349            1.0349         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         3_590.60       599.28     4_189.88       0.3874          1.0588            1.0605         7.13
IVFOPQ-m32-nl316-np8                                   3_384.89       773.02     4_157.91       0.3910          1.0568            1.0594         4.65
IVFOPQ-m64-nl316-np8                                   3_775.02     1_125.42     4_900.44       0.5031          1.0346            1.0346         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         3_590.60       789.78     4_380.38       0.3908          1.0570            1.0595         7.13
IVFOPQ-m32-nl316-np15                                  3_384.89     1_095.65     4_480.54       0.3910          1.0568            1.0594         4.65
IVFOPQ-m64-nl316-np15                                  3_775.02     1_722.44     5_497.46       0.5031          1.0346            1.0346         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        3_590.60     1_136.36     4_726.96       0.3910          1.0568            1.0594         7.13
IVFOPQ-m32-nl316-np17                                  3_384.89     1_193.85     4_578.74       0.3910          1.0568            1.0594         4.65
IVFOPQ-m64-nl316-np17                                  3_775.02     1_911.39     5_686.41       0.5031          1.0346            1.0346         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        3_590.60     1_217.72     4_808.32       0.3910          1.0568            1.0594         7.13
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
Exhaustive (query)                                        68.21     1_215.61     1_283.82       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_302.49       451.61     3_754.10       0.3408          1.1429            1.0711         6.32
SOAROPQ-near-np2                                       3_302.49       510.20     3_812.69       0.3688          1.0658            1.0662         6.32
SOAROPQ-near-np4                                       3_302.49       607.96     3_910.45       0.3712          1.0645            1.0651         6.32
SOAROPQ-near-np7                                       3_302.49       757.24     4_059.73       0.3714          1.0642            1.0651         6.32
SOAROPQ-near-np8                                       3_302.49       800.50     4_102.99       0.3714          1.0642            1.0651         6.32
SOAROPQ-near-np12                                      3_302.49       992.75     4_295.24       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.3-np1                                   3_294.40       468.47     3_762.87       0.3404          1.1438            1.0711         6.32
SOAROPQ-shift0.3-np2                                   3_294.40       515.48     3_809.88       0.3690          1.0659            1.0662         6.32
SOAROPQ-shift0.3-np4                                   3_294.40       611.56     3_905.97       0.3712          1.0644            1.0651         6.32
SOAROPQ-shift0.3-np7                                   3_294.40       814.32     4_108.73       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.3-np8                                   3_294.40       802.81     4_097.21       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.3-np12                                  3_294.40       994.36     4_288.77       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.7-np1                                   3_275.83       450.07     3_725.90       0.3399          1.1453            1.0712         6.32
SOAROPQ-shift0.7-np2                                   3_275.83       510.12     3_785.94       0.3689          1.0662            1.0663         6.32
SOAROPQ-shift0.7-np4                                   3_275.83       607.98     3_883.81       0.3712          1.0645            1.0651         6.32
SOAROPQ-shift0.7-np7                                   3_275.83       757.82     4_033.64       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.7-np8                                   3_275.83       802.10     4_077.93       0.3714          1.0642            1.0651         6.32
SOAROPQ-shift0.7-np12                                  3_275.83       993.91     4_269.73       0.3714          1.0642            1.0651         6.32
SOAROPQ-orth1-np1                                      3_302.41       449.14     3_751.54       0.3410          1.1440            1.0711         6.32
SOAROPQ-orth1-np2                                      3_302.41       510.33     3_812.74       0.3690          1.0656            1.0663         6.32
SOAROPQ-orth1-np4                                      3_302.41       607.88     3_910.29       0.3712          1.0645            1.0651         6.32
SOAROPQ-orth1-np7                                      3_302.41       753.52     4_055.93       0.3714          1.0642            1.0651         6.32
SOAROPQ-orth1-np8                                      3_302.41       833.27     4_135.68       0.3714          1.0642            1.0651         6.32
SOAROPQ-orth1-np12                                     3_302.41     1_013.59     4_316.00       0.3714          1.0642            1.0651         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (LowRank, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.53     1_307.08     1_375.60       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_838.00       440.95     3_278.95       0.6379          1.0357            1.0278         3.49
IVFOPQ-m64-nl111-np1                                   3_179.34       528.73     3_708.07       0.7227          1.0226            1.0132         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         3_002.23       454.65     3_456.89       0.6657          1.0326            1.0252         5.97
IVFOPQ-m32-nl111-np2                                   2_838.00       493.88     3_331.88       0.6782          1.0259            1.0242         3.49
IVFOPQ-m64-nl111-np2                                   3_179.34       628.64     3_807.97       0.7743          1.0123            1.0108         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         3_002.23       506.03     3_508.27       0.6781          1.0261            1.0243         5.97
IVFOPQ-m32-nl111-np4                                   2_838.00       589.85     3_427.85       0.6785          1.0258            1.0242         3.49
IVFOPQ-m64-nl111-np4                                   3_179.34       806.49     3_985.83       0.7748          1.0122            1.0108         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         3_002.23       604.02     3_606.25       0.6785          1.0258            1.0242         5.97
IVFOPQ-m32-nl111-np5                                   2_838.00       680.72     3_518.72       0.6785          1.0258            1.0242         3.49
IVFOPQ-m64-nl111-np5                                   3_179.34       901.62     4_080.96       0.7748          1.0122            1.0108         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         3_002.23       653.20     3_655.43       0.6785          1.0258            1.0242         5.97
IVFOPQ-m32-nl111-np8                                   2_838.00       782.68     3_620.68       0.6785          1.0258            1.0242         3.49
IVFOPQ-m64-nl111-np8                                   3_179.34     1_170.59     4_349.92       0.7748          1.0122            1.0108         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         3_002.23       818.72     3_820.95       0.6785          1.0258            1.0242         5.97
IVFOPQ-m32-nl111-np10                                  2_838.00       877.44     3_715.44       0.6785          1.0258            1.0242         3.49
IVFOPQ-m64-nl111-np10                                  3_179.34     1_340.22     4_519.56       0.7748          1.0122            1.0108         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        3_002.23       928.91     3_931.14       0.6785          1.0258            1.0242         5.97
IVFOPQ-m32-nl158-np1                                   3_065.46       441.68     3_507.14       0.6006          1.0433            1.0327         3.84
IVFOPQ-m64-nl158-np1                                   3_430.38       554.05     3_984.43       0.6704          1.0314            1.0188         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         3_284.70       454.28     3_738.98       0.6668          1.0291            1.0253         6.32
IVFOPQ-m32-nl158-np2                                   3_065.46       492.39     3_557.84       0.6784          1.0255            1.0238         3.84
IVFOPQ-m64-nl158-np2                                   3_430.38       619.44     4_049.82       0.7693          1.0130            1.0112         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         3_284.70       505.47     3_790.18       0.6848          1.0245            1.0230         6.32
IVFOPQ-m32-nl158-np4                                   3_065.46       589.10     3_654.56       0.6862          1.0241            1.0229         3.84
IVFOPQ-m64-nl158-np4                                   3_430.38       797.51     4_227.89       0.7800          1.0115            1.0104         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         3_284.70       608.59     3_893.29       0.6862          1.0241            1.0229         6.32
IVFOPQ-m32-nl158-np7                                   3_065.46       728.43     3_793.88       0.6862          1.0241            1.0229         3.84
IVFOPQ-m64-nl158-np7                                   3_430.38     1_056.43     4_486.81       0.7800          1.0115            1.0104         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         3_284.70       746.81     4_031.51       0.6862          1.0241            1.0229         6.32
IVFOPQ-m32-nl158-np8                                   3_065.46       775.86     3_841.32       0.6862          1.0241            1.0229         3.84
IVFOPQ-m64-nl158-np8                                   3_430.38     1_144.33     4_574.71       0.7800          1.0115            1.0104         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         3_284.70       791.64     4_076.34       0.6862          1.0241            1.0229         6.32
IVFOPQ-m32-nl158-np12                                  3_065.46       963.16     4_028.61       0.6862          1.0241            1.0229         3.84
IVFOPQ-m64-nl158-np12                                  3_430.38     1_498.09     4_928.47       0.7800          1.0115            1.0104         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        3_284.70       994.13     4_278.83       0.6862          1.0241            1.0229         6.32
IVFOPQ-m32-nl223-np1                                   3_913.79       431.64     4_345.43       0.5147          1.0613            1.0521         3.96
IVFOPQ-m64-nl223-np1                                   4_239.82       494.01     4_733.84       0.5596          1.0501            1.0411         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         3_594.04       451.94     4_045.98       0.6280          1.0356            1.0312         6.44
IVFOPQ-m32-nl223-np2                                   3_913.79       488.81     4_402.60       0.6391          1.0326            1.0291         3.96
IVFOPQ-m64-nl223-np2                                   4_239.82       622.49     4_862.32       0.7153          1.0207            1.0158         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         3_594.04       506.12     4_100.16       0.6771          1.0259            1.0245         6.44
IVFOPQ-m32-nl223-np4                                   3_913.79       586.02     4_499.81       0.6861          1.0242            1.0230         3.96
IVFOPQ-m64-nl223-np4                                   4_239.82       791.39     5_031.21       0.7776          1.0119            1.0108         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         3_594.04       603.82     4_197.86       0.6903          1.0235            1.0224         6.44
IVFOPQ-m32-nl223-np8                                   3_913.79       774.00     4_687.79       0.6906          1.0234            1.0224         3.96
IVFOPQ-m64-nl223-np8                                   4_239.82     1_132.83     5_372.66       0.7831          1.0111            1.0101         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         3_594.04       789.68     4_383.72       0.6906          1.0234            1.0224         6.44
IVFOPQ-m32-nl223-np11                                  3_913.79       912.26     4_826.04       0.6906          1.0234            1.0224         3.96
IVFOPQ-m64-nl223-np11                                  4_239.82     1_392.82     5_632.64       0.7831          1.0111            1.0101         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        3_594.04       933.07     4_527.11       0.6906          1.0234            1.0224         6.44
IVFOPQ-m32-nl223-np14                                  3_913.79     1_049.24     4_963.03       0.6906          1.0234            1.0224         3.96
IVFOPQ-m64-nl223-np14                                  4_239.82     1_644.43     5_884.25       0.7831          1.0111            1.0101         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        3_594.04     1_069.25     4_663.28       0.6906          1.0234            1.0224         6.44
IVFOPQ-m32-nl316-np1                                   3_384.68       428.31     3_813.00       0.4109          1.0844            1.0784         4.65
IVFOPQ-m64-nl316-np1                                   3_738.22       480.01     4_218.22       0.4291          1.0745            1.0679         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         3_599.05       437.30     4_036.35       0.5459          1.0506            1.0474         7.13
IVFOPQ-m32-nl316-np2                                   3_384.68       481.12     3_865.80       0.5565          1.0475            1.0444         4.65
IVFOPQ-m64-nl316-np2                                   3_738.22       573.89     4_312.10       0.6052          1.0365            1.0328         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         3_599.05       499.59     4_098.64       0.6412          1.0320            1.0298         7.13
IVFOPQ-m32-nl316-np4                                   3_384.68       582.56     3_967.24       0.6580          1.0290            1.0270         4.65
IVFOPQ-m64-nl316-np4                                   3_738.22       765.75     4_503.96       0.7361          1.0174            1.0146         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         3_599.05       602.35     4_201.41       0.6857          1.0243            1.0229         7.13
IVFOPQ-m32-nl316-np8                                   3_384.68       776.70     4_161.38       0.6929          1.0231            1.0217         4.65
IVFOPQ-m64-nl316-np8                                   3_738.22     1_123.93     4_862.14       0.7821          1.0112            1.0102         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         3_599.05       791.06     4_390.11       0.6944          1.0228            1.0215         7.13
IVFOPQ-m32-nl316-np15                                  3_384.68     1_094.10     4_478.78       0.6949          1.0228            1.0215         4.65
IVFOPQ-m64-nl316-np15                                  3_738.22     1_713.57     5_451.78       0.7844          1.0109            1.0100         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        3_599.05     1_111.82     4_710.87       0.6949          1.0228            1.0215         7.13
IVFOPQ-m32-nl316-np17                                  3_384.68     1_187.29     4_571.97       0.6949          1.0228            1.0215         4.65
IVFOPQ-m64-nl316-np17                                  3_738.22     1_885.14     5_623.35       0.7844          1.0109            1.0100         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        3_599.05     1_211.95     4_811.01       0.6949          1.0228            1.0215         7.13
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
Exhaustive (query)                                        68.53     1_307.08     1_375.60       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_290.04       487.54     3_777.58       0.6650          1.0298            1.0253         6.32
SOAROPQ-near-np2                                       3_290.04       509.94     3_799.98       0.6848          1.0245            1.0230         6.32
SOAROPQ-near-np4                                       3_290.04       609.84     3_899.88       0.6862          1.0241            1.0229         6.32
SOAROPQ-near-np7                                       3_290.04       751.10     4_041.14       0.6862          1.0241            1.0229         6.32
SOAROPQ-near-np8                                       3_290.04       808.90     4_098.94       0.6862          1.0241            1.0229         6.32
SOAROPQ-near-np12                                      3_290.04       987.89     4_277.93       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.3-np1                                   3_273.10       453.55     3_726.65       0.6664          1.0292            1.0253         6.32
SOAROPQ-shift0.3-np2                                   3_273.10       505.64     3_778.74       0.6848          1.0245            1.0230         6.32
SOAROPQ-shift0.3-np4                                   3_273.10       602.08     3_875.18       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.3-np7                                   3_273.10       747.67     4_020.77       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.3-np8                                   3_273.10       792.96     4_066.06       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.3-np12                                  3_273.10       985.24     4_258.34       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.7-np1                                   3_270.67       453.32     3_723.99       0.6671          1.0290            1.0253         6.32
SOAROPQ-shift0.7-np2                                   3_270.67       504.88     3_775.55       0.6848          1.0245            1.0230         6.32
SOAROPQ-shift0.7-np4                                   3_270.67       600.89     3_871.56       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.7-np7                                   3_270.67       744.34     4_015.01       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.7-np8                                   3_270.67       796.64     4_067.31       0.6862          1.0241            1.0229         6.32
SOAROPQ-shift0.7-np12                                  3_270.67       993.03     4_263.70       0.6862          1.0241            1.0229         6.32
SOAROPQ-orth1-np1                                      3_307.42       454.79     3_762.20       0.6666          1.0291            1.0253         6.32
SOAROPQ-orth1-np2                                      3_307.42       505.33     3_812.74       0.6848          1.0245            1.0230         6.32
SOAROPQ-orth1-np4                                      3_307.42       601.95     3_909.36       0.6862          1.0241            1.0229         6.32
SOAROPQ-orth1-np7                                      3_307.42       805.13     4_112.55       0.6862          1.0241            1.0229         6.32
SOAROPQ-orth1-np8                                      3_307.42       791.97     4_099.38       0.6862          1.0241            1.0229         6.32
SOAROPQ-orth1-np12                                     3_307.42       997.48     4_304.89       0.6862          1.0241            1.0229         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Euclidean (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.69     1_304.78     1_372.47       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   2_864.23       420.52     3_284.76       0.7297          1.1595            1.0773         3.49
IVFOPQ-m64-nl111-np1                                   3_190.41       482.39     3_672.80       0.7335          1.1569            1.0748         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         3_027.19       442.94     3_470.13       0.8479          1.0537            1.0263         5.97
IVFOPQ-m32-nl111-np2                                   2_864.23       482.51     3_346.75       0.8572          1.0441            1.0223         3.49
IVFOPQ-m64-nl111-np2                                   3_190.41       602.04     3_792.45       0.8650          1.0411            1.0188         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         3_027.19       524.84     3_552.03       0.8871          1.0256            1.0168         5.97
IVFOPQ-m32-nl111-np4                                   2_864.23       604.08     3_468.32       0.8923          1.0212            1.0153         3.49
IVFOPQ-m64-nl111-np4                                   3_190.41       836.51     4_026.92       0.9019          1.0181            1.0121         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         3_027.19       672.45     3_699.64       0.8947          1.0208            1.0149         5.97
IVFOPQ-m32-nl111-np5                                   2_864.23       664.08     3_528.32       0.8944          1.0200            1.0148         3.49
IVFOPQ-m64-nl111-np5                                   3_190.41       956.90     4_147.31       0.9043          1.0168            1.0117         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         3_027.19       749.91     3_777.10       0.8951          1.0203            1.0147         5.97
IVFOPQ-m32-nl111-np8                                   2_864.23       848.06     3_712.30       0.8956          1.0194            1.0145         3.49
IVFOPQ-m64-nl111-np8                                   3_190.41     1_307.07     4_497.49       0.9055          1.0162            1.0114         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         3_027.19       955.95     3_983.14       0.8956          1.0196            1.0145         5.97
IVFOPQ-m32-nl111-np10                                  2_864.23       974.03     3_838.27       0.8957          1.0194            1.0145         3.49
IVFOPQ-m64-nl111-np10                                  3_190.41     1_540.24     4_730.65       0.9057          1.0162            1.0114         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        3_027.19     1_087.96     4_115.15       0.8957          1.0195            1.0145         5.97
IVFOPQ-m32-nl158-np1                                   3_142.03       419.80     3_561.83       0.7161          1.1695            1.0931         3.84
IVFOPQ-m64-nl158-np1                                   3_479.83       503.63     3_983.47       0.7190          1.1674            1.0911         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         3_282.67       440.01     3_722.68       0.8432          1.0575            1.0274         6.32
IVFOPQ-m32-nl158-np2                                   3_142.03       474.95     3_616.98       0.8533          1.0473            1.0231         3.84
IVFOPQ-m64-nl158-np2                                   3_479.83       582.08     4_061.92       0.8596          1.0451            1.0202         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         3_282.67       504.02     3_786.69       0.8909          1.0248            1.0155         6.32
IVFOPQ-m32-nl158-np4                                   3_142.03       589.92     3_731.95       0.8967          1.0197            1.0137         3.84
IVFOPQ-m64-nl158-np4                                   3_479.83       798.09     4_277.92       0.9051          1.0172            1.0109         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         3_282.67       641.50     3_924.17       0.9003          1.0187            1.0131         6.32
IVFOPQ-m32-nl158-np7                                   3_142.03       754.29     3_896.32       0.9017          1.0173            1.0127         3.84
IVFOPQ-m64-nl158-np7                                   3_479.83     1_117.95     4_597.79       0.9105          1.0146            1.0099         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         3_282.67       835.20     4_117.87       0.9017          1.0176            1.0127         6.32
IVFOPQ-m32-nl158-np8                                   3_142.03       818.97     3_961.00       0.9019          1.0172            1.0126         3.84
IVFOPQ-m64-nl158-np8                                   3_479.83     1_232.28     4_712.11       0.9108          1.0145            1.0098         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         3_282.67       893.93     4_176.61       0.9019          1.0174            1.0127         6.32
IVFOPQ-m32-nl158-np12                                  3_142.03     1_031.14     4_173.17       0.9021          1.0171            1.0126         3.84
IVFOPQ-m64-nl158-np12                                  3_479.83     1_648.87     5_128.70       0.9110          1.0144            1.0098         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        3_282.67     1_137.63     4_420.30       0.9021          1.0172            1.0126         6.32
IVFOPQ-m32-nl223-np1                                   3_348.08       421.54     3_769.62       0.6999          1.1824            1.1101         3.96
IVFOPQ-m64-nl223-np1                                   3_686.48       469.46     4_155.93       0.7028          1.1792            1.1075         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         3_508.76       432.07     3_940.83       0.8355          1.0643            1.0322         6.44
IVFOPQ-m32-nl223-np2                                   3_348.08       472.24     3_820.32       0.8479          1.0517            1.0252         3.96
IVFOPQ-m64-nl223-np2                                   3_686.48       566.72     4_253.20       0.8547          1.0483            1.0209         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         3_508.76       495.07     4_003.83       0.8928          1.0254            1.0149         6.44
IVFOPQ-m32-nl223-np4                                   3_348.08       582.48     3_930.56       0.9003          1.0190            1.0125         3.96
IVFOPQ-m64-nl223-np4                                   3_686.48       766.66     4_453.13       0.9096          1.0155            1.0096         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         3_508.76       613.29     4_122.05       0.9049          1.0181            1.0119         6.44
IVFOPQ-m32-nl223-np8                                   3_348.08       784.64     4_132.72       0.9076          1.0155            1.0110         3.96
IVFOPQ-m64-nl223-np8                                   3_686.48     1_159.49     4_845.97       0.9174          1.0119            1.0081         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         3_508.76       848.58     4_357.33       0.9075          1.0160            1.0111         6.44
IVFOPQ-m32-nl223-np11                                  3_348.08       946.23     4_294.31       0.9080          1.0153            1.0109         3.96
IVFOPQ-m64-nl223-np11                                  3_686.48     1_456.20     5_142.67       0.9179          1.0117            1.0080         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        3_508.76     1_023.34     4_532.09       0.9079          1.0156            1.0110         6.44
IVFOPQ-m32-nl223-np14                                  3_348.08     1_096.36     4_444.44       0.9081          1.0152            1.0109         3.96
IVFOPQ-m64-nl223-np14                                  3_686.48     1_751.74     5_438.21       0.9180          1.0116            1.0080         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        3_508.76     1_201.59     4_710.35       0.9081          1.0154            1.0109         6.44
IVFOPQ-m32-nl316-np1                                   3_383.36       425.27     3_808.63       0.6802          1.1986            1.1335         4.65
IVFOPQ-m64-nl316-np1                                   3_771.12       470.38     4_241.50       0.6816          1.1968            1.1316         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         3_589.64       429.95     4_019.59       0.8253          1.0714            1.0362         7.13
IVFOPQ-m32-nl316-np2                                   3_383.36       471.15     3_854.51       0.8395          1.0575            1.0274         4.65
IVFOPQ-m64-nl316-np2                                   3_771.12       562.70     4_333.82       0.8437          1.0555            1.0255         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         3_589.64       484.73     4_074.37       0.8947          1.0260            1.0140         7.13
IVFOPQ-m32-nl316-np4                                   3_383.36       571.07     3_954.43       0.9044          1.0178            1.0109         4.65
IVFOPQ-m64-nl316-np4                                   3_771.12       746.91     4_518.02       0.9109          1.0157            1.0090         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         3_589.64       591.34     4_180.98       0.9104          1.0168            1.0101         7.13
IVFOPQ-m32-nl316-np8                                   3_383.36       761.32     4_144.68       0.9143          1.0131            1.0090         4.65
IVFOPQ-m64-nl316-np8                                   3_771.12     1_119.15     4_890.27       0.9211          1.0108            1.0073         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         3_589.64       805.34     4_394.98       0.9140          1.0139            1.0092         7.13
IVFOPQ-m32-nl316-np15                                  3_383.36     1_093.47     4_476.83       0.9150          1.0128            1.0089         4.65
IVFOPQ-m64-nl316-np15                                  3_771.12     1_775.91     5_547.03       0.9219          1.0105            1.0071         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        3_589.64     1_171.78     4_761.43       0.9149          1.0129            1.0089         7.13
IVFOPQ-m32-nl316-np17                                  3_383.36     1_197.36     4_580.72       0.9150          1.0128            1.0089         4.65
IVFOPQ-m64-nl316-np17                                  3_771.12     1_953.89     5_725.01       0.9219          1.0105            1.0071         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        3_589.64     1_276.50     4_866.14       0.9150          1.0129            1.0089         7.13
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
Exhaustive (query)                                        67.69     1_304.78     1_372.47       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       3_254.48       437.36     3_691.85       0.8427          1.0586            1.0257         6.32
SOAROPQ-near-np2                                       3_254.48       505.19     3_759.67       0.8908          1.0242            1.0151         6.32
SOAROPQ-near-np4                                       3_254.48       638.51     3_892.99       0.9008          1.0182            1.0130         6.32
SOAROPQ-near-np7                                       3_254.48       820.13     4_074.61       0.9019          1.0175            1.0126         6.32
SOAROPQ-near-np8                                       3_254.48       877.85     4_132.34       0.9020          1.0173            1.0126         6.32
SOAROPQ-near-np12                                      3_254.48     1_123.16     4_377.64       0.9021          1.0172            1.0126         6.32
SOAROPQ-shift0.3-np1                                   3_243.58       432.91     3_676.49       0.8453          1.0559            1.0264         6.32
SOAROPQ-shift0.3-np2                                   3_243.58       503.29     3_746.87       0.8916          1.0242            1.0152         6.32
SOAROPQ-shift0.3-np4                                   3_243.58       636.01     3_879.59       0.9006          1.0185            1.0130         6.32
SOAROPQ-shift0.3-np7                                   3_243.58       819.10     4_062.68       0.9018          1.0176            1.0127         6.32
SOAROPQ-shift0.3-np8                                   3_243.58       881.12     4_124.70       0.9019          1.0174            1.0126         6.32
SOAROPQ-shift0.3-np12                                  3_243.58     1_112.81     4_356.39       0.9021          1.0172            1.0126         6.32
SOAROPQ-shift0.7-np1                                   3_268.13       433.48     3_701.61       0.8402          1.0603            1.0288         6.32
SOAROPQ-shift0.7-np2                                   3_268.13       500.92     3_769.05       0.8897          1.0257            1.0157         6.32
SOAROPQ-shift0.7-np4                                   3_268.13       632.27     3_900.39       0.9001          1.0190            1.0132         6.32
SOAROPQ-shift0.7-np7                                   3_268.13       817.21     4_085.34       0.9016          1.0178            1.0128         6.32
SOAROPQ-shift0.7-np8                                   3_268.13       878.36     4_146.49       0.9018          1.0175            1.0127         6.32
SOAROPQ-shift0.7-np12                                  3_268.13     1_114.30     4_382.43       0.9021          1.0172            1.0126         6.32
SOAROPQ-orth1-np1                                      3_305.38       434.91     3_740.29       0.8426          1.0584            1.0268         6.32
SOAROPQ-orth1-np2                                      3_305.38       503.06     3_808.44       0.8908          1.0247            1.0153         6.32
SOAROPQ-orth1-np4                                      3_305.38       632.73     3_938.11       0.9004          1.0186            1.0130         6.32
SOAROPQ-orth1-np7                                      3_305.38       822.10     4_127.48       0.9018          1.0176            1.0127         6.32
SOAROPQ-orth1-np8                                      3_305.38       880.07     4_185.45       0.9019          1.0174            1.0126         6.32
SOAROPQ-orth1-np12                                     3_305.38     1_113.83     4_419.21       0.9021          1.0172            1.0126         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SOAR-OPQ - Cosine (Cell embeddings, 512D)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: Sweep A: SOAR-OPQ vs IVF-OPQ, 50k samples, 512D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.48     1_316.80     1_388.28       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   2_879.97       415.94     3_295.91       0.7778          1.1321            1.0546         3.49
IVFOPQ-m64-nl111-np1                                   3_188.32       473.97     3_662.30       0.7808          1.1300            1.0510         5.02
SOAROPQ-orth1-m32-nl111-np1                            3_037.21       433.60     3_470.81       0.8620          1.0587            1.0263         5.97
IVFOPQ-m32-nl111-np2                                   2_879.97       476.42     3_356.39       0.8817          1.0353            1.0190         3.49
IVFOPQ-m64-nl111-np2                                   3_188.32       587.51     3_775.84       0.8867          1.0331            1.0166         5.02
SOAROPQ-orth1-m32-nl111-np2                            3_037.21       503.70     3_540.90       0.8946          1.0306            1.0173         5.97
IVFOPQ-m32-nl111-np4                                   2_879.97       581.24     3_461.22       0.9012          1.0217            1.0152         3.49
IVFOPQ-m64-nl111-np4                                   3_188.32       805.73     3_994.06       0.9072          1.0194            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np4                            3_037.21       638.96     3_676.17       0.9012          1.0235            1.0154         5.97
IVFOPQ-m32-nl111-np5                                   2_879.97       646.98     3_526.95       0.9023          1.0211            1.0150         3.49
IVFOPQ-m64-nl111-np5                                   3_188.32       921.64     4_109.97       0.9083          1.0188            1.0125         5.02
SOAROPQ-orth1-m32-nl111-np5                            3_037.21       701.37     3_738.57       0.9020          1.0225            1.0152         5.97
IVFOPQ-m32-nl111-np8                                   2_879.97       801.19     3_681.16       0.9030          1.0208            1.0148         3.49
IVFOPQ-m64-nl111-np8                                   3_188.32     1_246.82     4_435.14       0.9091          1.0185            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np8                            3_037.21       900.07     3_937.28       0.9028          1.0211            1.0149         5.97
IVFOPQ-m32-nl111-np10                                  2_879.97       930.12     3_810.09       0.9030          1.0208            1.0148         3.49
IVFOPQ-m64-nl111-np10                                  3_188.32     1_464.96     4_653.29       0.9091          1.0185            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np10                           3_037.21     1_027.57     4_064.77       0.9030          1.0209            1.0148         5.97
IVFOPQ-m32-nl158-np1                                   3_117.71       415.43     3_533.14       0.7570          1.1532            1.0712         3.84
IVFOPQ-m64-nl158-np1                                   3_480.81       467.48     3_948.30       0.7594          1.1509            1.0687         5.36
SOAROPQ-orth1-m32-nl158-np1                            3_332.77       430.46     3_763.22       0.8555          1.0648            1.0289         6.32
IVFOPQ-m32-nl158-np2                                   3_117.71       466.01     3_583.71       0.8760          1.0408            1.0202         3.84
IVFOPQ-m64-nl158-np2                                   3_480.81       569.18     4_050.00       0.8807          1.0386            1.0177         5.36
SOAROPQ-orth1-m32-nl158-np2                            3_332.77       494.96     3_827.73       0.8965          1.0300            1.0165         6.32
IVFOPQ-m32-nl158-np4                                   3_117.71       570.14     3_687.85       0.9049          1.0201            1.0138         3.84
IVFOPQ-m64-nl158-np4                                   3_480.81       774.07     4_254.88       0.9107          1.0178            1.0116         5.36
SOAROPQ-orth1-m32-nl158-np4                            3_332.77       611.20     3_943.97       0.9050          1.0220            1.0141         6.32
IVFOPQ-m32-nl158-np7                                   3_117.71       723.10     3_840.81       0.9072          1.0190            1.0133         3.84
IVFOPQ-m64-nl158-np7                                   3_480.81     1_076.04     4_556.85       0.9131          1.0167            1.0111         5.36
SOAROPQ-orth1-m32-nl158-np7                            3_332.77       786.82     4_119.59       0.9069          1.0197            1.0134         6.32
IVFOPQ-m32-nl158-np8                                   3_117.71       776.47     3_894.17       0.9073          1.0189            1.0132         3.84
IVFOPQ-m64-nl158-np8                                   3_480.81     1_181.54     4_662.36       0.9133          1.0166            1.0111         5.36
SOAROPQ-orth1-m32-nl158-np8                            3_332.77       858.93     4_191.69       0.9071          1.0195            1.0133         6.32
IVFOPQ-m32-nl158-np12                                  3_117.71       982.22     4_099.93       0.9074          1.0189            1.0132         3.84
IVFOPQ-m64-nl158-np12                                  3_480.81     1_586.61     5_067.42       0.9134          1.0166            1.0110         5.36
SOAROPQ-orth1-m32-nl158-np12                           3_332.77     1_090.40     4_423.17       0.9074          1.0189            1.0132         6.32
IVFOPQ-m32-nl223-np1                                   3_380.29       416.73     3_797.02       0.7350          1.1743            1.0954         3.96
IVFOPQ-m64-nl223-np1                                   3_747.13       463.22     4_210.35       0.7371          1.1717            1.0929         5.49
SOAROPQ-orth1-m32-nl223-np1                            3_598.51       425.35     4_023.86       0.8488          1.0698            1.0308         6.44
IVFOPQ-m32-nl223-np2                                   3_380.29       488.80     3_869.09       0.8710          1.0451            1.0209         3.96
IVFOPQ-m64-nl223-np2                                   3_747.13       561.28     4_308.41       0.8752          1.0426            1.0184         5.49
SOAROPQ-orth1-m32-nl223-np2                            3_598.51       487.56     4_086.07       0.8987          1.0296            1.0151         6.44
IVFOPQ-m32-nl223-np4                                   3_380.29       565.19     3_945.48       0.9096          1.0183            1.0120         3.96
IVFOPQ-m64-nl223-np4                                   3_747.13       749.57     4_496.70       0.9156          1.0157            1.0102         5.49
SOAROPQ-orth1-m32-nl223-np4                            3_598.51       592.39     4_190.90       0.9099          1.0206            1.0123         6.44
IVFOPQ-m32-nl223-np8                                   3_380.29       756.50     4_136.79       0.9129          1.0166            1.0113         3.96
IVFOPQ-m64-nl223-np8                                   3_747.13     1_134.56     4_881.68       0.9192          1.0140            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np8                            3_598.51       805.85     4_404.36       0.9126          1.0174            1.0115         6.44
IVFOPQ-m32-nl223-np11                                  3_380.29       901.58     4_281.87       0.9131          1.0165            1.0113         3.96
IVFOPQ-m64-nl223-np11                                  3_747.13     1_437.04     5_184.17       0.9194          1.0140            1.0093         5.49
SOAROPQ-orth1-m32-nl223-np11                           3_598.51       974.20     4_572.71       0.9130          1.0168            1.0113         6.44
IVFOPQ-m32-nl223-np14                                  3_380.29     1_047.79     4_428.08       0.9131          1.0165            1.0113         3.96
IVFOPQ-m64-nl223-np14                                  3_747.13     1_699.32     5_446.45       0.9194          1.0140            1.0093         5.49
SOAROPQ-orth1-m32-nl223-np14                           3_598.51     1_132.80     4_731.31       0.9131          1.0166            1.0113         6.44
IVFOPQ-m32-nl316-np1                                   3_422.68       423.70     3_846.38       0.7069          1.2052            1.1249         4.65
IVFOPQ-m64-nl316-np1                                   3_795.01       468.59     4_263.59       0.7085          1.2033            1.1226         6.17
SOAROPQ-orth1-m32-nl316-np1                            3_701.47       427.23     4_128.70       0.8365          1.0789            1.0355         7.13
IVFOPQ-m32-nl316-np2                                   3_422.68       469.04     3_891.72       0.8592          1.0540            1.0239         4.65
IVFOPQ-m64-nl316-np2                                   3_795.01       557.06     4_352.06       0.8628          1.0520            1.0215         6.17
SOAROPQ-orth1-m32-nl316-np2                            3_701.47       485.52     4_186.99       0.9007          1.0281            1.0144         7.13
IVFOPQ-m32-nl316-np4                                   3_422.68       571.86     3_994.55       0.9127          1.0170            1.0107         4.65
IVFOPQ-m64-nl316-np4                                   3_795.01       740.52     4_535.53       0.9183          1.0150            1.0089         6.17
SOAROPQ-orth1-m32-nl316-np4                            3_701.47       594.03     4_295.49       0.9147          1.0182            1.0106         7.13
IVFOPQ-m32-nl316-np8                                   3_422.68       748.77     4_171.45       0.9177          1.0143            1.0097         4.65
IVFOPQ-m64-nl316-np8                                   3_795.01     1_098.80     4_893.80       0.9240          1.0122            1.0078         6.17
SOAROPQ-orth1-m32-nl316-np8                            3_701.47       785.03     4_486.50       0.9175          1.0151            1.0099         7.13
IVFOPQ-m32-nl316-np15                                  3_422.68     1_070.80     4_493.48       0.9181          1.0142            1.0096         4.65
IVFOPQ-m64-nl316-np15                                  3_795.01     1_735.87     5_530.88       0.9244          1.0121            1.0077         6.17
SOAROPQ-orth1-m32-nl316-np15                           3_701.47     1_137.71     4_839.17       0.9180          1.0142            1.0097         7.13
IVFOPQ-m32-nl316-np17                                  3_422.68     1_173.59     4_596.27       0.9181          1.0141            1.0096         4.65
IVFOPQ-m64-nl316-np17                                  3_795.01     1_915.25     5_710.26       0.9244          1.0121            1.0077         6.17
SOAROPQ-orth1-m32-nl316-np17                           3_701.47     1_320.52     5_021.98       0.9180          1.0142            1.0096         7.13
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
Exhaustive (query)                                        71.48     1_316.80     1_388.28       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                       3_312.77       427.94     3_740.71       0.8613          1.0569            1.0267         6.32
SOAROPQ-near-np2                                       3_312.77       490.58     3_803.35       0.8982          1.0267            1.0160         6.32
SOAROPQ-near-np4                                       3_312.77       613.04     3_925.81       0.9056          1.0206            1.0138         6.32
SOAROPQ-near-np7                                       3_312.77       787.78     4_100.55       0.9071          1.0192            1.0133         6.32
SOAROPQ-near-np8                                       3_312.77       864.82     4_177.59       0.9072          1.0191            1.0133         6.32
SOAROPQ-near-np12                                      3_312.77     1_143.20     4_455.97       0.9074          1.0189            1.0132         6.32
SOAROPQ-shift0.3-np1                                   3_312.80       430.14     3_742.94       0.8608          1.0591            1.0281         6.32
SOAROPQ-shift0.3-np2                                   3_312.80       489.85     3_802.65       0.8969          1.0289            1.0165         6.32
SOAROPQ-shift0.3-np4                                   3_312.80       611.67     3_924.46       0.9047          1.0218            1.0142         6.32
SOAROPQ-shift0.3-np7                                   3_312.80       785.34     4_098.14       0.9068          1.0195            1.0134         6.32
SOAROPQ-shift0.3-np8                                   3_312.80       854.89     4_167.69       0.9070          1.0193            1.0133         6.32
SOAROPQ-shift0.3-np12                                  3_312.80     1_081.04     4_393.84       0.9074          1.0189            1.0132         6.32
SOAROPQ-shift0.7-np1                                   3_260.67       427.37     3_688.04       0.8562          1.0648            1.0293         6.32
SOAROPQ-shift0.7-np2                                   3_260.67       492.21     3_752.89       0.8945          1.0319            1.0171         6.32
SOAROPQ-shift0.7-np4                                   3_260.67       610.13     3_870.80       0.9037          1.0233            1.0144         6.32
SOAROPQ-shift0.7-np7                                   3_260.67       791.63     4_052.30       0.9065          1.0201            1.0135         6.32
SOAROPQ-shift0.7-np8                                   3_260.67       848.45     4_109.12       0.9068          1.0197            1.0134         6.32
SOAROPQ-shift0.7-np12                                  3_260.67     1_085.19     4_345.86       0.9073          1.0190            1.0132         6.32
SOAROPQ-orth1-np1                                      3_343.43       427.94     3_771.37       0.8555          1.0648            1.0289         6.32
SOAROPQ-orth1-np2                                      3_343.43       493.29     3_836.72       0.8965          1.0300            1.0165         6.32
SOAROPQ-orth1-np4                                      3_343.43       611.04     3_954.47       0.9050          1.0220            1.0141         6.32
SOAROPQ-orth1-np7                                      3_343.43       790.32     4_133.76       0.9069          1.0197            1.0134         6.32
SOAROPQ-orth1-np8                                      3_343.43       849.09     4_192.52       0.9071          1.0195            1.0133         6.32
SOAROPQ-orth1-np12                                     3_343.43     1_087.35     4_430.78       0.9074          1.0189            1.0132         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
