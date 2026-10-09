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
Exhaustive (query)                                        11.21       629.44       640.64       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.21     6_276.50     6_287.71       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.93     1_236.62     1_250.54       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    13.93    12_452.15    12_466.07       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              178.93        89.36       268.28       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              178.93        97.02       275.95       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              178.93       133.09       312.02       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    178.93     1_402.40     1_581.33       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              270.02        88.66       358.69       0.9820          1.0001            1.0000         9.21
IVF-BF16-nl387-np27 (query)                              270.02       115.96       385.98       0.9828          1.0001            1.0000         9.21
IVF-BF16-nl387 (self)                                    270.02     1_216.44     1_486.46       0.9798          1.0001            1.0000         9.21
IVF-BF16-nl547-np23 (query)                              422.46        83.25       505.71       0.9776          1.0005            1.0000         9.23
IVF-BF16-nl547-np27 (query)                              422.46        91.76       514.22       0.9817          1.0002            1.0000         9.23
IVF-BF16-nl547-np33 (query)                              422.46       108.05       530.51       0.9828          1.0001            1.0000         9.23
IVF-BF16-nl547 (self)                                    422.46     1_122.40     1_544.86       0.9798          1.0001            1.0000         9.23
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
Exhaustive (query)                                        11.64       739.16       750.80       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.64     7_176.66     7_188.30       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   13.24     1_262.51     1_275.75       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    13.24    12_772.60    12_785.84       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              153.71        92.01       245.72       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              153.71       105.06       258.77       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              153.71       147.44       301.16       0.8871          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    153.71     1_530.86     1_684.57       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              234.28       101.12       335.41       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              234.28       138.21       372.49       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    234.28     1_276.58     1_510.86       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              418.02        85.82       503.83       0.8849          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              418.02        96.49       514.51       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              418.02       115.25       533.27       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    418.02     1_181.62     1_599.64       0.8852          1.0073            1.0020         9.51
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
Exhaustive (query)                                        11.22       655.16       666.38       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.22     6_285.87     6_297.09       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.65     1_203.55     1_218.20       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    14.65    11_937.31    11_951.96       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              196.75        83.70       280.45       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              196.75        91.22       287.97       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              196.75       123.19       319.94       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    196.75     1_203.68     1_400.43       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              273.49        84.53       358.02       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387-np27 (query)                              273.49       103.93       377.42       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387 (self)                                    273.49     1_040.65     1_314.14       0.9184          1.0030            1.0021         9.21
IVF-BF16-nl547-np23 (query)                              436.29        78.15       514.44       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np27 (query)                              436.29        86.23       522.52       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np33 (query)                              436.29        98.86       535.15       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547 (self)                                    436.29       970.76     1_407.05       0.9184          1.0030            1.0021         9.23
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
Exhaustive (query)                                        11.32       627.95       639.27       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.32     6_203.12     6_214.44       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.72     1_192.54     1_207.26       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    14.72    11_779.74    11_794.46       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              206.88        78.05       284.93       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              206.88        88.24       295.12       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              206.88       118.41       325.30       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    206.88     1_209.32     1_416.20       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              266.62        80.50       347.12       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387-np27 (query)                              266.62       103.25       369.86       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387 (self)                                    266.62     1_037.58     1_304.20       0.9429          1.0017            1.0009         9.21
IVF-BF16-nl547-np23 (query)                              449.14        77.55       526.70       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np27 (query)                              449.14        85.64       534.78       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np33 (query)                              449.14        97.38       546.52       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547 (self)                                    449.14       967.86     1_417.00       0.9429          1.0017            1.0009         9.23
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
Exhaustive (query)                                        51.82     1_221.63     1_273.45       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         51.82    12_089.24    12_141.06       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   58.95     5_144.14     5_203.09       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    58.95    52_860.25    52_919.20       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              448.26       265.93       714.19       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              448.26       303.89       752.15       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              448.26       420.39       868.65       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    448.26     4_277.02     4_725.28       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              706.32       276.71       983.04       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              706.32       360.97     1_067.30       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    706.32     3_641.76     4_348.08       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            1_107.81       261.89     1_369.71       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            1_107.81       289.91     1_397.72       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            1_107.81       337.52     1_445.33       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                  1_107.81     3_392.53     4_500.34       0.9679          1.0005            1.0000        36.89
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
Exhaustive (query)                                        11.16       631.14       642.30       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.16     6_155.10     6_166.27       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.49       984.30     1_003.78       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     19.49     9_955.63     9_975.12       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               163.59        61.46       225.05       0.9233          1.0021            1.0010         6.33
IVF-SQ8-nl273-np16 (query)                               163.59        69.32       232.91       0.9246          1.0019            1.0009         6.33
IVF-SQ8-nl273-np23 (query)                               163.59        94.60       258.19       0.9249          1.0019            1.0009         6.33
IVF-SQ8-nl273 (self)                                     163.59       906.01     1_069.61       0.9249          1.0018            1.0009         6.33
IVF-SQ8-nl387-np19 (query)                               252.32        63.99       316.31       0.9255          1.0019            1.0009         6.35
IVF-SQ8-nl387-np27 (query)                               252.32        80.94       333.26       0.9260          1.0018            1.0009         6.35
IVF-SQ8-nl387 (self)                                     252.32       798.30     1_050.62       0.9252          1.0018            1.0009         6.35
IVF-SQ8-nl547-np23 (query)                               425.82        59.99       485.81       0.9223          1.0022            1.0010         6.37
IVF-SQ8-nl547-np27 (query)                               425.82        67.03       492.85       0.9250          1.0019            1.0009         6.37
IVF-SQ8-nl547-np33 (query)                               425.82        76.71       502.53       0.9257          1.0018            1.0009         6.37
IVF-SQ8-nl547 (self)                                     425.82       745.28     1_171.10       0.9252          1.0018            1.0009         6.37
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
Exhaustive (query)                                        11.70       718.53       730.23       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.70     6_927.68     6_939.38       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    21.50       915.19       936.69       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     21.50    10_167.20    10_188.70       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               163.80        60.02       223.81       0.7368          1.0365            1.0161         6.33
IVF-SQ8-nl273-np16 (query)                               163.80        68.00       231.79       0.7369          1.0365            1.0161         6.33
IVF-SQ8-nl273-np23 (query)                               163.80        89.55       253.34       0.7369          1.0365            1.0161         6.33
IVF-SQ8-nl273 (self)                                     163.80       946.33     1_110.13       0.7358          1.0368            1.0158         6.33
IVF-SQ8-nl387-np19 (query)                               240.75        60.51       301.27       0.7379          1.0358            1.0161         6.35
IVF-SQ8-nl387-np27 (query)                               240.75        77.15       317.90       0.7380          1.0358            1.0161         6.35
IVF-SQ8-nl387 (self)                                     240.75       814.71     1_055.46       0.7387          1.0356            1.0157         6.35
IVF-SQ8-nl547-np23 (query)                               416.35        55.32       471.67       0.7359          1.0369            1.0163         6.37
IVF-SQ8-nl547-np27 (query)                               416.35        61.73       478.07       0.7362          1.0369            1.0161         6.37
IVF-SQ8-nl547-np33 (query)                               416.35        75.39       491.74       0.7362          1.0369            1.0161         6.37
IVF-SQ8-nl547 (self)                                     416.35       756.38     1_172.73       0.7361          1.0368            1.0159         6.37
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
Exhaustive (query)                                        11.03       648.48       659.51       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.03     6_355.84     6_366.86       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.05     1_002.16     1_021.21       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     19.05     9_868.97     9_888.02       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               184.57        61.32       245.88       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273-np16 (query)                               184.57        65.24       249.81       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273-np23 (query)                               184.57        87.29       271.86       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273 (self)                                     184.57       805.80       990.36       0.8119          1.0175            1.0155         6.33
IVF-SQ8-nl387-np19 (query)                               282.26        59.82       342.08       0.8150          1.0164            1.0145         6.35
IVF-SQ8-nl387-np27 (query)                               282.26        74.20       356.46       0.8150          1.0164            1.0145         6.35
IVF-SQ8-nl387 (self)                                     282.26       710.59       992.84       0.8122          1.0174            1.0155         6.35
IVF-SQ8-nl547-np23 (query)                               448.18        55.50       503.68       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547-np27 (query)                               448.18        60.72       508.90       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547-np33 (query)                               448.18        68.69       516.87       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547 (self)                                     448.18       674.47     1_122.65       0.8122          1.0174            1.0155         6.37
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
Exhaustive (query)                                        11.35       650.40       661.76       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.35     6_260.30     6_271.66       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.99       991.63     1_010.62       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     18.99     9_865.26     9_884.25       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                               286.43        56.30       342.73       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273-np16 (query)                               286.43        66.16       352.58       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273-np23 (query)                               286.43        82.64       369.07       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273 (self)                                     286.43       812.28     1_098.71       0.7897          1.0281            1.0256         6.33
IVF-SQ8-nl387-np19 (query)                               279.78        57.95       337.73       0.7899          1.0265            1.0243         6.35
IVF-SQ8-nl387-np27 (query)                               279.78        73.06       352.84       0.7899          1.0265            1.0243         6.35
IVF-SQ8-nl387 (self)                                     279.78       758.55     1_038.33       0.7901          1.0280            1.0256         6.35
IVF-SQ8-nl547-np23 (query)                               458.29        55.32       513.61       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547-np27 (query)                               458.29        59.44       517.73       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547-np33 (query)                               458.29        68.56       526.85       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547 (self)                                     458.29       672.45     1_130.74       0.7902          1.0280            1.0256         6.37
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
Exhaustive (query)                                        51.02     1_211.31     1_262.33       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         51.02    11_881.81    11_932.83       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    81.37     1_191.47     1_272.83       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     81.37    11_867.44    11_948.80       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               473.81        79.12       552.94       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np16 (query)                               473.81        82.61       556.42       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np23 (query)                               473.81       108.94       582.76       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273 (self)                                     473.81       912.92     1_386.74       0.8865          1.0073            1.0059        20.16
IVF-SQ8-nl387-np19 (query)                               715.19        84.49       799.69       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387-np27 (query)                               715.19       102.79       817.98       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387 (self)                                     715.19       805.18     1_520.37       0.8867          1.0073            1.0059        20.22
IVF-SQ8-nl547-np23 (query)                             1_150.47        86.60     1_237.06       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np27 (query)                             1_150.47        88.99     1_239.45       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np33 (query)                             1_150.47       101.55     1_252.01       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547 (self)                                   1_150.47       782.81     1_933.28       0.8865          1.0073            1.0059        20.30
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
Exhaustive (query)                                        11.33       667.76       679.09       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.33     6_542.00     6_553.32       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.85     1_041.34     1_060.19       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               802.04        46.98       849.02       0.9294          1.0120            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              802.04        84.57       886.61       0.9637          1.0063            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              802.04       166.42       968.46       0.9827          1.0032            1.0000        38.52
HNSW-M16-ef100 (self)                                    802.04       829.61     1_631.66       0.9641          1.0071            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_560.45        51.48     1_611.92       0.9573          1.0480            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_560.45        91.90     1_652.35       0.9829          1.0087            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_560.45       179.24     1_739.69       0.9924          1.0027            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_560.45       899.20     2_459.65       0.9833          1.0131            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_679.42        57.41     1_736.83       0.9678          1.0389            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_679.42       103.75     1_783.17       0.9868          1.0240            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_679.42       185.79     1_865.22       0.9951          1.0038            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_679.42     1_078.47     2_757.89       0.9874          1.0202            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_770.02        58.79     1_828.81       0.9734          1.0059            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_770.02       104.91     1_874.94       0.9902          1.0025            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_770.02       187.33     1_957.36       0.9963          1.0006            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_770.02     1_036.21     2_806.24       0.9904          1.0045            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          721.41        35.72       757.13       0.8779          1.0403            1.0032        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         721.41        84.48       805.89       0.9030          1.0086            1.0020        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         721.41       121.51       842.92       0.9152          1.0053            1.0014        26.89
HNSW-SQ8U-M16-ef100 (self)                               721.41       644.96     1_366.37       0.9019          1.0094            1.0020        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_392.08        39.02     1_431.10       0.8993          1.0110            1.0021        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_392.08        70.93     1_463.02       0.9150          1.0066            1.0014        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_392.08       127.73     1_519.82       0.9202          1.0051            1.0011        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_392.08       654.60     2_046.69       0.9142          1.0070            1.0014        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_507.10        40.23     1_547.33       0.9064          1.0113            1.0017        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_507.10        75.82     1_582.91       0.9180          1.0080            1.0012        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_507.10       135.85     1_642.95       0.9228          1.0039            1.0010        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_507.10       718.99     2_226.09       0.9179          1.0056            1.0012        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_579.11        44.61     1_623.72       0.9088          1.0094            1.0016        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_579.11        83.36     1_662.47       0.9195          1.0065            1.0012        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_579.11       144.05     1_723.16       0.9234          1.0026            1.0010        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_579.11       768.07     2_347.18       0.9193          1.0035            1.0012        45.20
HNSW-SQ8U-drop0 (query)                                1_388.41        68.61     1_457.02       0.8950          1.0088            1.0024        26.89
HNSW-SQ8U-drop0.001 (query)                            1_477.69        67.75     1_545.44       0.9150          1.0063            1.0014        26.89
HNSW-SQ8U-drop0.01 (query)                             1_400.27        70.41     1_470.68       0.8989          1.0077            1.0018        26.89
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
Exhaustive (query)                                        11.66       709.17       720.84       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.66     6_830.52     6_842.18       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    21.26       914.87       936.13       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               873.58        50.57       924.15       0.9343          1.0174            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              873.58        88.36       961.94       0.9676          1.0109            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              873.58       162.61     1_036.19       0.9861          1.0059            1.0000        39.09
HNSW-M16-ef100 (self)                                    873.58       865.36     1_738.94       0.9682          1.0111            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_632.94        52.20     1_685.14       0.9646          1.0137            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_632.94        97.98     1_730.92       0.9872          1.0037            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_632.94       176.73     1_809.67       0.9948          1.0020            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_632.94       929.32     2_562.27       0.9869          1.0059            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_736.41        58.65     1_795.06       0.9734          1.0025            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_736.41       102.76     1_839.17       0.9912          1.0010            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_736.41       184.66     1_921.07       0.9972          1.0002            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_736.41     1_012.65     2_749.06       0.9911          1.0011            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_781.41        61.35     1_842.76       0.9755          1.0265            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_781.41       108.38     1_889.80       0.9919          1.0005            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_781.41       199.04     1_980.46       0.9974          1.0002            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_781.41     1_066.84     2_848.26       0.9917          1.0035            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          749.91        36.62       786.53       0.6842          1.0630            1.0292        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         749.91        69.14       819.06       0.7085          1.0496            1.0242        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         749.91       122.05       871.97       0.7227          1.0429            1.0208        26.89
HNSW-SQ8U-M16-ef100 (self)                               749.91       634.54     1_384.46       0.7080          1.0484            1.0239        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_415.74        37.03     1_452.77       0.7080          1.0487            1.0231        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_415.74        69.56     1_485.30       0.7246          1.0406            1.0197        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_415.74       136.43     1_552.17       0.7318          1.0385            1.0184        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_415.74       680.49     2_096.23       0.7239          1.0424            1.0197        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_530.42        41.73     1_572.15       0.7176          1.0406            1.0208        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_530.42        76.10     1_606.52       0.7302          1.0380            1.0182        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_530.42       137.90     1_668.32       0.7352          1.0368            1.0171        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_530.42       725.38     2_255.80       0.7296          1.0383            1.0181        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_599.85        44.54     1_644.39       0.7211          1.0660            1.0196        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_599.85        79.44     1_679.29       0.7327          1.0427            1.0176        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_599.85       144.67     1_744.52       0.7367          1.0364            1.0168        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_599.85       808.54     2_408.39       0.7319          1.0411            1.0173        45.20
HNSW-SQ8U-drop0 (query)                                1_429.35        76.05     1_505.40       0.6642          1.0641            1.0307        26.89
HNSW-SQ8U-drop0.001 (query)                            1_464.61        69.99     1_534.60       0.7241          1.0410            1.0199        26.89
HNSW-SQ8U-drop0.01 (query)                             1_383.04        77.88     1_460.92       0.6893          1.0539            1.0259        26.89
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
Exhaustive (query)                                        11.21       673.82       685.02       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.21     6_724.37     6_735.58       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.07     1_003.27     1_022.34       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               816.12        52.01       868.13       0.9542         97.3425            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              816.12        89.13       905.25       0.9775          4.7135            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              816.12       156.48       972.60       0.9790          1.0052            1.0000        38.52
HNSW-M16-ef100 (self)                                    816.12       838.97     1_655.09       0.9783          4.7070            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_422.73        49.15     1_471.88       0.9733         53.2779            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_422.73        86.57     1_509.30       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_422.73       151.32     1_574.05       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_422.73       824.26     2_246.99       0.9998          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_461.52        59.13     1_520.64       0.9992          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_461.52        91.56     1_553.07       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_461.52       160.58     1_622.10       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_461.52       858.08     2_319.60       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_518.74        52.84     1_571.58       0.9993          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_518.74        94.87     1_613.61       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_518.74       160.10     1_678.84       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_518.74       877.69     2_396.43       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          707.53        38.18       745.70       0.8133          1.2718            1.0149        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         707.53        67.12       774.65       0.8141          1.1932            1.0148        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         707.53       118.16       825.69       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef100 (self)                               707.53       633.36     1_340.89       0.8116          1.0834            1.0156        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_295.59        36.39     1_331.98       0.8142          1.0166            1.0148        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_295.59        65.49     1_361.08       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_295.59       119.31     1_414.90       0.8146          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_295.59       611.72     1_907.31       0.8119          1.0175            1.0155        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_383.18        39.26     1_422.44       0.8144          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_383.18        70.57     1_453.75       0.8145          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_383.18       125.21     1_508.39       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_383.18       671.11     2_054.29       0.8119          1.0175            1.0155        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_444.63        40.85     1_485.48       0.8144          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_444.63        75.95     1_520.58       0.8145          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_444.63       127.53     1_572.16       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_444.63       688.69     2_133.32       0.8119          1.0175            1.0155        45.20
HNSW-SQ8U-drop0 (query)                                1_293.04        66.21     1_359.25       0.8081          1.0176            1.0157        26.89
HNSW-SQ8U-drop0.001 (query)                            1_280.98        64.53     1_345.51       0.8145          1.0288            1.0148        26.89
HNSW-SQ8U-drop0.01 (query)                             1_294.66        66.37     1_361.03       0.8049          1.0235            1.0163        26.89
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
Exhaustive (query)                                        11.18       679.18       690.36       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.18     6_803.58     6_814.76       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    22.26       998.26     1_020.52       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               877.62        55.11       932.73       0.9981          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              877.62        92.97       970.59       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              877.62       168.83     1_046.44       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    877.62       913.46     1_791.08       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_566.58        53.47     1_620.05       0.9987          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_566.58        95.66     1_662.23       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_566.58       173.31     1_739.89       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_566.58       922.41     2_488.98       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_663.68        58.84     1_722.53       0.9994          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_663.68       103.72     1_767.40       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_663.68       183.65     1_847.33       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_663.68       997.56     2_661.24       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_720.94        62.35     1_783.29       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_720.94       105.77     1_826.71       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_720.94       186.42     1_907.35       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_720.94     1_034.97     2_755.91       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          793.96        56.42       850.38       0.7888          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         793.96        72.42       866.38       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         793.96       138.84       932.81       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100 (self)                               793.96       680.74     1_474.71       0.7896          1.0282            1.0258        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_415.42        59.17     1_474.58       0.7891          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_415.42        73.76     1_489.18       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_415.42       134.21     1_549.63       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_415.42       720.75     2_136.17       0.7897          1.0281            1.0258        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_544.05        53.07     1_597.12       0.7892          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_544.05       102.18     1_646.23       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_544.05       174.73     1_718.78       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_544.05       892.45     2_436.50       0.7897          1.0281            1.0258        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_816.83        51.75     1_868.58       0.7892          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_816.83        88.75     1_905.58       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_816.83       169.97     1_986.80       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_816.83       858.78     2_675.61       0.7897          1.0281            1.0258        45.20
HNSW-SQ8U-drop0 (query)                                1_479.39        86.05     1_565.44       0.7860          1.0279            1.0254        26.89
HNSW-SQ8U-drop0.001 (query)                            1_752.02       102.73     1_854.76       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-drop0.01 (query)                             1_508.44        71.97     1_580.42       0.7830          1.0294            1.0262        26.89
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
Exhaustive (query)                                        50.93     1_278.15     1_329.07       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.93    12_841.26    12_892.19       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    85.06     1_214.21     1_299.26       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_368.29        80.09     1_448.38       0.9935          1.0370            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_368.29       138.73     1_507.02       0.9959          1.0168            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_368.29       241.46     1_609.75       0.9977          1.0059            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_368.29     1_336.63     2_704.92       0.9961          1.0160            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_460.65        85.45     2_546.09       0.9966          1.0187            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_460.65       144.08     2_604.73       0.9981          1.0103            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_460.65       255.96     2_716.60       0.9991          1.0038            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_460.65     1_406.48     3_867.12       0.9980          1.0130            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_616.49        90.02     2_706.50       0.9985          1.0080            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_616.49       152.65     2_769.14       0.9991          1.0051            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_616.49       265.85     2_882.33       0.9997          1.0007            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_616.49     1_497.99     4_114.48       0.9992          1.0034            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_682.16        92.45     2_774.62       0.9988          1.0069            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_682.16       152.63     2_834.79       0.9992          1.0050            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_682.16       267.68     2_949.84       0.9997          1.0016            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_682.16     1_506.16     4_188.32       0.9994          1.0032            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          828.89        40.18       869.07       0.9267          1.0524            1.0038        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         828.89        70.10       898.99       0.9301          1.0299            1.0038        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         828.89       123.44       952.33       0.9318          1.0189            1.0037        40.63
HNSW-SQ8U-M16-ef100 (self)                               828.89       675.36     1_504.25       0.9304          1.0252            1.0038        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_508.69        38.94     1_547.64       0.9320          1.0196            1.0036        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_508.69        68.58     1_577.28       0.9326          1.0165            1.0036        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_508.69       128.19     1_636.89       0.9333          1.0121            1.0036        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_508.69       657.63     2_166.33       0.9325          1.0161            1.0037        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_640.22        42.03     1_682.25       0.9321          1.0208            1.0036        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_640.22        72.21     1_712.43       0.9333          1.0123            1.0036        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_640.22       131.60     1_771.82       0.9337          1.0089            1.0036        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_640.22       697.51     2_337.72       0.9328          1.0138            1.0036        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_728.19        43.87     1_772.07       0.9325          1.0163            1.0036        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_728.19        73.81     1_802.00       0.9334          1.0103            1.0036        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_728.19       132.41     1_860.60       0.9339          1.0080            1.0036        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_728.19       710.17     2_438.37       0.9332          1.0122            1.0036        58.94
HNSW-SQ8U-drop0 (query)                                1_524.94        88.74     1_613.69       0.8641          1.0440            1.0221        40.63
HNSW-SQ8U-drop0.001 (query)                            1_539.40        72.19     1_611.59       0.9321          1.0202            1.0036        40.63
HNSW-SQ8U-drop0.01 (query)                             1_501.21        72.50     1_573.72       0.9320          1.0401            1.0020        40.63
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
Exhaustive (query)                                        53.06     1_294.52     1_347.58       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         53.06    13_148.21    13_201.27       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    93.91     1_236.26     1_330.17       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_250.88        69.34     1_320.22       0.9933          1.0949            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_250.88       118.87     1_369.75       0.9961          1.0308            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_250.88       211.45     1_462.32       0.9970          1.0196            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_250.88     1_155.65     2_406.52       0.9963          1.0290            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_332.56        73.43     2_405.99       0.9926          1.1668            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_332.56       122.95     2_455.51       0.9962          1.0635            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_332.56       220.80     2_553.36       0.9979          1.0240            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_332.56     1_198.33     3_530.90       0.9960          1.0843            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_425.24        76.53     2_501.77       0.9986          1.0192            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_425.24       131.22     2_556.46       0.9994          1.0073            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_425.24       232.21     2_657.45       0.9997          1.0034            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_425.24     1_252.31     3_677.55       0.9993          1.0081            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_484.37        75.67     2_560.04       0.9988          1.0230            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_484.37       133.12     2_617.49       0.9995          1.0077            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_484.37       248.22     2_732.60       0.9999          1.0003            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_484.37     1_278.06     3_762.43       0.9996          1.0052            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          828.97        38.95       867.92       0.6645          1.3968            1.1629        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         828.97        65.88       894.85       0.6660          1.3691            1.1620        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         828.97       120.75       949.72       0.6668          1.3543            1.1616        40.63
HNSW-SQ8U-M16-ef100 (self)                               828.97       625.12     1_454.08       0.6652          1.3795            1.1632        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_476.93        38.01     1_514.95       0.6639          1.4246            1.1636        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_476.93        74.06     1_550.99       0.6655          1.3869            1.1624        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_476.93       125.79     1_602.72       0.6662          1.3676            1.1621        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_476.93       674.53     2_151.46       0.6652          1.3917            1.1630        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_583.71        43.35     1_627.05       0.6665          1.3610            1.1620        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_583.71        75.85     1_659.56       0.6667          1.3572            1.1619        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_583.71       131.96     1_715.67       0.6672          1.3494            1.1614        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_583.71       704.81     2_288.51       0.6666          1.3590            1.1619        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_666.86        45.16     1_712.02       0.6669          1.3553            1.1616        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_666.86        75.59     1_742.45       0.6672          1.3501            1.1614        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_666.86       135.36     1_802.22       0.6674          1.3479            1.1613        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_666.86       780.71     2_447.58       0.6669          1.3558            1.1616        58.94
HNSW-SQ8U-drop0 (query)                                1_483.51        68.92     1_552.43       0.6189          1.5559            1.2372        40.63
HNSW-SQ8U-drop0.001 (query)                            1_525.37        73.91     1_599.28       0.6644          1.4121            1.1633        40.63
HNSW-SQ8U-drop0.01 (query)                             1_477.19        77.57     1_554.76       0.6789          1.3399            1.1504        40.63
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
Exhaustive (query)                                        34.04       704.57       738.61       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.04     2_300.31     2_334.35       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                692.17       675.17     1_367.34       0.2581          1.1827            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 692.17     2_199.28     2_891.45       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_369.42     1_529.60     2_899.02       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_369.42     5_044.47     6_413.89       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              2_288.58     3_625.06     5_913.64       0.3611          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               2_288.58    11_992.90    14_281.49       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                             885.76       199.31     1_085.07       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np12 (query)                            885.76       310.23     1_195.99       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np17 (query)                            885.76       422.97     1_308.72       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16 (self)                                  885.76     1_419.57     2_305.33       0.3038          1.1285            1.1336         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_423.08       369.32     1_792.40       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_423.08       565.18     1_988.27       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_423.08       779.47     2_202.56       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32 (self)                                1_423.08     2_574.10     3_997.18       0.4075          1.0802            1.0796         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_004.84       667.33     2_672.18       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_004.84     1_000.35     3_005.19       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_004.84     1_369.71     3_374.55       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64 (self)                                2_004.84     4_589.19     6_594.03       0.6325          1.0279            1.0244         3.46
IVF-PQ-nl223-m16-np11 (query)                            876.59       297.87     1_174.46       0.3866          1.0890            1.0897         1.23
IVF-PQ-nl223-m16-np14 (query)                            876.59       364.26     1_240.85       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16-np21 (query)                            876.59       532.02     1_408.61       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16 (self)                                  876.59     1_775.80     2_652.39       0.3106          1.1231            1.1271         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_499.26       518.86     2_018.12       0.4961          1.0568            1.0521         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_499.26       634.57     2_133.83       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_499.26       938.48     2_437.73       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32 (self)                                1_499.26     3_060.24     4_559.50       0.4138          1.0784            1.0759         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_040.90       906.93     2_947.83       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_040.90     1_124.51     3_165.41       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_040.90     1_635.00     3_675.89       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64 (self)                                2_040.90     5_413.57     7_454.47       0.6393          1.0273            1.0234         3.52
IVF-PQ-nl316-m16-np15 (query)                            993.05       378.66     1_371.70       0.3990          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np17 (query)                            993.05       419.89     1_412.94       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np25 (query)                            993.05       599.84     1_592.89       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16 (self)                                  993.05     2_013.90     3_006.95       0.3170          1.1180            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_495.30       655.60     2_150.90       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_495.30       730.76     2_226.06       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_495.30     1_046.65     2_541.95       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32 (self)                                1_495.30     3_490.38     4_985.68       0.4239          1.0739            1.0729         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_093.27     1_170.43     3_263.70       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_093.27     1_313.66     3_406.92       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_093.27     1_887.81     3_981.07       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64 (self)                                2_093.27     6_271.50     8_364.77       0.6491          1.0248            1.0220         3.61
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
Exhaustive (query)                                        68.82     1_267.10     1_335.92       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.82     4_498.98     4_567.80       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_022.76       692.08     1_714.85       0.2443          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                               1_022.76     2_267.85     3_290.61       0.2277          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_353.94     1_571.77     2_925.71       0.2649          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_353.94     5_152.82     6_506.75       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_150.13     3_627.15     5_777.28       0.2958          1.0991            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_150.13    11_951.19    14_101.32       0.2627          1.1103            1.1143         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_149.38       289.30     1_438.68       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_149.38       444.79     1_594.17       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_149.38       633.01     1_782.39       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16 (self)                                1_149.38     2_035.36     3_184.74       0.2613          1.1090            1.1156         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_507.30       404.53     1_911.83       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_507.30       643.41     2_150.71       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_507.30       868.00     2_375.30       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32 (self)                                1_507.30     2_851.44     4_358.75       0.2899          1.0920            1.0958         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_376.58       731.39     3_107.96       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_376.58     1_144.62     3_521.19       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_376.58     1_607.39     3_983.97       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64 (self)                                2_376.58     5_225.62     7_602.19       0.3928          1.0580            1.0570         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_151.80       419.50     1_571.30       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_151.80       512.34     1_664.14       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_151.80       723.29     1_875.10       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16 (self)                                1_151.80     2_400.18     3_551.98       0.2659          1.1043            1.1093         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_511.18       588.32     2_099.50       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_511.18       718.66     2_229.85       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_511.18     1_027.73     2_538.91       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                1_511.18     3_426.46     4_937.64       0.2945          1.0892            1.0915         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_359.55     1_048.07     3_407.62       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_359.55     1_283.39     3_642.94       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_359.55     1_856.08     4_215.63       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64 (self)                                2_359.55     6_162.58     8_522.13       0.3971          1.0575            1.0549         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_237.15       528.61     1_765.77       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_237.15       570.79     1_807.95       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_237.15       822.45     2_059.60       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16 (self)                                1_237.15     2_689.01     3_926.16       0.2713          1.0997            1.1062         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_618.82       731.53     2_350.35       0.3797          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_618.82       817.77     2_436.59       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_618.82     1_167.83     2_786.65       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32 (self)                                1_618.82     3_830.41     5_449.22       0.3003          1.0857            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_427.43     1_348.34     3_775.77       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_427.43     1_505.06     3_932.49       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_427.43     2_149.12     4_576.55       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64 (self)                                2_427.43     7_108.51     9_535.94       0.4062          1.0539            1.0530         4.17
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
Exhaustive (query)                                       101.22     1_771.10     1_872.32       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.22     5_885.39     5_986.61       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_149.94       701.13     1_851.07       0.2345          1.1095            1.1000         1.51
Exhaustive-PQ-m16 (self)                               1_149.94     2_270.47     3_420.41       0.2206          1.1180            1.1048         1.51
Exhaustive-PQ-m32 (query)                              1_583.67     1_542.62     3_126.29       0.2567          1.0943            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_583.67     5_053.65     6_637.32       0.2391          1.1012            1.1021         2.28
Exhaustive-PQ-m64 (query)                              2_446.24     3_583.08     6_029.32       0.2775          1.0855            1.0910         3.80
Exhaustive-PQ-m64 (self)                               2_446.24    11_886.64    14_332.88       0.2515          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_211.77     7_844.46    12_056.23       0.3162          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_211.77    26_076.06    30_287.83       0.2755          1.0817            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_687.18       421.43     2_108.61       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_687.18       606.74     2_293.92       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_687.18       815.47     2_502.65       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16 (self)                                1_687.18     2_613.51     4_300.69       0.2512          1.0937            1.1005         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_061.18       519.60     2_580.79       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_061.18       820.94     2_882.12       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_061.18     1_133.25     3_194.43       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32 (self)                                2_061.18     3_759.09     5_820.27       0.2627          1.0859            1.0912         2.74
IVF-PQ-nl158-m64-np7 (query)                           2_896.81       860.53     3_757.34       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np12 (query)                          2_896.81     1_363.46     4_260.27       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np17 (query)                          2_896.81     1_877.57     4_774.38       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64 (self)                                2_896.81     6_446.65     9_343.46       0.3100          1.0666            1.0678         4.27
IVF-PQ-nl158-m128-np7 (query)                          4_611.29     1_584.36     6_195.64       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np12 (query)                         4_611.29     2_505.02     7_116.31       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np17 (query)                         4_611.29     3_422.10     8_033.39       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128 (self)                               4_611.29    11_355.59    15_966.88       0.4627          1.0347            1.0321         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_518.59       503.44     2_022.03       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_518.59       605.57     2_124.16       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_518.59       894.66     2_413.25       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16 (self)                                1_518.59     3_054.36     4_572.95       0.2554          1.0889            1.0948         2.17
IVF-PQ-nl223-m32-np11 (query)                          1_983.76       748.98     2_732.73       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np14 (query)                          1_983.76       911.45     2_895.21       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np21 (query)                          1_983.76     1_328.18     3_311.94       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32 (self)                                1_983.76     4_347.04     6_330.79       0.2681          1.0815            1.0863         2.93
IVF-PQ-nl223-m64-np11 (query)                          2_862.63     1_223.65     4_086.27       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          2_862.63     1_517.34     4_379.96       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          2_862.63     2_206.57     5_069.19       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                2_862.63     7_266.74    10_129.37       0.3139          1.0649            1.0653         4.46
IVF-PQ-nl223-m128-np11 (query)                         4_623.57     2_290.86     6_914.43       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np14 (query)                         4_623.57     2_841.69     7_465.26       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np21 (query)                         4_623.57     4_132.91     8_756.47       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128 (self)                               4_623.57    13_823.23    18_446.80       0.4702          1.0336            1.0308         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_628.14       642.09     2_270.23       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_628.14       710.55     2_338.69       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_628.14     1_015.71     2_643.86       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16 (self)                                1_628.14     3_391.79     5_019.94       0.2591          1.0854            1.0920         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_076.15       979.15     3_055.30       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_076.15     1_097.72     3_173.87       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_076.15     1_573.50     3_649.65       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32 (self)                                2_076.15     5_233.51     7_309.66       0.2701          1.0793            1.0841         3.21
IVF-PQ-nl316-m64-np15 (query)                          2_982.01     1_627.10     4_609.11       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np17 (query)                          2_982.01     1_870.48     4_852.50       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np25 (query)                          2_982.01     2_697.01     5_679.02       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64 (self)                                2_982.01     8_678.66    11_660.67       0.3199          1.0622            1.0632         4.73
IVF-PQ-nl316-m128-np15 (query)                         4_850.92     2_997.40     7_848.33       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np17 (query)                         4_850.92     3_348.01     8_198.93       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np25 (query)                         4_850.92     4_801.03     9_651.95       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128 (self)                               4_850.92    16_042.20    20_893.12       0.4774          1.0316            1.0300         7.78
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
Exhaustive (query)                                        34.45       707.64       742.09       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.45     2_306.66     2_341.11       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                708.84       675.87     1_384.71       0.2932          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 708.84     2_207.37     2_916.21       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                              1_164.74     1_568.51     2_733.25       0.4008          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                               1_164.74     5_139.96     6_304.70       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                              2_013.94     3_662.79     5_676.73       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                               2_013.94    12_031.47    14_045.41       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                             886.22       196.18     1_082.40       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np12 (query)                            886.22       305.70     1_191.92       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np17 (query)                            886.22       422.19     1_308.41       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16 (self)                                  886.22     1_358.50     2_244.73       0.4294          1.1639            1.1599         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_361.71       354.68     1_716.39       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_361.71       583.70     1_945.41       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_361.71       773.73     2_135.44       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32 (self)                                1_361.71     2_575.57     3_937.28       0.6060          1.0690            1.0642         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_072.45       623.77     2_696.22       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_072.45       993.31     3_065.76       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_072.45     1_388.30     3_460.75       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                2_072.45     4_591.83     6_664.28       0.7974          1.0165            1.0143         3.46
IVF-PQ-nl223-m16-np11 (query)                            893.48       284.92     1_178.41       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np14 (query)                            893.48       352.40     1_245.88       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np21 (query)                            893.48       518.81     1_412.29       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16 (self)                                  893.48     1_751.08     2_644.56       0.4242          1.1680            1.1638         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_298.61       532.48     1_831.09       0.6766          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_298.61       637.50     1_936.10       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_298.61       967.31     2_265.91       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32 (self)                                1_298.61     3_109.82     4_408.42       0.6033          1.0702            1.0654         2.00
IVF-PQ-nl223-m64-np11 (query)                          1_910.58       898.19     2_808.77       0.8369          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                          1_910.58     1_111.15     3_021.73       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                          1_910.58     1_651.17     3_561.75       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                1_910.58     5_497.80     7_408.38       0.8007          1.0159            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                            924.76       393.10     1_317.86       0.5369          1.0879            1.0854         1.32
IVF-PQ-nl316-m16-np17 (query)                            924.76       422.45     1_347.22       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16-np25 (query)                            924.76       614.75     1_539.52       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16 (self)                                  924.76     2_030.37     2_955.13       0.4164          1.1739            1.1698         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_403.41       654.39     2_057.80       0.6804          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_403.41       743.45     2_146.85       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_403.41     1_061.05     2_464.46       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32 (self)                                1_403.41     3_529.51     4_932.92       0.6013          1.0708            1.0663         2.09
IVF-PQ-nl316-m64-np15 (query)                          1_986.84     1_163.21     3_150.05       0.8382          1.0090            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          1_986.84     1_304.25     3_291.09       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                          1_986.84     1_891.13     3_877.97       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                1_986.84     6_308.84     8_295.68       0.8022          1.0156            1.0137         3.61
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
Exhaustive (query)                                        68.61     1_235.03     1_303.64       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.61     4_035.07     4_103.68       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                844.22       672.58     1_516.80       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 844.22     2_206.38     3_050.60       0.1772          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_222.52     1_520.13     2_742.65       0.2802          1.1736            1.1699         2.03
Exhaustive-PQ-m32 (self)                               1_222.52     4_994.84     6_217.36       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                              2_042.01     3_567.31     5_609.32       0.3752          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                               2_042.01    11_810.68    13_852.69       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_120.68       255.95     1_376.63       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_120.68       399.49     1_520.17       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_120.68       542.61     1_663.29       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16 (self)                                1_120.68     1_777.83     2_898.51       0.2739          1.2064            1.2097         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_506.19       388.01     1_894.19       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_506.19       603.44     2_109.63       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_506.19       826.46     2_332.64       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32 (self)                                1_506.19     2_695.93     4_202.11       0.3946          1.1241            1.1226         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_322.95       705.81     3_028.77       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_322.95     1_117.94     3_440.90       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_322.95     1_520.27     3_843.22       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64 (self)                                2_322.95     5_038.06     7_361.01       0.5740          1.0544            1.0503         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_146.97       393.20     1_540.17       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_146.97       475.75     1_622.73       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_146.97       693.73     1_840.71       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16 (self)                                1_146.97     2_306.10     3_453.07       0.2680          1.2120            1.2166         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_506.78       580.32     2_087.11       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_506.78       715.27     2_222.05       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_506.78     1_032.44     2_539.22       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32 (self)                                1_506.78     3_428.34     4_935.12       0.3847          1.1293            1.1282         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_385.89     1_037.83     3_423.71       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_385.89     1_285.46     3_671.35       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_385.89     1_882.31     4_268.19       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64 (self)                                2_385.89     6_225.11     8_611.00       0.5727          1.0549            1.0508         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_261.45       526.50     1_787.95       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_261.45       578.62     1_840.07       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_261.45       824.64     2_086.09       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16 (self)                                1_261.45     2_731.97     3_993.42       0.2621          1.2176            1.2225         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_657.91       749.87     2_407.78       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_657.91       836.56     2_494.46       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_657.91     1_197.29     2_855.19       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32 (self)                                1_657.91     3_938.32     5_596.23       0.3740          1.1347            1.1339         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_470.33     1_338.08     3_808.41       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_470.33     1_493.49     3_963.82       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_470.33     2_151.09     4_621.42       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64 (self)                                2_470.33     7_083.76     9_554.09       0.5697          1.0555            1.0517         4.17
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
Exhaustive (query)                                       101.30     1_770.47     1_871.77       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.30     5_803.97     5_905.27       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_136.34       695.22     1_831.56       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                               1_136.34     2_240.40     3_376.74       0.1758          1.3086            1.3090         1.51
Exhaustive-PQ-m32 (query)                              1_596.44     1_548.05     3_144.49       0.2712          1.1686            1.1636         2.28
Exhaustive-PQ-m32 (self)                               1_596.44     5_063.60     6_660.04       0.2191          1.2527            1.2505         2.28
Exhaustive-PQ-m64 (query)                              2_677.54     3_616.35     6_293.89       0.3546          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                               2_677.54    12_005.92    14_683.46       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             4_291.54     7_943.61    12_235.15       0.4597          1.0781            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_291.54    26_422.43    30_713.96       0.3908          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_515.59       348.37     1_863.95       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_515.59       536.90     2_052.48       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_515.59       746.08     2_261.66       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16 (self)                                1_515.59     2_418.64     3_934.23       0.2589          1.2185            1.2225         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_040.87       514.47     2_555.34       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_040.87       806.10     2_846.97       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_040.87     1_114.98     3_155.85       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32 (self)                                2_040.87     3_714.19     5_755.06       0.3677          1.1383            1.1375         2.74
IVF-PQ-nl158-m64-np7 (query)                           2_929.76       846.35     3_776.11       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np12 (query)                          2_929.76     1_341.72     4_271.48       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np17 (query)                          2_929.76     1_845.55     4_775.31       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64 (self)                                2_929.76     6_220.47     9_150.23       0.5182          1.0711            1.0673         4.27
IVF-PQ-nl158-m128-np7 (query)                          4_901.19     1_560.32     6_461.51       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np12 (query)                         4_901.19     2_480.44     7_381.63       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np17 (query)                         4_901.19     3_451.48     8_352.67       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128 (self)                               4_901.19    11_303.86    16_205.05       0.7139          1.0240            1.0194         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_579.35       506.17     2_085.52       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_579.35       625.44     2_204.79       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_579.35       895.64     2_474.99       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16 (self)                                1_579.35     3_124.26     4_703.61       0.2524          1.2254            1.2305         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_008.52       740.72     2_749.25       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_008.52       914.95     2_923.48       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_008.52     1_327.72     3_336.24       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32 (self)                                2_008.52     4_365.44     6_373.96       0.3495          1.1486            1.1485         2.93
IVF-PQ-nl223-m64-np11 (query)                          2_918.52     1_262.84     4_181.36       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np14 (query)                          2_918.52     1_506.84     4_425.36       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np21 (query)                          2_918.52     2_210.37     5_128.89       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64 (self)                                2_918.52     7_267.06    10_185.58       0.5036          1.0764            1.0726         4.46
IVF-PQ-nl223-m128-np11 (query)                         4_655.84     2_275.61     6_931.45       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np14 (query)                         4_655.84     2_829.83     7_485.67       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np21 (query)                         4_655.84     4_144.04     8_799.88       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128 (self)                               4_655.84    13_747.37    18_403.21       0.7136          1.0239            1.0198         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_708.33       658.09     2_366.41       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_708.33       724.56     2_432.89       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_708.33     1_033.17     2_741.50       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16 (self)                                1_708.33     3_458.41     5_166.74       0.2447          1.2328            1.2393         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_156.49       985.21     3_141.70       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_156.49     1_095.34     3_251.83       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_156.49     1_571.77     3_728.27       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32 (self)                                2_156.49     5_243.29     7_399.78       0.3316          1.1584            1.1592         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_027.71     1_604.78     4_632.49       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_027.71     1_792.53     4_820.23       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_027.71     2_586.07     5_613.78       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                3_027.71     8_558.45    11_586.16       0.4868          1.0819            1.0783         4.73
IVF-PQ-nl316-m128-np15 (query)                         4_762.72     2_967.60     7_730.31       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np17 (query)                         4_762.72     3_322.19     8_084.91       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np25 (query)                         4_762.72     4_930.19     9_692.91       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128 (self)                               4_762.72    16_222.35    20_985.06       0.7111          1.0240            1.0202         7.78
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
Exhaustive (query)                                        32.87       711.61       744.48       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.87     2_305.18     2_338.06       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                665.35       668.81     1_334.16       0.7118          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 665.35     2_204.46     2_869.81       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_163.19     1_520.74     2_683.93       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_163.19     5_064.45     6_227.64       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              1_908.67     3_591.14     5_499.82       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               1_908.67    11_990.13    13_898.81       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                             993.67       210.66     1_204.33       0.8274          1.0521            1.0445         1.17
IVF-PQ-nl158-m16-np12 (query)                            993.67       344.01     1_337.68       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16-np17 (query)                            993.67       483.12     1_476.79       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16 (self)                                  993.67     1_562.82     2_556.49       0.7673          1.0987            1.0833         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_434.60       399.61     1_834.21       0.8739          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_434.60       670.23     2_104.83       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_434.60       920.40     2_355.00       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32 (self)                                1_434.60     3_071.89     4_506.49       0.8286          1.0513            1.0425         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_037.77       713.89     2_751.66       0.9048          1.0151            1.0116         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_037.77     1_211.64     3_249.41       0.9055          1.0148            1.0113         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_037.77     1_704.95     3_742.71       0.9055          1.0147            1.0113         3.46
IVF-PQ-nl158-m64 (self)                                2_037.77     5_663.19     7_700.96       0.8704          1.0287            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_069.82       309.10     1_378.92       0.8421          1.0435            1.0371         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_069.82       372.63     1_442.44       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_069.82       546.16     1_615.98       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16 (self)                                1_069.82     1_807.13     2_876.95       0.7842          1.0842            1.0703         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_554.11       545.30     2_099.41       0.8836          1.0226            1.0184         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_554.11       682.09     2_236.20       0.8837          1.0225            1.0184         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_554.11     1_012.08     2_566.19       0.8838          1.0225            1.0184         2.00
IVF-PQ-nl223-m32 (self)                                1_554.11     3_378.74     4_932.85       0.8398          1.0442            1.0357         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_143.23       960.10     3_103.33       0.9103          1.0133            1.0100         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_143.23     1_209.43     3_352.67       0.9104          1.0132            1.0100         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_143.23     1_800.89     3_944.12       0.9105          1.0132            1.0100         3.52
IVF-PQ-nl223-m64 (self)                                2_143.23     5_987.93     8_131.16       0.8772          1.0255            1.0197         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_211.58       390.47     1_602.05       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_211.58       434.01     1_645.59       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_211.58       645.32     1_856.89       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16 (self)                                1_211.58     2_106.03     3_317.60       0.7916          1.0787            1.0640         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_651.17       704.02     2_355.19       0.8864          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_651.17       800.31     2_451.48       0.8865          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_651.17     1_161.07     2_812.25       0.8865          1.0213            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_651.17     3_827.86     5_479.04       0.8427          1.0431            1.0342         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_425.34     1_241.60     3_666.94       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_425.34     1_426.99     3_852.33       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_425.34     2_464.16     4_889.50       0.9129          1.0124            1.0094         3.61
IVF-PQ-nl316-m64 (self)                                2_425.34     6_731.38     9_156.72       0.8794          1.0245            1.0188         3.61
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
Exhaustive (query)                                        68.95     1_250.94     1_319.89       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.95     4_092.11     4_161.06       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                844.27       677.81     1_522.08       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 844.27     2_215.95     3_060.22       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_215.11     1_529.32     2_744.43       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_215.11     5_014.86     6_229.97       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_042.12     3_562.61     5_604.73       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_042.12    11_814.08    13_856.20       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_347.99       280.02     1_628.00       0.8367          1.0503            1.0400         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_347.99       438.04     1_786.03       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_347.99       621.70     1_969.69       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16 (self)                                1_347.99     1_993.42     3_341.41       0.7723          1.1013            1.0737         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_706.94       424.52     2_131.46       0.8684          1.0311            1.0245         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_706.94       697.45     2_404.39       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_706.94       969.92     2_676.86       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32 (self)                                1_706.94     3_181.04     4_887.98       0.8159          1.0642            1.0463         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_557.11       788.77     3_345.89       0.8900          1.0215            1.0164         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_557.11     1_327.66     3_884.77       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_557.11     1_850.88     4_407.99       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64 (self)                                2_557.11     6_384.18     8_941.29       0.8456          1.0436            1.0313         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_423.96       412.53     1_836.49       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_423.96       512.83     1_936.79       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_423.96       743.11     2_167.07       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16 (self)                                1_423.96     2_459.36     3_883.33       0.7969          1.0803            1.0562         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_776.42       611.29     2_387.71       0.8796          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_776.42       764.32     2_540.74       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_776.42     1_127.86     2_904.28       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                1_776.42     3_709.16     5_485.57       0.8298          1.0557            1.0378         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_601.31     1_099.06     3_700.38       0.9001          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_601.31     1_378.63     3_979.94       0.9002          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_601.31     2_055.74     4_657.06       0.9002          1.0178            1.0132         3.99
IVF-PQ-nl223-m64 (self)                                2_601.31     6_778.18     9_379.50       0.8569          1.0374            1.0254         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_511.97       531.68     2_043.65       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_511.97       583.17     2_095.15       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_511.97       839.43     2_351.40       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16 (self)                                1_511.97     2_777.69     4_289.66       0.8151          1.0656            1.0459         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_865.84       771.24     2_637.08       0.8921          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_865.84       867.20     2_733.05       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_865.84     1_248.04     3_113.88       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32 (self)                                1_865.84     4_117.26     5_983.11       0.8453          1.0453            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_702.64     1_385.66     4_088.30       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_702.64     1_557.64     4_260.28       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_702.64     2_279.23     4_981.87       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64 (self)                                2_702.64     7_521.56    10_224.21       0.8660          1.0332            1.0218         4.17
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
Exhaustive (query)                                       101.18     1_826.88     1_928.06       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.18     5_903.39     6_004.57       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_148.27       705.60     1_853.87       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_148.27     2_275.81     3_424.09       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                              1_645.69     1_597.93     3_243.61       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                               1_645.69     5_135.57     6_781.25       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                              2_674.20     3_660.81     6_335.02       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                               2_674.20    12_050.90    14_725.10       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             4_589.92     7_955.01    12_544.93       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              4_589.92    26_376.21    30_966.13       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_905.03       377.96     2_282.99       0.8519          1.0423            1.0326         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_905.03       581.68     2_486.70       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_905.03       801.48     2_706.51       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16 (self)                                1_905.03     2_674.46     4_579.48       0.7910          1.0832            1.0598         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_446.93       559.77     3_006.70       0.9001          1.0207            1.0128         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_446.93       911.60     3_358.53       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_446.93     1_289.06     3_735.99       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32 (self)                                2_446.93     4_226.06     6_672.99       0.8549          1.0435            1.0236         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_495.60       945.72     4_441.31       0.9204          1.0131            1.0070         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_495.60     1_578.26     5_073.86       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_495.60     2_192.15     5_687.75       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64 (self)                                3_495.60     7_275.64    10_771.23       0.8830          1.0284            1.0134         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_287.08     1_765.84     7_052.92       0.9394          1.0072            1.0031         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_287.08     2_952.69     8_239.77       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_287.08     4_101.84     9_388.92       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128 (self)                               5_287.08    13_626.57    18_913.65       0.9073          1.0171            1.0071         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_100.16       543.63     2_643.79       0.8626          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_100.16       651.46     2_751.61       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_100.16       951.87     3_052.03       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16 (self)                                2_100.16     3_096.05     5_196.21       0.8067          1.0699            1.0512         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_575.89       775.76     3_351.65       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_575.89       976.54     3_552.43       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_575.89     1_426.96     4_002.85       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32 (self)                                2_575.89     4_726.06     7_301.95       0.8676          1.0353            1.0194         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_519.12     1_289.23     4_808.35       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_519.12     1_623.35     5_142.47       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_519.12     2_437.52     5_956.64       0.9269          1.0111            1.0056         4.46
IVF-PQ-nl223-m64 (self)                                3_519.12     7_944.23    11_463.35       0.8921          1.0237            1.0111         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_503.97     2_432.29     7_936.26       0.9441          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_503.97     3_066.04     8_570.01       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_503.97     4_842.94    10_346.91       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128 (self)                               5_503.97    15_087.18    20_591.15       0.9134          1.0147            1.0057         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_370.80       706.39     3_077.19       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_370.80       784.87     3_155.67       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_370.80     1_115.12     3_485.92       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16 (self)                                2_370.80     3_718.64     6_089.43       0.8142          1.0645            1.0463         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_862.95     1_034.24     3_897.19       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_862.95     1_156.86     4_019.81       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_862.95     1_757.56     4_620.51       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32 (self)                                2_862.95     5_538.55     8_401.50       0.8728          1.0330            1.0175         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_862.94     1_671.58     5_534.52       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_862.94     1_885.05     5_747.99       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_862.94     2_735.80     6_598.74       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64 (self)                                3_862.94     9_057.85    12_920.79       0.8963          1.0221            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_723.03     3_123.01     8_846.04       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_723.03     3_505.73     9_228.76       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_723.03     5_155.00    10_878.03       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128 (self)                               5_723.03    17_039.00    22_762.03       0.9166          1.0138            1.0052         7.78
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
Exhaustive (query)                                        32.94       683.71       716.65       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.94     2_228.33     2_261.27       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_485.89       729.22     4_215.10       0.2865          1.1528            1.1332         1.26
Exhaustive-OPQ-m16 (self)                              3_485.89     2_724.07     6_209.96       0.2585          1.1711            1.1497         1.26
Exhaustive-OPQ-m32 (query)                             5_775.61     1_586.81     7_362.42       0.3260          1.1208            1.1171         2.03
Exhaustive-OPQ-m32 (self)                              5_775.61     5_527.58    11_303.19       0.2831          1.1440            1.1382         2.03
Exhaustive-OPQ-m64 (query)                             8_956.33     3_664.73    12_621.06       0.3797          1.0983            1.0951         3.55
Exhaustive-OPQ-m64 (self)                              8_956.33    12_514.36    21_470.70       0.3219          1.1205            1.1171         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_545.44       272.80     3_818.25       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_545.44       391.07     3_936.51       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_545.44       509.31     4_054.76       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16 (self)                               3_545.44     2_020.03     5_565.47       0.3169          1.1190            1.1234         1.67
IVF-OPQ-nl158-m32-np7 (query)                          5_821.74       433.22     6_254.96       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32-np12 (query)                         5_821.74       643.84     6_465.58       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32-np17 (query)                         5_821.74       842.15     6_663.89       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32 (self)                               5_821.74     3_074.22     8_895.96       0.4164          1.0763            1.0768         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_776.01       716.63     9_492.64       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_776.01     1_085.47     9_861.49       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_776.01     1_455.89    10_231.90       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64 (self)                               8_776.01     5_173.02    13_949.03       0.6380          1.0262            1.0237         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_674.29       361.68     4_035.97       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_674.29       472.41     4_146.70       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_674.29       606.88     4_281.17       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16 (self)                               3_674.29     2_377.71     6_052.00       0.3210          1.1162            1.1203         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_016.08       579.50     6_595.58       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_016.08       705.20     6_721.28       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_016.08     1_003.16     7_019.24       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32 (self)                               6_016.08     3_691.07     9_707.15       0.4230          1.0744            1.0737         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_845.79     1_010.67    10_856.46       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_845.79     1_232.06    11_077.86       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_845.79     1_758.61    11_604.40       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64 (self)                               9_845.79     6_217.12    16_062.91       0.6451          1.0255            1.0227         4.02
IVF-OPQ-nl316-m16-np15 (query)                         3_746.46       460.35     4_206.81       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16-np17 (query)                         3_746.46       517.04     4_263.49       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16-np25 (query)                         3_746.46       706.43     4_452.88       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16 (self)                               3_746.46     2_703.96     6_450.42       0.3260          1.1127            1.1171         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_096.01       746.45     6_842.46       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_096.01       818.24     6_914.26       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_096.01     1_161.52     7_257.54       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32 (self)                               6_096.01     4_168.93    10_264.95       0.4329          1.0702            1.0708         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_283.18     1_269.59    10_552.77       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_283.18     1_421.34    10_704.52       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_283.18     2_098.96    11_382.14       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64 (self)                               9_283.18     7_011.42    16_294.60       0.6520          1.0238            1.0217         4.36
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
Exhaustive (query)                                        69.33     1_306.69     1_376.02       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.33     4_337.67     4_407.00       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_929.38     1_079.21     7_008.59       0.2659          1.1129            1.1008         2.26
Exhaustive-OPQ-m16 (self)                              5_929.38     4_769.86    10_699.24       0.2458          1.1245            1.1094         2.26
Exhaustive-OPQ-m32 (query)                             7_817.73     1_870.23     9_687.96       0.2865          1.0971            1.0962         3.03
Exhaustive-OPQ-m32 (self)                              7_817.73     7_525.10    15_342.84       0.2608          1.1083            1.1060         3.03
Exhaustive-OPQ-m64 (query)                            12_454.78     3_965.10    16_419.88       0.3196          1.0822            1.0844         4.55
Exhaustive-OPQ-m64 (self)                             12_454.78    14_420.78    26_875.56       0.2789          1.0967            1.0990         4.55
Exhaustive-OPQ-m128 (query)                           18_438.50     8_231.30    26_669.79       0.3687          1.0680            1.0687         7.61
Exhaustive-OPQ-m128 (self)                            18_438.50    28_806.97    47_245.47       0.3154          1.0825            1.0831         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_052.35       603.86     6_656.21       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_052.35       779.47     6_831.82       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_052.35       975.62     7_027.97       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16 (self)                               6_052.35     4_581.86    10_634.22       0.2764          1.0979            1.1033         3.07
IVF-OPQ-nl158-m32-np7 (query)                          7_949.61       753.13     8_702.75       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32-np12 (query)                         7_949.61       971.15     8_920.76       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32-np17 (query)                         7_949.61     1_214.73     9_164.34       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32 (self)                               7_949.61     5_452.92    13_402.53       0.3020          1.0859            1.0898         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_293.52     1_077.66    13_371.18       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_293.52     1_510.52    13_804.04       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_293.52     1_946.65    14_240.18       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64 (self)                              12_293.52     7_846.53    20_140.06       0.4021          1.0546            1.0550         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_321.29     1_634.33    19_955.61       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_321.29     2_358.49    20_679.78       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_321.29     3_084.22    21_405.51       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128 (self)                             18_321.29    11_748.04    30_069.33       0.6253          1.0192            1.0171         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_112.17       727.66     6_839.83       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_112.17       812.46     6_924.63       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_112.17     1_030.22     7_142.39       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16 (self)                               6_112.17     4_842.01    10_954.19       0.2769          1.0973            1.1025         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_096.94       927.34     9_024.28       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_096.94     1_067.30     9_164.25       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_096.94     1_383.47     9_480.42       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32 (self)                               8_096.94     6_055.64    14_152.59       0.3038          1.0847            1.0880         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_607.23     1_406.63    14_013.86       0.4869          1.0387            1.0372         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_607.23     1_657.81    14_265.04       0.4869          1.0386            1.0372         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_607.23     2_266.12    14_873.35       0.4869          1.0386            1.0372         5.49
IVF-OPQ-nl223-m64 (self)                              12_607.23     9_036.76    21_643.99       0.4074          1.0533            1.0532         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_713.98     2_191.94    20_905.92       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_713.98     2_631.29    21_345.27       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_713.98     3_672.86    22_386.84       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128 (self)                             18_713.98    13_849.44    32_563.42       0.6310          1.0188            1.0165         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_318.60       848.31     7_166.91       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_318.60       903.34     7_221.94       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_318.60     1_160.66     7_479.26       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16 (self)                               6_318.60     5_333.19    11_651.79       0.2799          1.0946            1.1004         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_281.59     1_109.38     9_390.96       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_281.59     1_195.06     9_476.65       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_281.59     1_573.68     9_855.27       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32 (self)                               8_281.59     6_603.68    14_885.26       0.3088          1.0817            1.0856         4.65
IVF-OPQ-nl316-m64-np15 (query)                        12_872.14     1_727.49    14_599.63       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64-np17 (query)                        12_872.14     1_927.71    14_799.85       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64-np25 (query)                        12_872.14     2_616.39    15_488.53       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64 (self)                              12_872.14    10_131.33    23_003.47       0.4153          1.0508            1.0515         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_974.50     2_714.71    21_689.21       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_974.50     3_005.28    21_979.78       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_974.50     4_168.69    23_143.19       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128 (self)                             18_974.50    15_385.98    34_360.48       0.6400          1.0173            1.0157         9.23
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
Exhaustive (query)                                       101.26     1_824.71     1_925.97       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.26     6_044.61     6_145.87       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_286.87     1_526.96    10_813.84       0.2602          1.0925            1.0831         3.76
Exhaustive-OPQ-m16 (self)                              9_286.87     8_260.17    17_547.04       0.2417          1.1021            1.0892         3.76
Exhaustive-OPQ-m32 (query)                            11_411.35     2_363.48    13_774.83       0.2792          1.0787            1.0797         4.53
Exhaustive-OPQ-m32 (self)                             11_411.35    10_963.59    22_374.94       0.2568          1.0877            1.0870         4.53
Exhaustive-OPQ-m64 (query)                            16_810.39     4_420.03    21_230.42       0.2961          1.0725            1.0758         6.05
Exhaustive-OPQ-m64 (self)                             16_810.39    17_796.51    34_606.91       0.2656          1.0825            1.0855         6.05
Exhaustive-OPQ-m128 (query)                           24_453.80     8_719.63    33_173.44       0.3311          1.0632            1.0658         9.11
Exhaustive-OPQ-m128 (self)                            24_453.80    32_458.81    56_912.61       0.2834          1.0754            1.0784         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_409.24     1_160.60    10_569.84       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_409.24     1_334.38    10_743.62       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_409.24     1_521.99    10_931.23       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16 (self)                               9_409.24     8_266.00    17_675.24       0.2665          1.0821            1.0878         4.98
IVF-OPQ-nl158-m32-np7 (query)                         11_574.87     1_346.44    12_921.31       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32-np12 (query)                        11_574.87     1_631.63    13_206.50       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32-np17 (query)                        11_574.87     1_933.11    13_507.98       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32 (self)                              11_574.87     9_638.59    21_213.46       0.2747          1.0784            1.0827         5.74
IVF-OPQ-nl158-m64-np7 (query)                         16_346.67     1_710.75    18_057.42       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64-np12 (query)                        16_346.67     2_185.96    18_532.63       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64-np17 (query)                        16_346.67     2_829.94    19_176.61       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64 (self)                              16_346.67    12_227.81    28_574.48       0.3209          1.0625            1.0645         7.27
IVF-OPQ-nl158-m128-np7 (query)                        28_579.17     2_486.13    31_065.30       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128-np12 (query)                       28_579.17     3_436.35    32_015.53       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128-np17 (query)                       28_579.17     4_386.44    32_965.61       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128 (self)                             28_579.17    17_974.92    46_554.09       0.4722          1.0322            1.0310        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_655.67     1_438.52    12_094.19       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_655.67     1_429.96    12_085.63       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_655.67     1_739.02    12_394.70       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16 (self)                              10_655.67     8_942.47    19_598.14       0.2676          1.0813            1.0870         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_945.81     1_601.46    14_547.26       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_945.81     1_805.07    14_750.88       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_945.81     2_201.80    15_147.60       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32 (self)                              12_945.81    10_552.82    23_498.63       0.2763          1.0771            1.0817         5.93
IVF-OPQ-nl223-m64-np11 (query)                        18_469.25     2_075.28    20_544.53       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64-np14 (query)                        18_469.25     2_366.82    20_836.07       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64-np21 (query)                        18_469.25     3_079.77    21_549.02       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64 (self)                              18_469.25    13_664.47    32_133.72       0.3239          1.0609            1.0630         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_597.65     3_210.95    29_808.60       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_597.65     3_774.97    30_372.62       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_597.65     5_127.00    31_724.65       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128 (self)                             26_597.65    20_364.99    46_962.64       0.4799          1.0308            1.0299        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_202.50     1_467.11    11_669.62       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_202.50     1_541.89    11_744.39       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_202.50     1_839.57    12_042.07       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16 (self)                              10_202.50     9_347.63    19_550.13       0.2690          1.0802            1.0862         6.19
IVF-OPQ-nl316-m32-np15 (query)                        12_626.80     1_828.33    14_455.13       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32-np17 (query)                        12_626.80     1_959.01    14_585.81       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32-np25 (query)                        12_626.80     2_434.20    15_061.00       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32 (self)                              12_626.80    11_344.10    23_970.89       0.2788          1.0756            1.0804         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_345.56     2_455.11    19_800.67       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_345.56     2_651.21    19_996.77       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_345.56     3_484.41    20_829.97       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64 (self)                              17_345.56    15_002.92    32_348.48       0.3275          1.0596            1.0619         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_980.61     3_930.76    30_911.37       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_980.61     4_299.88    31_280.48       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_980.61     5_833.74    32_814.35       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128 (self)                             26_980.61    23_085.49    50_066.10       0.4862          1.0295            1.0291        11.54
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
Exhaustive (query)                                        33.30       726.41       759.71       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.30     2_376.16     2_409.46       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_653.32       737.26     4_390.58       0.3009          1.2503            1.2433         1.26
Exhaustive-OPQ-m16 (self)                              3_653.32     2_726.64     6_379.96       0.2368          1.3778            1.3714         1.26
Exhaustive-OPQ-m32 (query)                             5_823.22     1_616.62     7_439.85       0.4204          1.1526            1.1474         2.03
Exhaustive-OPQ-m32 (self)                              5_823.22     5_524.41    11_347.64       0.3378          1.2478            1.2416         2.03
Exhaustive-OPQ-m64 (query)                             8_972.57     3_658.96    12_631.53       0.5662          1.0765            1.0733         3.55
Exhaustive-OPQ-m64 (self)                              8_972.57    12_678.56    21_651.13       0.4876          1.1287            1.1241         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_745.20       266.19     4_011.39       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_745.20       386.57     4_131.77       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_745.20       508.53     4_253.73       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16 (self)                               3_745.20     2_037.79     5_782.99       0.6197          1.0627            1.0600         1.67
IVF-OPQ-nl158-m32-np7 (query)                          5_794.82       443.31     6_238.13       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np12 (query)                         5_794.82       648.42     6_443.24       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np17 (query)                         5_794.82       881.63     6_676.44       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32 (self)                               5_794.82     3_290.53     9_085.35       0.7470          1.0260            1.0240         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_451.83       710.78    10_162.62       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_451.83     1_106.34    10_558.18       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_451.83     1_495.20    10_947.04       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64 (self)                               9_451.83     5_311.52    14_763.36       0.8304          1.0113            1.0097         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_142.55       357.81     4_500.36       0.7067          1.0311            1.0296         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_142.55       432.54     4_575.09       0.7068          1.0311            1.0295         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_142.55       638.41     4_780.96       0.7068          1.0311            1.0295         1.73
IVF-OPQ-nl223-m16 (self)                               4_142.55     2_404.75     6_547.30       0.6288          1.0595            1.0570         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_606.10       600.10     7_206.20       0.8046          1.0131            1.0120         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_606.10       725.96     7_332.07       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_606.10     1_058.41     7_664.52       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32 (self)                               6_606.10     3_861.54    10_467.64       0.7549          1.0243            1.0225         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_471.24       995.76    10_466.99       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_471.24     1_241.13    10_712.37       0.8638          1.0061            1.0051         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_471.24     1_798.85    11_270.09       0.8638          1.0061            1.0051         4.02
IVF-OPQ-nl223-m64 (self)                               9_471.24     6_333.35    15_804.58       0.8352          1.0106            1.0092         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_141.89       450.52     4_592.40       0.7112          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_141.89       497.06     4_638.95       0.7113          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_141.89       688.65     4_830.54       0.7113          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16 (self)                               4_141.89     2_645.39     6_787.28       0.6343          1.0576            1.0549         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_756.69       804.15     7_560.84       0.8076          1.0128            1.0115         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_756.69       852.63     7_609.32       0.8077          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_756.69     1_218.90     7_975.59       0.8077          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32 (self)                               6_756.69     4_378.20    11_134.89       0.7581          1.0237            1.0219         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_772.11     1_269.74    11_041.85       0.8662          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_772.11     1_431.53    11_203.64       0.8663          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_772.11     2_049.38    11_821.49       0.8663          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               9_772.11     7_160.18    16_932.29       0.8373          1.0103            1.0090         4.36
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
Exhaustive (query)                                        69.61     1_278.76     1_348.37       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.61     4_285.14     4_354.75       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_813.40     1_017.74     6_831.14       0.2317          1.2142            1.2107         2.26
Exhaustive-OPQ-m16 (self)                              5_813.40     4_752.14    10_565.53       0.1879          1.2983            1.2985         2.26
Exhaustive-OPQ-m32 (query)                             7_790.85     1_866.06     9_656.91       0.3189          1.1505            1.1472         3.03
Exhaustive-OPQ-m32 (self)                              7_790.85     7_536.42    15_327.27       0.2588          1.2171            1.2145         3.03
Exhaustive-OPQ-m64 (query)                            13_252.80     4_112.60    17_365.41       0.4332          1.0939            1.0912         4.55
Exhaustive-OPQ-m64 (self)                             13_252.80    14_937.11    28_189.92       0.3620          1.1417            1.1393         4.55
Exhaustive-OPQ-m128 (query)                           18_517.12     8_270.03    26_787.14       0.5699          1.0489            1.0476         7.61
Exhaustive-OPQ-m128 (self)                            18_517.12    28_908.15    47_425.27       0.4998          1.0773            1.0755         7.61
IVF-OPQ-nl158-m16-np7 (query)                          5_966.26       601.37     6_567.64       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16-np12 (query)                         5_966.26       752.46     6_718.72       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16-np17 (query)                         5_966.26       917.59     6_883.85       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16 (self)                               5_966.26     4_545.47    10_511.73       0.4370          1.1017            1.0996         3.07
IVF-OPQ-nl158-m32-np7 (query)                          7_915.50       745.42     8_660.92       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32-np12 (query)                         7_915.50     1_008.45     8_923.95       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32-np17 (query)                         7_915.50     1_193.62     9_109.12       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32 (self)                               7_915.50     5_398.98    13_314.48       0.6083          1.0441            1.0419         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_466.59     1_074.79    13_541.38       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_466.59     1_497.68    13_964.27       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_466.59     1_956.03    14_422.63       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64 (self)                              12_466.59     7_988.44    20_455.03       0.7308          1.0199            1.0179         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_340.65     1_609.95    19_950.60       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_340.65     2_325.22    20_665.87       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_340.65     3_064.29    21_404.94       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128 (self)                             18_340.65    11_754.47    30_095.12       0.8095          1.0099            1.0081         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_130.99       735.45     6_866.44       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_130.99       809.12     6_940.11       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_130.99     1_037.83     7_168.82       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16 (self)                               6_130.99     4_861.00    10_991.99       0.4482          1.0970            1.0949         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_221.80       926.83     9_148.63       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_221.80     1_083.69     9_305.49       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_221.80     1_366.98     9_588.78       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32 (self)                               8_221.80     5_929.76    14_151.56       0.6178          1.0417            1.0398         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_729.95     1_398.36    14_128.31       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_729.95     1_654.65    14_384.60       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_729.95     2_395.92    15_125.87       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64 (self)                              12_729.95     9_005.49    21_735.44       0.7366          1.0189            1.0172         5.49
IVF-OPQ-nl223-m128-np11 (query)                       19_141.67     2_165.78    21_307.45       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np14 (query)                       19_141.67     2_602.70    21_744.37       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np21 (query)                       19_141.67     3_659.52    22_801.19       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128 (self)                             19_141.67    14_645.84    33_787.51       0.8130          1.0094            1.0079         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_260.96       840.79     7_101.76       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_260.96       929.54     7_190.50       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_260.96     1_161.80     7_422.76       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16 (self)                               6_260.96     5_251.36    11_512.32       0.4530          1.0950            1.0932         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_288.21     1_080.11     9_368.31       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_288.21     1_165.22     9_453.42       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_288.21     1_525.06     9_813.27       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32 (self)                               8_288.21     6_556.91    14_845.12       0.6213          1.0409            1.0391         4.65
IVF-OPQ-nl316-m64-np15 (query)                        12_801.29     1_714.93    14_516.22       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np17 (query)                        12_801.29     1_882.06    14_683.35       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np25 (query)                        12_801.29     2_646.60    15_447.89       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64 (self)                              12_801.29    10_126.54    22_927.83       0.7386          1.0185            1.0169         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_968.64     2_707.50    21_676.14       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_968.64     3_017.50    21_986.15       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_968.64     4_183.04    23_151.68       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128 (self)                             18_968.64    15_445.57    34_414.21       0.8143          1.0092            1.0078         9.23
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
Exhaustive (query)                                       102.31     1_834.66     1_936.97       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.31     6_100.70     6_203.01       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_569.78     1_542.21    11_111.99       0.2295          1.2024            1.1985         3.76
Exhaustive-OPQ-m16 (self)                              9_569.78     8_271.23    17_841.01       0.1868          1.2974            1.2971         3.76
Exhaustive-OPQ-m32 (query)                            11_918.25     2_385.37    14_303.61       0.3123          1.1452            1.1413         4.53
Exhaustive-OPQ-m32 (self)                             11_918.25    11_060.89    22_979.13       0.2574          1.2173            1.2150         4.53
Exhaustive-OPQ-m64 (query)                            16_495.32     4_468.57    20_963.89       0.4084          1.0974            1.0946         6.05
Exhaustive-OPQ-m64 (self)                             16_495.32    17_930.42    34_425.74       0.3479          1.1498            1.1473         6.05
Exhaustive-OPQ-m128 (query)                           26_238.34     8_948.73    35_187.08       0.5283          1.0567            1.0548         9.11
Exhaustive-OPQ-m128 (self)                            26_238.34    32_629.53    58_867.87       0.4662          1.0907            1.0887         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_642.76     1_167.41    10_810.17       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_642.76     1_341.61    10_984.36       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_642.76     1_531.41    11_174.17       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16 (self)                               9_642.76     8_360.15    18_002.91       0.4254          1.1062            1.1043         4.98
IVF-OPQ-nl158-m32-np7 (query)                         12_090.70     1_351.43    13_442.13       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32-np12 (query)                        12_090.70     1_641.68    13_732.38       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32-np17 (query)                        12_090.70     1_949.97    14_040.67       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32 (self)                              12_090.70     9_783.84    21_874.55       0.6000          1.0460            1.0436         5.74
IVF-OPQ-nl158-m64-np7 (query)                         16_726.68     1_678.12    18_404.80       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64-np12 (query)                        16_726.68     2_175.57    18_902.25       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64-np17 (query)                        16_726.68     2_695.27    19_421.95       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64 (self)                              16_726.68    12_185.53    28_912.20       0.7240          1.0209            1.0186         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_990.08     2_478.14    28_468.22       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_990.08     3_388.84    29_378.93       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_990.08     4_340.05    30_330.13       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128 (self)                             25_990.08    17_766.96    43_757.05       0.8040          1.0106            1.0085        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_250.97     1_320.67    11_571.64       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_250.97     1_423.72    11_674.69       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_250.97     1_686.02    11_936.99       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16 (self)                              10_250.97     8_852.15    19_103.12       0.4406          1.0996            1.0977         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_656.82     1_591.22    15_248.05       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_656.82     1_792.96    15_449.78       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_656.82     2_201.78    15_858.61       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32 (self)                              13_656.82    10_555.03    24_211.85       0.6099          1.0434            1.0413         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_238.78     2_069.52    19_308.29       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_238.78     2_516.09    19_754.86       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_238.78     3_100.68    20_339.46       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64 (self)                              17_238.78    13_620.99    30_859.77       0.7314          1.0196            1.0177         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_898.60     3_186.22    30_084.81       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_898.60     3_765.58    30_664.18       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_898.60     5_133.37    32_031.96       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128 (self)                             26_898.60    20_434.27    47_332.86       0.8077          1.0100            1.0082        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_301.45     1_469.32    11_770.76       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_301.45     1_547.14    11_848.58       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_301.45     1_840.29    12_141.74       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16 (self)                              10_301.45     9_381.11    19_682.55       0.4446          1.0980            1.0962         6.19
IVF-OPQ-nl316-m32-np15 (query)                        12_625.69     1_840.03    14_465.73       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32-np17 (query)                        12_625.69     1_951.81    14_577.51       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32-np25 (query)                        12_625.69     2_444.75    15_070.45       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32 (self)                              12_625.69    11_355.99    23_981.69       0.6139          1.0423            1.0405         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_390.86     2_454.84    19_845.70       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_390.86     2_658.94    20_049.81       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_390.86     3_454.65    20_845.51       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64 (self)                              17_390.86    14_831.43    32_222.29       0.7332          1.0192            1.0176         8.48
IVF-OPQ-nl316-m128-np15 (query)                       27_358.05     3_904.79    31_262.84       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np17 (query)                       27_358.05     4_285.96    31_644.00       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np25 (query)                       27_358.05     5_880.16    33_238.20       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128 (self)                             27_358.05    22_728.03    50_086.07       0.8088          1.0097            1.0082        11.54
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
Exhaustive (query)                                        34.70       741.18       775.88       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.70     2_413.89     2_448.59       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_957.49       750.29     4_707.78       0.7911          1.0819            1.0684         1.26
Exhaustive-OPQ-m16 (self)                              3_957.49     2_788.02     6_745.50       0.7232          1.1502            1.1255         1.26
Exhaustive-OPQ-m32 (query)                             6_445.38     1_600.69     8_046.07       0.8303          1.0536            1.0424         2.03
Exhaustive-OPQ-m32 (self)                              6_445.38     5_626.46    12_071.83       0.7763          1.0975            1.0767         2.03
Exhaustive-OPQ-m64 (query)                             9_576.69     3_852.43    13_429.12       0.8562          1.0398            1.0292         3.55
Exhaustive-OPQ-m64 (self)                              9_576.69    12_669.72    22_246.42       0.8092          1.0723            1.0534         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_153.47       325.29     4_478.77       0.8907          1.0205            1.0162         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_153.47       423.86     4_577.34       0.8914          1.0201            1.0160         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_153.47       564.82     4_718.29       0.8914          1.0201            1.0161         1.67
IVF-OPQ-nl158-m16 (self)                               4_153.47     2_237.94     6_391.42       0.8482          1.0400            1.0322         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_521.10       475.51     6_996.61       0.9109          1.0134            1.0098         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_521.10       762.45     7_283.55       0.9118          1.0130            1.0097         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_521.10     1_012.90     7_534.00       0.9118          1.0129            1.0097         2.43
IVF-OPQ-nl158-m32 (self)                               6_521.10     3_762.46    10_283.56       0.8768          1.0257            1.0197         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_643.23       800.46    10_443.69       0.9243          1.0097            1.0066         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_643.23     1_314.07    10_957.31       0.9251          1.0093            1.0064         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_643.23     1_819.13    11_462.37       0.9251          1.0093            1.0064         3.96
IVF-OPQ-nl158-m64 (self)                               9_643.23     6_342.04    15_985.27       0.8964          1.0184            1.0132         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_488.95       372.41     4_861.35       0.8976          1.0178            1.0137         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_488.95       453.17     4_942.12       0.8978          1.0177            1.0137         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_488.95       649.56     5_138.50       0.8978          1.0177            1.0137         1.73
IVF-OPQ-nl223-m16 (self)                               4_488.95     2_462.52     6_951.46       0.8577          1.0353            1.0274         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_799.51       627.89     7_427.41       0.9156          1.0118            1.0087         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_799.51       778.80     7_578.31       0.9158          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_799.51     1_114.50     7_914.01       0.9158          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32 (self)                               6_799.51     4_043.61    10_843.12       0.8834          1.0232            1.0174         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_405.10     1_044.20    10_449.31       0.9269          1.0091            1.0058         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_405.10     1_303.92    10_709.03       0.9270          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_405.10     1_952.32    11_357.43       0.9271          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64 (self)                               9_405.10     6_706.90    16_112.00       0.8991          1.0176            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_055.22       461.62     4_516.84       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_055.22       521.86     4_577.08       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_055.22       715.63     4_770.85       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16 (self)                               4_055.22     2_742.61     6_797.82       0.8627          1.0334            1.0249         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_345.94       782.57     7_128.51       0.9170          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_345.94       873.96     7_219.90       0.9170          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_345.94     1_241.43     7_587.37       0.9171          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32 (self)                               6_345.94     4_432.11    10_778.05       0.8845          1.0229            1.0168         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_507.37     1_307.17    10_814.54       0.9287          1.0086            1.0056         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_507.37     1_470.74    10_978.11       0.9288          1.0085            1.0056         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_507.37     2_142.56    11_649.93       0.9288          1.0085            1.0056         4.36
IVF-OPQ-nl316-m64 (self)                               9_507.37     7_468.21    16_975.59       0.9011          1.0169            1.0116         4.36
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
Exhaustive (query)                                        68.77     1_296.82     1_365.59       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.77     4_305.94     4_374.71       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_909.17     1_026.22     6_935.39       0.7546          1.1136            1.0983         2.26
Exhaustive-OPQ-m16 (self)                              5_909.17     4_779.26    10_688.43       0.6788          1.2037            1.1739         2.26
Exhaustive-OPQ-m32 (query)                             7_893.33     1_884.40     9_777.72       0.8064          1.0692            1.0572         3.03
Exhaustive-OPQ-m32 (self)                              7_893.33     7_646.79    15_540.12       0.7455          1.1245            1.1019         3.03
Exhaustive-OPQ-m64 (query)                            12_492.94     3_955.80    16_448.74       0.8413          1.0455            1.0364         4.55
Exhaustive-OPQ-m64 (self)                             12_492.94    14_478.59    26_971.53       0.7916          1.0819            1.0654         4.55
Exhaustive-OPQ-m128 (query)                           18_622.99     8_209.73    26_832.72       0.9198          1.0107            1.0069         7.61
Exhaustive-OPQ-m128 (self)                            18_622.99    28_786.66    47_409.65       0.8933          1.0192            1.0139         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_272.76       626.65     6_899.41       0.8873          1.0229            1.0172         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_272.76       785.60     7_058.35       0.8876          1.0228            1.0171         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_272.76       992.63     7_265.38       0.8876          1.0227            1.0171         3.07
IVF-OPQ-nl158-m16 (self)                               6_272.76     4_694.19    10_966.94       0.8423          1.0463            1.0331         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_077.77       781.66     8_859.42       0.9007          1.0173            1.0129         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_077.77     1_093.75     9_171.52       0.9011          1.0171            1.0129         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_077.77     1_429.53     9_507.30       0.9011          1.0171            1.0129         3.84
IVF-OPQ-nl158-m32 (self)                               8_077.77     5_868.12    13_945.89       0.8619          1.0347            1.0247         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_565.17     1_150.84    13_716.01       0.9101          1.0145            1.0102         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_565.17     1_694.38    14_259.56       0.9106          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_565.17     2_240.42    14_805.59       0.9106          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64 (self)                              12_565.17     8_866.91    21_432.08       0.8751          1.0284            1.0196         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_445.00     1_780.85    20_225.85       0.9613          1.0029            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_445.00     2_761.83    21_206.83       0.9620          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_445.00     3_743.45    22_188.45       0.9621          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                             18_445.00    13_945.04    32_390.03       0.9450          1.0056            1.0016         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_326.64       728.31     7_054.95       0.8976          1.0194            1.0144         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_326.64       833.12     7_159.76       0.8977          1.0194            1.0144         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_326.64     1_048.97     7_375.61       0.8977          1.0194            1.0143         3.20
IVF-OPQ-nl223-m16 (self)                               6_326.64     4_900.81    11_227.44       0.8550          1.0394            1.0271         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_622.92       953.16     9_576.09       0.9080          1.0153            1.0107         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_622.92     1_101.72     9_724.64       0.9081          1.0153            1.0107         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_622.92     1_460.11    10_083.03       0.9081          1.0152            1.0107         3.96
IVF-OPQ-nl223-m32 (self)                               8_622.92     6_262.11    14_885.03       0.8702          1.0311            1.0211         3.96
IVF-OPQ-nl223-m64-np11 (query)                        13_018.14     1_487.01    14_505.15       0.9168          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64-np14 (query)                        13_018.14     1_756.91    14_775.05       0.9169          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64-np21 (query)                        13_018.14     2_458.51    15_476.65       0.9169          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64 (self)                              13_018.14     9_536.43    22_554.57       0.8830          1.0245            1.0169         5.49
IVF-OPQ-nl223-m128-np11 (query)                       19_221.31     2_307.40    21_528.72       0.9658          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                       19_221.31     2_813.48    22_034.80       0.9659          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                       19_221.31     4_030.89    23_252.21       0.9660          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                             19_221.31    14_889.41    34_110.72       0.9485          1.0050            1.0009         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_566.33       854.79     7_421.12       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_566.33       909.72     7_476.06       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_566.33     1_177.84     7_744.17       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16 (self)                               6_566.33     5_346.22    11_912.55       0.8659          1.0333            1.0225         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_519.32     1_119.92     9_639.24       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_519.32     1_217.30     9_736.62       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_519.32     1_671.24    10_190.56       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32 (self)                               8_519.32     7_199.77    15_719.09       0.8789          1.0266            1.0180         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_142.63     1_885.03    15_027.66       0.9210          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_142.63     2_006.62    15_149.25       0.9210          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_142.63     2_674.09    15_816.72       0.9211          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64 (self)                              13_142.63    10_276.56    23_419.19       0.8886          1.0222            1.0150         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_513.28     2_790.30    21_303.58       0.9689          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_513.28     3_142.16    21_655.44       0.9690          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_513.28     4_386.84    22_900.12       0.9690          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                             18_513.28    16_156.48    34_669.77       0.9519          1.0045            1.0005         9.23
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
Exhaustive (query)                                       102.16     1_840.47     1_942.63       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.16     6_126.50     6_228.66       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_675.10     1_538.92    11_214.02       0.7383          1.1306            1.1121         3.76
Exhaustive-OPQ-m16 (self)                              9_675.10     8_294.02    17_969.12       0.6595          1.2295            1.1962         3.76
Exhaustive-OPQ-m32 (query)                            12_000.09     2_398.99    14_399.09       0.8493          1.0411            1.0321         4.53
Exhaustive-OPQ-m32 (self)                             12_000.09    11_067.12    23_067.21       0.8006          1.0714            1.0580         4.53
Exhaustive-OPQ-m64 (query)                            16_621.80     4_519.78    21_141.58       0.8796          1.0255            1.0186         6.05
Exhaustive-OPQ-m64 (self)                             16_621.80    17_963.71    34_585.52       0.8413          1.0441            1.0341         6.05
Exhaustive-OPQ-m128 (query)                           26_379.60     8_799.34    35_178.94       0.9051          1.0148            1.0104         9.11
Exhaustive-OPQ-m128 (self)                            26_379.60    32_543.25    58_922.86       0.8741          1.0264            1.0201         9.11
IVF-OPQ-nl158-m16-np7 (query)                         10_103.51     1_178.31    11_281.82       0.8919          1.0221            1.0161         4.98
IVF-OPQ-nl158-m16-np12 (query)                        10_103.51     1_389.95    11_493.46       0.8921          1.0220            1.0160         4.98
IVF-OPQ-nl158-m16-np17 (query)                        10_103.51     1_606.12    11_709.63       0.8921          1.0220            1.0160         4.98
IVF-OPQ-nl158-m16 (self)                              10_103.51     8_569.74    18_673.26       0.8469          1.0440            1.0305         4.98
IVF-OPQ-nl158-m32-np7 (query)                         12_415.45     1_380.79    13_796.24       0.9368          1.0085            1.0035         5.74
IVF-OPQ-nl158-m32-np12 (query)                        12_415.45     1_722.43    14_137.88       0.9370          1.0084            1.0035         5.74
IVF-OPQ-nl158-m32-np17 (query)                        12_415.45     2_087.23    14_502.68       0.9370          1.0084            1.0035         5.74
IVF-OPQ-nl158-m32 (self)                              12_415.45    10_274.61    22_690.06       0.9075          1.0183            1.0075         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_009.52     1_760.79    18_770.32       0.9510          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_009.52     2_381.82    19_391.34       0.9513          1.0050            1.0012         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_009.52     3_146.65    20_156.17       0.9513          1.0050            1.0012         7.27
IVF-OPQ-nl158-m64 (self)                              17_009.52    13_159.06    30_168.58       0.9271          1.0115            1.0036         7.27
IVF-OPQ-nl158-m128-np7 (query)                        26_259.31     2_776.45    29_035.76       0.9609          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128-np12 (query)                       26_259.31     3_844.06    30_103.37       0.9611          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128-np17 (query)                       26_259.31     5_024.39    31_283.70       0.9611          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                             26_259.31    20_015.50    46_274.81       0.9395          1.0077            1.0017        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_598.29     1_326.24    11_924.53       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_598.29     1_451.00    12_049.29       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_598.29     1_720.21    12_318.49       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16 (self)                              10_598.29     8_922.03    19_520.31       0.8590          1.0364            1.0253         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_889.13     1_617.36    14_506.50       0.9424          1.0072            1.0027         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_889.13     1_801.24    14_690.37       0.9425          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_889.13     2_259.94    15_149.08       0.9425          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32 (self)                              12_889.13    10_755.88    23_645.01       0.9162          1.0147            1.0058         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_533.46     2_125.71    19_659.17       0.9546          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_533.46     2_449.88    19_983.35       0.9547          1.0045            1.0007         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_533.46     3_249.12    20_782.59       0.9547          1.0045            1.0007         7.46
IVF-OPQ-nl223-m64 (self)                              17_533.46    14_213.18    31_746.65       0.9327          1.0098            1.0027         7.46
IVF-OPQ-nl223-m128-np11 (query)                       27_258.60     3_314.64    30_573.24       0.9638          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                       27_258.60     3_963.69    31_222.28       0.9639          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                       27_258.60     5_488.25    32_746.85       0.9639          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                             27_258.60    21_570.36    48_828.95       0.9434          1.0068            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_953.70     1_464.36    12_418.06       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_953.70     1_539.45    12_493.14       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_953.70     1_859.82    12_813.52       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16 (self)                              10_953.70     9_426.78    20_380.47       0.8645          1.0338            1.0231         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_164.01     1_838.20    15_002.20       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_164.01     1_961.50    15_125.51       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_164.01     2_490.60    15_654.61       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32 (self)                              13_164.01    11_493.46    24_657.46       0.9193          1.0137            1.0052         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_986.76     2_487.65    20_474.41       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_986.76     2_699.16    20_685.91       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_986.76     3_564.78    21_551.54       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64 (self)                              17_986.76    15_285.28    33_272.04       0.9359          1.0090            1.0023         8.48
IVF-OPQ-nl316-m128-np15 (query)                       27_856.29     4_288.63    32_144.92       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np17 (query)                       27_856.29     4_819.34    32_675.63       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np25 (query)                       27_856.29     6_122.80    33_979.09       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128 (self)                             27_856.29    23_601.89    51_458.18       0.9462          1.0061            1.0009        11.54
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
Exhaustive (query)                                        69.35     1_230.67     1_300.02       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_493.44       129.41     1_622.85       0.3468          1.0755            1.0751         2.24
IVFPQ-m64-nl111-np1                                    2_291.61       231.69     2_523.30       0.4500          1.0504            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_554.01       134.91     1_688.92       0.3219          1.2732            1.0774         4.72
IVFPQ-m32-nl111-np2                                    1_493.44       177.31     1_670.75       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np2                                    2_291.61       328.63     2_620.24       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_554.01       182.61     1_736.62       0.3477          1.0764            1.0749         4.72
IVFPQ-m32-nl111-np4                                    1_493.44       271.00     1_764.44       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np4                                    2_291.61       491.71     2_783.32       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_554.01       277.44     1_831.45       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np5                                    1_493.44       313.23     1_806.67       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np5                                    2_291.61       580.29     2_871.90       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_554.01       335.74     1_889.75       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np8                                    1_493.44       460.30     1_953.74       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np8                                    2_291.61       845.41     3_137.01       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_554.01       481.17     2_035.17       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np10                                   1_493.44       584.63     2_078.07       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np10                                   2_291.61     1_018.41     3_310.02       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_554.01       578.08     2_132.09       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl158-np1                                    1_537.44       126.25     1_663.69       0.3490          1.0730            1.0736         2.34
IVFPQ-m64-nl158-np1                                    2_422.59       225.49     2_648.09       0.4593          1.0469            1.0428         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_732.51       131.90     1_864.41       0.3165          1.1950            1.0771         4.82
IVFPQ-m32-nl158-np2                                    1_537.44       179.85     1_717.28       0.3526          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np2                                    2_422.59       316.06     2_738.65       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_732.51       186.57     1_919.07       0.3520          1.0726            1.0731         4.82
IVFPQ-m32-nl158-np4                                    1_537.44       264.34     1_801.77       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np4                                    2_422.59       501.00     2_923.59       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_732.51       274.15     2_006.66       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np7                                    1_537.44       399.05     1_936.49       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np7                                    2_422.59       755.36     3_177.95       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_732.51       410.98     2_143.49       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np8                                    1_537.44       448.74     1_986.17       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np8                                    2_422.59       829.94     3_252.53       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_732.51       462.87     2_195.37       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np12                                   1_537.44       647.12     2_184.55       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np12                                   2_422.59     1_186.86     3_609.45       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_732.51       667.60     2_400.10       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl223-np1                                    1_559.35       105.37     1_664.72       0.3524          1.0705            1.0702         2.46
IVFPQ-m64-nl223-np1                                    2_456.07       163.93     2_620.00       0.4383          1.0504            1.0448         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_779.89       113.72     1_893.61       0.3306          1.1705            1.0732         4.95
IVFPQ-m32-nl223-np2                                    1_559.35       155.75     1_715.10       0.3660          1.0664            1.0666         2.46
IVFPQ-m64-nl223-np2                                    2_456.07       262.33     2_718.40       0.4668          1.0444            1.0401         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_779.89       172.68     1_952.57       0.3633          1.0686            1.0686         4.95
IVFPQ-m32-nl223-np4                                    1_559.35       258.68     1_818.03       0.3690          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np4                                    2_456.07       455.08     2_911.15       0.4762          1.0430            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_779.89       272.85     2_052.74       0.3673          1.0667            1.0665         4.95
IVFPQ-m32-nl223-np8                                    1_559.35       446.87     2_006.22       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np8                                    2_456.07       803.84     3_259.91       0.4775          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_779.89       465.78     2_245.66       0.3692          1.0656            1.0657         4.95
IVFPQ-m32-nl223-np11                                   1_559.35       592.26     2_151.61       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np11                                   2_456.07     1_084.28     3_540.35       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_779.89       618.40     2_398.29       0.3693          1.0656            1.0657         4.95
IVFPQ-m32-nl223-np14                                   1_559.35       740.61     2_299.96       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np14                                   2_456.07     1_333.06     3_789.13       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_779.89       770.51     2_550.40       0.3693          1.0656            1.0657         4.95
IVFPQ-m32-nl316-np1                                    1_738.50       104.94     1_843.44       0.3529          1.0694            1.0685         2.65
IVFPQ-m64-nl316-np1                                    2_684.80       150.24     2_835.05       0.4277          1.0517            1.0458         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_969.26       110.88     2_080.14       0.3254          1.1561            1.0727         5.13
IVFPQ-m32-nl316-np2                                    1_738.50       162.19     1_900.69       0.3727          1.0626            1.0636         2.65
IVFPQ-m64-nl316-np2                                    2_684.80       247.18     2_931.99       0.4696          1.0426            1.0391         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_969.26       168.27     2_137.53       0.3692          1.0659            1.0659         5.13
IVFPQ-m32-nl316-np4                                    1_738.50       254.54     1_993.04       0.3787          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np4                                    2_684.80       437.59     3_122.39       0.4855          1.0401            1.0368         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_969.26       271.16     2_240.42       0.3751          1.0632            1.0637         5.13
IVFPQ-m32-nl316-np8                                    1_738.50       453.03     2_191.53       0.3797          1.0610            1.0619         2.65
IVFPQ-m64-nl316-np8                                    2_684.80       800.84     3_485.64       0.4890          1.0397            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_969.26       470.11     2_439.37       0.3789          1.0615            1.0622         5.13
IVFPQ-m32-nl316-np15                                   1_738.50       796.85     2_535.35       0.3797          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np15                                   2_684.80     1_411.44     4_096.24       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_969.26       812.29     2_781.55       0.3797          1.0610            1.0618         5.13
IVFPQ-m32-nl316-np17                                   1_738.50       900.86     2_639.36       0.3798          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np17                                   2_684.80     1_603.72     4_288.52       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_969.26       917.52     2_886.78       0.3798          1.0610            1.0618         5.13
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
Exhaustive (query)                                        69.35     1_230.67     1_300.02       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_738.71       131.55     1_870.26       0.3175          1.1920            1.0770         4.82
SOARPQ-near-np2                                        1_738.71       182.38     1_921.09       0.3523          1.0721            1.0731         4.82
SOARPQ-near-np4                                        1_738.71       276.48     2_015.19       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np7                                        1_738.71       419.61     2_158.32       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np8                                        1_738.71       473.77     2_212.48       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np12                                       1_738.71       678.03     2_416.74       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np1                                    1_911.10       151.13     2_062.23       0.3167          1.1944            1.0771         4.82
SOARPQ-shift0.3-np2                                    1_911.10       183.00     2_094.10       0.3521          1.0725            1.0731         4.82
SOARPQ-shift0.3-np4                                    1_911.10       294.38     2_205.48       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np7                                    1_911.10       428.37     2_339.47       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np8                                    1_911.10       490.44     2_401.54       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np12                                   1_911.10       683.48     2_594.58       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np1                                    2_048.74       135.48     2_184.23       0.3164          1.1957            1.0771         4.82
SOARPQ-shift0.7-np2                                    2_048.74       181.87     2_230.61       0.3519          1.0730            1.0731         4.82
SOARPQ-shift0.7-np4                                    2_048.74       276.46     2_325.20       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np7                                    2_048.74       428.28     2_477.02       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np8                                    2_048.74       479.59     2_528.33       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np12                                   2_048.74       692.73     2_741.47       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np1                                       1_983.21       132.54     2_115.75       0.3179          1.1930            1.0769         4.82
SOARPQ-orth1-np2                                       1_983.21       188.08     2_171.30       0.3525          1.0718            1.0730         4.82
SOARPQ-orth1-np4                                       1_983.21       280.34     2_263.55       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np7                                       1_983.21       438.88     2_422.09       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np8                                       1_983.21       488.27     2_471.49       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np12                                      1_983.21       681.35     2_664.56       0.3527          1.0715            1.0730         4.82
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
Exhaustive (query)                                        69.99     1_397.97     1_467.96       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_598.27       125.71     1_723.98       0.4790          1.0776            1.0739         2.24
IVFPQ-m64-nl111-np1                                    2_591.99       219.66     2_811.65       0.6165          1.0395            1.0353         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_755.21       130.01     1_885.22       0.4659          1.0980            1.0769         4.72
IVFPQ-m32-nl111-np2                                    1_598.27       172.44     1_770.71       0.4838          1.0757            1.0732         2.24
IVFPQ-m64-nl111-np2                                    2_591.99       308.29     2_900.28       0.6244          1.0367            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_755.21       218.08     1_973.29       0.4828          1.0776            1.0736         4.72
IVFPQ-m32-nl111-np4                                    1_598.27       269.96     1_868.23       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np4                                    2_591.99       491.25     3_083.24       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_755.21       348.87     2_104.08       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np5                                    1_598.27       332.63     1_930.90       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np5                                    2_591.99       581.52     3_173.51       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_755.21       331.17     2_086.38       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np8                                    1_598.27       475.24     2_073.51       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np8                                    2_591.99       853.29     3_445.28       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_755.21       520.17     2_275.38       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np10                                   1_598.27       598.11     2_196.38       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np10                                   2_591.99     1_038.94     3_630.93       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_755.21       625.83     2_381.04       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl158-np1                                    1_724.64       131.22     1_855.86       0.4830          1.0756            1.0721         2.34
IVFPQ-m64-nl158-np1                                    2_601.64       225.90     2_827.53       0.6147          1.0403            1.0347         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_959.44       134.52     2_093.96       0.4858          1.0778            1.0727         4.82
IVFPQ-m32-nl158-np2                                    1_724.64       177.51     1_902.15       0.4914          1.0723            1.0710         2.34
IVFPQ-m64-nl158-np2                                    2_601.64       308.42     2_910.06       0.6281          1.0356            1.0337         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_959.44       181.81     2_141.25       0.4912          1.0727            1.0713         4.82
IVFPQ-m32-nl158-np4                                    1_724.64       280.42     2_005.07       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np4                                    2_601.64       484.14     3_085.77       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_959.44       295.87     2_255.31       0.4919          1.0721            1.0709         4.82
IVFPQ-m32-nl158-np7                                    1_724.64       427.07     2_151.71       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np7                                    2_601.64       749.64     3_351.27       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_959.44       431.35     2_390.79       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np8                                    1_724.64       497.89     2_222.53       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np8                                    2_601.64       832.09     3_433.72       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_959.44       488.31     2_447.74       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np12                                   1_724.64       673.07     2_397.71       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np12                                   2_601.64     1_193.40     3_795.03       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_959.44       692.40     2_651.84       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl223-np1                                    1_838.80       110.66     1_949.46       0.3968          1.1044            1.1004         2.46
IVFPQ-m64-nl223-np1                                    2_852.76       167.91     3_020.67       0.4720          1.0736            1.0668         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_962.91       119.61     2_082.52       0.4490          1.0881            1.0842         4.95
IVFPQ-m32-nl223-np2                                    1_838.80       171.56     2_010.36       0.4582          1.0822            1.0800         2.46
IVFPQ-m64-nl223-np2                                    2_852.76       267.25     3_120.02       0.5701          1.0475            1.0435         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_962.91       182.24     2_145.15       0.4798          1.0760            1.0744         4.95
IVFPQ-m32-nl223-np4                                    1_838.80       265.32     2_104.12       0.4852          1.0739            1.0723         2.46
IVFPQ-m64-nl223-np4                                    2_852.76       486.39     3_339.15       0.6182          1.0373            1.0354         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_962.91       295.35     2_258.26       0.4900          1.0725            1.0712         4.95
IVFPQ-m32-nl223-np8                                    1_838.80       481.17     2_319.97       0.4917          1.0720            1.0707         2.46
IVFPQ-m64-nl223-np8                                    2_852.76       842.32     3_695.08       0.6323          1.0346            1.0332         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_962.91       489.09     2_452.00       0.4919          1.0720            1.0707         4.95
IVFPQ-m32-nl223-np11                                   1_838.80       657.93     2_496.73       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np11                                   2_852.76     1_119.35     3_972.11       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_962.91       643.21     2_606.12       0.4920          1.0719            1.0706         4.95
IVFPQ-m32-nl223-np14                                   1_838.80       788.40     2_627.20       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np14                                   2_852.76     1_388.17     4_240.93       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_962.91       815.52     2_778.43       0.4920          1.0719            1.0706         4.95
IVFPQ-m32-nl316-np1                                    1_850.23       109.67     1_959.90       0.3503          1.1215            1.1185         2.65
IVFPQ-m64-nl316-np1                                    2_973.93       149.45     3_123.38       0.3989          1.0931            1.0885         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_152.84       116.20     2_269.05       0.4206          1.0956            1.0942         5.13
IVFPQ-m32-nl316-np2                                    1_850.23       156.43     2_006.66       0.4250          1.0922            1.0912         2.65
IVFPQ-m64-nl316-np2                                    2_973.93       248.13     3_222.06       0.5171          1.0589            1.0562         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_152.84       171.59     2_324.43       0.4612          1.0816            1.0804         5.13
IVFPQ-m32-nl316-np4                                    1_850.23       261.19     2_111.42       0.4685          1.0785            1.0771         2.65
IVFPQ-m64-nl316-np4                                    2_973.93       454.45     3_428.38       0.5954          1.0419            1.0397         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_152.84       281.36     2_434.20       0.4828          1.0746            1.0734         5.13
IVFPQ-m32-nl316-np8                                    1_850.23       467.03     2_317.25       0.4864          1.0733            1.0718         2.65
IVFPQ-m64-nl316-np8                                    2_973.93       817.67     3_791.60       0.6309          1.0349            1.0331         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_152.84       490.00     2_642.84       0.4879          1.0731            1.0717         5.13
IVFPQ-m32-nl316-np15                                   1_850.23       835.51     2_685.74       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np15                                   2_973.93     1_445.68     4_419.61       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_152.84       856.40     3_009.24       0.4881          1.0728            1.0715         5.13
IVFPQ-m32-nl316-np17                                   1_850.23       928.89     2_779.12       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np17                                   2_973.93     1_628.85     4_602.78       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_152.84       943.17     3_096.01       0.4881          1.0728            1.0715         5.13
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
Exhaustive (query)                                        69.99     1_397.97     1_467.96       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_000.51       141.03     2_141.54       0.4863          1.0774            1.0725         4.82
SOARPQ-near-np2                                        2_000.51       187.53     2_188.04       0.4913          1.0726            1.0713         4.82
SOARPQ-near-np4                                        2_000.51       277.19     2_277.70       0.4920          1.0721            1.0709         4.82
SOARPQ-near-np7                                        2_000.51       431.73     2_432.24       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np8                                        2_000.51       493.36     2_493.87       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np12                                       2_000.51       696.54     2_697.05       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np1                                    2_116.10       141.07     2_257.17       0.4861          1.0776            1.0726         4.82
SOARPQ-shift0.3-np2                                    2_116.10       213.56     2_329.66       0.4912          1.0726            1.0712         4.82
SOARPQ-shift0.3-np4                                    2_116.10       287.49     2_403.58       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.3-np7                                    2_116.10       428.28     2_544.38       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np8                                    2_116.10       482.88     2_598.97       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np12                                   2_116.10       707.25     2_823.35       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np1                                    2_105.18       137.23     2_242.41       0.4855          1.0779            1.0727         4.82
SOARPQ-shift0.7-np2                                    2_105.18       181.54     2_286.72       0.4911          1.0727            1.0713         4.82
SOARPQ-shift0.7-np4                                    2_105.18       274.53     2_379.71       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.7-np7                                    2_105.18       422.23     2_527.41       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np8                                    2_105.18       473.58     2_578.76       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np12                                   2_105.18       683.08     2_788.26       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np1                                       1_956.30       129.74     2_086.04       0.4855          1.0780            1.0728         4.82
SOARPQ-orth1-np2                                       1_956.30       182.61     2_138.90       0.4911          1.0728            1.0713         4.82
SOARPQ-orth1-np4                                       1_956.30       275.42     2_231.72       0.4919          1.0722            1.0709         4.82
SOARPQ-orth1-np7                                       1_956.30       430.15     2_386.45       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np8                                       1_956.30       482.38     2_438.68       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np12                                      1_956.30       699.89     2_656.19       0.4921          1.0720            1.0708         4.82
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
Exhaustive (query)                                        74.10     1_403.93     1_478.03       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_773.75       103.60     1_877.34       0.7050          1.1847            1.0985         2.24
IVFPQ-m64-nl111-np1                                    2_717.93       176.81     2_894.74       0.7176          1.1741            1.0872         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_935.37       128.00     2_063.38       0.8135          1.0743            1.0456         4.72
IVFPQ-m32-nl111-np2                                    1_773.75       164.46     1_938.20       0.8205          1.0642            1.0404         2.24
IVFPQ-m64-nl111-np2                                    2_717.93       292.84     3_010.77       0.8435          1.0523            1.0274         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_935.37       209.43     2_144.81       0.8481          1.0449            1.0335         4.72
IVFPQ-m32-nl111-np4                                    1_773.75       296.21     2_069.95       0.8534          1.0387            1.0308         2.24
IVFPQ-m64-nl111-np4                                    2_717.93       547.13     3_265.06       0.8808          1.0260            1.0194         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_935.37       380.43     2_315.80       0.8549          1.0391            1.0309         4.72
IVFPQ-m32-nl111-np5                                    1_773.75       362.40     2_136.15       0.8551          1.0376            1.0303         2.24
IVFPQ-m64-nl111-np5                                    2_717.93       691.91     3_409.84       0.8826          1.0249            1.0190         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_935.37       455.20     2_390.58       0.8554          1.0385            1.0305         4.72
IVFPQ-m32-nl111-np8                                    1_773.75       558.92     2_332.66       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np8                                    2_717.93     1_086.59     3_804.52       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_935.37       692.50     2_627.87       0.8561          1.0374            1.0301         4.72
IVFPQ-m32-nl111-np10                                   1_773.75       710.90     2_484.65       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np10                                   2_717.93     1_336.19     4_054.12       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_935.37       821.35     2_756.72       0.8561          1.0371            1.0300         4.72
IVFPQ-m32-nl158-np1                                    2_057.67       102.13     2_159.80       0.6971          1.1918            1.1067         2.34
IVFPQ-m64-nl158-np1                                    3_010.84       154.53     3_165.37       0.7064          1.1830            1.0984         3.86
SOARPQ-shift0.5-m32-nl158-np1                          2_120.70       122.90     2_243.61       0.8151          1.0761            1.0446         4.82
IVFPQ-m32-nl158-np2                                    2_057.67       160.62     2_218.28       0.8240          1.0636            1.0374         2.34
IVFPQ-m64-nl158-np2                                    3_010.84       265.33     3_276.17       0.8414          1.0541            1.0271         3.86
SOARPQ-shift0.5-m32-nl158-np2                          2_120.70       185.01     2_305.71       0.8573          1.0417            1.0292         4.82
IVFPQ-m32-nl158-np4                                    2_057.67       278.54     2_336.21       0.8638          1.0338            1.0259         2.34
IVFPQ-m64-nl158-np4                                    3_010.84       493.42     3_504.26       0.8850          1.0243            1.0175         3.86
SOARPQ-shift0.5-m32-nl158-np4                          2_120.70       330.65     2_451.35       0.8657          1.0347            1.0259         4.82
IVFPQ-m32-nl158-np7                                    2_057.67       454.02     2_511.68       0.8684          1.0311            1.0245         2.34
IVFPQ-m64-nl158-np7                                    3_010.84       840.42     3_851.26       0.8900          1.0215            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np7                          2_120.70       550.34     2_671.05       0.8680          1.0321            1.0248         4.82
IVFPQ-m32-nl158-np8                                    2_057.67       525.92     2_583.58       0.8685          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np8                                    3_010.84     1_003.16     4_014.00       0.8902          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np8                          2_120.70       608.12     2_728.82       0.8682          1.0317            1.0247         4.82
IVFPQ-m32-nl158-np12                                   2_057.67       768.96     2_826.63       0.8687          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np12                                   3_010.84     1_425.03     4_435.87       0.8903          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np12                         2_120.70       888.06     3_008.76       0.8686          1.0311            1.0245         4.82
IVFPQ-m32-nl223-np1                                    2_056.86       100.55     2_157.41       0.6865          1.1976            1.1231         2.46
IVFPQ-m64-nl223-np1                                    3_118.64       144.55     3_263.19       0.6934          1.1901            1.1152         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_289.67       105.89     2_395.56       0.8126          1.0799            1.0465         4.95
IVFPQ-m32-nl223-np2                                    2_056.86       149.09     2_205.95       0.8242          1.0647            1.0369         2.46
IVFPQ-m64-nl223-np2                                    3_118.64       245.14     3_363.78       0.8391          1.0564            1.0285         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_289.67       172.62     2_462.29       0.8641          1.0402            1.0265         4.95
IVFPQ-m32-nl223-np4                                    2_056.86       255.20     2_312.06       0.8725          1.0307            1.0222         2.46
IVFPQ-m64-nl223-np4                                    3_118.64       450.27     3_568.90       0.8923          1.0219            1.0149         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_289.67       303.88     2_593.55       0.8759          1.0314            1.0220         4.95
IVFPQ-m32-nl223-np8                                    2_056.86       497.86     2_554.72       0.8793          1.0270            1.0204         2.46
IVFPQ-m64-nl223-np8                                    3_118.64       863.36     3_981.99       0.8997          1.0180            1.0133         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_289.67       553.41     2_843.08       0.8788          1.0281            1.0206         4.95
IVFPQ-m32-nl223-np11                                   2_056.86       653.75     2_710.61       0.8796          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np11                                   3_118.64     1_170.38     4_289.02       0.9001          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_289.67       729.54     3_019.21       0.8794          1.0273            1.0203         4.95
IVFPQ-m32-nl223-np14                                   2_056.86       833.21     2_890.07       0.8797          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np14                                   3_118.64     1_495.78     4_614.42       0.9002          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_289.67       915.66     3_205.34       0.8796          1.0270            1.0203         4.95
IVFPQ-m32-nl316-np1                                    2_141.42       114.79     2_256.21       0.6730          1.2103            1.1374         2.65
IVFPQ-m64-nl316-np1                                    3_147.97       162.79     3_310.76       0.6777          1.2048            1.1331         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_404.15       115.62     2_519.77       0.8095          1.0841            1.0485         5.13
IVFPQ-m32-nl316-np2                                    2_141.42       149.24     2_290.66       0.8234          1.0660            1.0367         2.65
IVFPQ-m64-nl316-np2                                    3_147.97       234.23     3_382.20       0.8337          1.0605            1.0304         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_404.15       168.03     2_572.18       0.8708          1.0386            1.0233         5.13
IVFPQ-m32-nl316-np4                                    2_141.42       252.75     2_394.17       0.8823          1.0266            1.0185         2.65
IVFPQ-m64-nl316-np4                                    3_147.97       434.70     3_582.67       0.8968          1.0204            1.0133         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_404.15       275.50     2_679.66       0.8865          1.0283            1.0182         5.13
IVFPQ-m32-nl316-np8                                    2_141.42       465.76     2_607.18       0.8914          1.0217            1.0162         2.65
IVFPQ-m64-nl316-np8                                    3_147.97       828.24     3_976.22       0.9066          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_404.15       524.03     2_928.19       0.8906          1.0239            1.0167         5.13
IVFPQ-m32-nl316-np15                                   2_141.42       848.77     2_990.19       0.8921          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np15                                   3_147.97     1_479.05     4_627.03       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_404.15       907.60     3_311.76       0.8919          1.0218            1.0162         5.13
IVFPQ-m32-nl316-np17                                   2_141.42       949.07     3_090.50       0.8922          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np17                                   3_147.97     1_692.15     4_840.13       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_404.15     1_015.98     3_420.14       0.8920          1.0216            1.0161         5.13
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
Exhaustive (query)                                        74.10     1_403.93     1_478.03       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_147.13       114.97     2_262.10       0.8150          1.0767            1.0413         4.82
SOARPQ-near-np2                                        2_147.13       193.77     2_340.90       0.8584          1.0400            1.0282         4.82
SOARPQ-near-np4                                        2_147.13       328.90     2_476.03       0.8665          1.0333            1.0255         4.82
SOARPQ-near-np7                                        2_147.13       531.36     2_678.49       0.8682          1.0316            1.0247         4.82
SOARPQ-near-np8                                        2_147.13       594.00     2_741.13       0.8684          1.0314            1.0246         4.82
SOARPQ-near-np12                                       2_147.13       852.73     2_999.86       0.8687          1.0311            1.0244         4.82
SOARPQ-shift0.3-np1                                    2_147.25       116.89     2_264.14       0.8175          1.0739            1.0429         4.82
SOARPQ-shift0.3-np2                                    2_147.25       184.80     2_332.04       0.8583          1.0407            1.0288         4.82
SOARPQ-shift0.3-np4                                    2_147.25       327.94     2_475.19       0.8660          1.0342            1.0258         4.82
SOARPQ-shift0.3-np7                                    2_147.25       536.34     2_683.59       0.8681          1.0318            1.0248         4.82
SOARPQ-shift0.3-np8                                    2_147.25       594.53     2_741.78       0.8683          1.0315            1.0246         4.82
SOARPQ-shift0.3-np12                                   2_147.25       848.73     2_995.97       0.8686          1.0311            1.0245         4.82
SOARPQ-shift0.7-np1                                    2_106.30       115.59     2_221.89       0.8118          1.0794            1.0463         4.82
SOARPQ-shift0.7-np2                                    2_106.30       183.68     2_289.98       0.8562          1.0430            1.0297         4.82
SOARPQ-shift0.7-np4                                    2_106.30       321.14     2_427.43       0.8653          1.0353            1.0261         4.82
SOARPQ-shift0.7-np7                                    2_106.30       525.21     2_631.51       0.8679          1.0324            1.0249         4.82
SOARPQ-shift0.7-np8                                    2_106.30       592.02     2_698.31       0.8682          1.0319            1.0247         4.82
SOARPQ-shift0.7-np12                                   2_106.30       843.43     2_949.72       0.8686          1.0312            1.0245         4.82
SOARPQ-orth1-np1                                       2_021.24       113.84     2_135.08       0.8149          1.0770            1.0431         4.82
SOARPQ-orth1-np2                                       2_021.24       186.82     2_208.06       0.8579          1.0413            1.0287         4.82
SOARPQ-orth1-np4                                       2_021.24       326.58     2_347.82       0.8661          1.0343            1.0258         4.82
SOARPQ-orth1-np7                                       2_021.24       530.97     2_552.21       0.8681          1.0319            1.0248         4.82
SOARPQ-orth1-np8                                       2_021.24       606.33     2_627.57       0.8683          1.0316            1.0246         4.82
SOARPQ-orth1-np12                                      2_021.24       851.08     2_872.32       0.8686          1.0311            1.0244         4.82
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
Exhaustive (query)                                        72.87     1_385.38     1_458.25       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                    1_571.08        95.64     1_666.72       0.7674          1.1423            1.0643         2.24
IVFPQ-m64-nl111-np1                                    2_461.11       150.38     2_611.49       0.7738          1.1368            1.0562         3.77
SOARPQ-orth1-m32-nl111-np1                             1_755.62       111.61     1_867.23       0.8492          1.0681            1.0350         4.72
IVFPQ-m32-nl111-np2                                    1_571.08       151.00     1_722.08       0.8656          1.0441            1.0265         2.24
IVFPQ-m64-nl111-np2                                    2_461.11       265.28     2_726.39       0.8760          1.0383            1.0207         3.77
SOARPQ-orth1-m32-nl111-np2                             1_755.62       184.26     1_939.88       0.8782          1.0396            1.0248         4.72
IVFPQ-m32-nl111-np4                                    1_571.08       267.50     1_838.58       0.8857          1.0293            1.0218         2.24
IVFPQ-m64-nl111-np4                                    2_461.11       493.06     2_954.16       0.8968          1.0237            1.0165         3.77
SOARPQ-orth1-m32-nl111-np4                             1_755.62       326.43     2_082.05       0.8854          1.0317            1.0223         4.72
IVFPQ-m32-nl111-np5                                    1_571.08       329.00     1_900.08       0.8866          1.0289            1.0215         2.24
IVFPQ-m64-nl111-np5                                    2_461.11       613.00     3_074.10       0.8978          1.0232            1.0163         3.77
SOARPQ-orth1-m32-nl111-np5                             1_755.62       403.74     2_159.36       0.8862          1.0305            1.0219         4.72
IVFPQ-m32-nl111-np8                                    1_571.08       513.95     2_085.03       0.8872          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np8                                    2_461.11       968.81     3_429.92       0.8984          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np8                             1_755.62       614.59     2_370.21       0.8871          1.0290            1.0215         4.72
IVFPQ-m32-nl111-np10                                   1_571.08       638.31     2_209.39       0.8873          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np10                                   2_461.11     1_210.51     3_671.62       0.8985          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np10                            1_755.62       751.47     2_507.09       0.8872          1.0288            1.0214         4.72
IVFPQ-m32-nl158-np1                                    1_831.67        95.82     1_927.48       0.7517          1.1575            1.0756         2.34
IVFPQ-m64-nl158-np1                                    2_731.78       145.07     2_876.86       0.7581          1.1518            1.0676         3.86
SOARPQ-orth1-m32-nl158-np1                             2_031.14       107.67     2_138.82       0.8454          1.0732            1.0355         4.82
IVFPQ-m32-nl158-np2                                    1_831.67       148.77     1_980.43       0.8638          1.0466            1.0258         2.34
IVFPQ-m64-nl158-np2                                    2_731.78       262.06     2_993.85       0.8739          1.0408            1.0199         3.86
SOARPQ-orth1-m32-nl158-np2                             2_031.14       175.61     2_206.76       0.8822          1.0394            1.0227         4.82
IVFPQ-m32-nl158-np4                                    1_831.67       259.33     2_091.00       0.8918          1.0259            1.0187         2.34
IVFPQ-m64-nl158-np4                                    2_731.78       468.90     3_200.68       0.9036          1.0200            1.0138         3.86
SOARPQ-orth1-m32-nl158-np4                             2_031.14       309.30     2_340.44       0.8918          1.0289            1.0191         4.82
IVFPQ-m32-nl158-np7                                    1_831.67       434.01     2_265.67       0.8942          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np7                                    2_731.78       802.42     3_534.21       0.9060          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np7                             2_031.14       507.17     2_538.31       0.8940          1.0254            1.0182         4.82
IVFPQ-m32-nl158-np8                                    1_831.67       496.88     2_328.55       0.8943          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np8                                    2_731.78       916.74     3_648.53       0.9061          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np8                             2_031.14       572.09     2_603.24       0.8941          1.0250            1.0182         4.82
IVFPQ-m32-nl158-np12                                   1_831.67       732.09     2_563.76       0.8944          1.0244            1.0180         2.34
IVFPQ-m64-nl158-np12                                   2_731.78     1_369.45     4_101.23       0.9063          1.0186            1.0132         3.86
SOARPQ-orth1-m32-nl158-np12                            2_031.14       828.04     2_859.18       0.8944          1.0245            1.0180         4.82
IVFPQ-m32-nl223-np1                                    1_973.76        98.24     2_072.00       0.7283          1.1815            1.1041         2.46
IVFPQ-m64-nl223-np1                                    2_870.87       137.47     3_008.34       0.7318          1.1772            1.1004         3.99
SOARPQ-orth1-m32-nl223-np1                             2_219.06       101.20     2_320.26       0.8393          1.0777            1.0371         4.95
IVFPQ-m32-nl223-np2                                    1_973.76       142.68     2_116.44       0.8613          1.0501            1.0263         2.46
IVFPQ-m64-nl223-np2                                    2_870.87       232.00     3_102.87       0.8679          1.0460            1.0217         3.99
SOARPQ-orth1-m32-nl223-np2                             2_219.06       162.55     2_381.61       0.8871          1.0370            1.0205         4.95
IVFPQ-m32-nl223-np4                                    1_973.76       253.15     2_226.90       0.8979          1.0231            1.0163         2.46
IVFPQ-m64-nl223-np4                                    2_870.87       425.93     3_296.81       0.9065          1.0191            1.0133         3.99
SOARPQ-orth1-m32-nl223-np4                             2_219.06       275.63     2_494.69       0.8976          1.0276            1.0170         4.95
IVFPQ-m32-nl223-np8                                    1_973.76       455.98     2_429.73       0.9010          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np8                                    2_870.87       817.00     3_687.88       0.9099          1.0173            1.0126         3.99
SOARPQ-orth1-m32-nl223-np8                             2_219.06       500.84     2_719.90       0.9006          1.0227            1.0160         4.95
IVFPQ-m32-nl223-np11                                   1_973.76       619.62     2_593.38       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np11                                   2_870.87     1_113.21     3_984.09       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np11                            2_219.06       680.05     2_899.11       0.9009          1.0218            1.0158         4.95
IVFPQ-m32-nl223-np14                                   1_973.76       781.86     2_755.62       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np14                                   2_870.87     1_412.68     4_283.55       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np14                            2_219.06       870.06     3_089.12       0.9011          1.0216            1.0157         4.95
IVFPQ-m32-nl316-np1                                    2_204.51        99.51     2_304.03       0.7037          1.2091            1.1328         2.65
IVFPQ-m64-nl316-np1                                    3_121.69       137.32     3_259.01       0.7071          1.2044            1.1272         4.17
SOARPQ-orth1-m32-nl316-np1                             2_510.89       102.73     2_613.62       0.8250          1.0882            1.0452         5.13
IVFPQ-m32-nl316-np2                                    2_204.51       153.48     2_357.99       0.8495          1.0587            1.0313         2.65
IVFPQ-m64-nl316-np2                                    3_121.69       229.58     3_351.27       0.8573          1.0541            1.0258         4.17
SOARPQ-orth1-m32-nl316-np2                             2_510.89       161.62     2_672.52       0.8855          1.0375            1.0212         5.13
IVFPQ-m32-nl316-np4                                    2_204.51       241.35     2_445.86       0.8982          1.0231            1.0160         2.65
IVFPQ-m64-nl316-np4                                    3_121.69       409.27     3_530.96       0.9091          1.0184            1.0119         4.17
SOARPQ-orth1-m32-nl316-np4                             2_510.89       263.56     2_774.45       0.8994          1.0269            1.0166         5.13
IVFPQ-m32-nl316-np8                                    2_204.51       448.39     2_652.91       0.9033          1.0203            1.0148         2.65
IVFPQ-m64-nl316-np8                                    3_121.69       774.41     3_896.10       0.9144          1.0156            1.0109         4.17
SOARPQ-orth1-m32-nl316-np8                             2_510.89       472.10     2_982.99       0.9028          1.0220            1.0152         5.13
IVFPQ-m32-nl316-np15                                   2_204.51       801.69     3_006.20       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np15                                   3_121.69     1_436.54     4_558.23       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np15                            2_510.89       895.34     3_406.24       0.9035          1.0204            1.0148         5.13
IVFPQ-m32-nl316-np17                                   2_204.51       908.51     3_113.03       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np17                                   3_121.69     1_619.98     4_741.68       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np17                            2_510.89       971.87     3_482.76       0.9035          1.0203            1.0148         5.13
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
Exhaustive (query)                                        72.87     1_385.38     1_458.25       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        2_007.39       109.78     2_117.17       0.8501          1.0646            1.0334         4.82
SOARPQ-near-np2                                        2_007.39       174.16     2_181.55       0.8844          1.0345            1.0218         4.82
SOARPQ-near-np4                                        2_007.39       301.80     2_309.19       0.8927          1.0266            1.0188         4.82
SOARPQ-near-np7                                        2_007.39       528.30     2_535.69       0.8942          1.0248            1.0181         4.82
SOARPQ-near-np8                                        2_007.39       561.78     2_569.17       0.8943          1.0246            1.0181         4.82
SOARPQ-near-np12                                       2_007.39       802.30     2_809.69       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.3-np1                                    1_985.23       107.60     2_092.83       0.8492          1.0675            1.0348         4.82
SOARPQ-shift0.3-np2                                    1_985.23       174.13     2_159.36       0.8831          1.0374            1.0226         4.82
SOARPQ-shift0.3-np4                                    1_985.23       301.86     2_287.09       0.8918          1.0282            1.0191         4.82
SOARPQ-shift0.3-np7                                    1_985.23       498.08     2_483.31       0.8940          1.0252            1.0182         4.82
SOARPQ-shift0.3-np8                                    1_985.23       574.24     2_559.47       0.8941          1.0249            1.0182         4.82
SOARPQ-shift0.3-np12                                   1_985.23       804.95     2_790.18       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.7-np1                                    1_959.12       107.54     2_066.66       0.8440          1.0748            1.0368         4.82
SOARPQ-shift0.7-np2                                    1_959.12       173.70     2_132.82       0.8805          1.0417            1.0235         4.82
SOARPQ-shift0.7-np4                                    1_959.12       303.01     2_262.13       0.8906          1.0306            1.0196         4.82
SOARPQ-shift0.7-np7                                    1_959.12       497.24     2_456.36       0.8936          1.0261            1.0184         4.82
SOARPQ-shift0.7-np8                                    1_959.12       563.16     2_522.29       0.8939          1.0256            1.0183         4.82
SOARPQ-shift0.7-np12                                   1_959.12       803.16     2_762.28       0.8943          1.0247            1.0180         4.82
SOARPQ-orth1-np1                                       1_997.66       107.19     2_104.85       0.8454          1.0732            1.0355         4.82
SOARPQ-orth1-np2                                       1_997.66       175.88     2_173.54       0.8822          1.0394            1.0227         4.82
SOARPQ-orth1-np4                                       1_997.66       302.81     2_300.46       0.8918          1.0289            1.0191         4.82
SOARPQ-orth1-np7                                       1_997.66       499.84     2_497.49       0.8940          1.0254            1.0182         4.82
SOARPQ-orth1-np8                                       1_997.66       566.46     2_564.12       0.8941          1.0250            1.0182         4.82
SOARPQ-orth1-np12                                      1_997.66       803.91     2_801.57       0.8944          1.0245            1.0180         4.82
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
Exhaustive (query)                                        69.25     1_259.19     1_328.43       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_424.09       460.11     7_884.20       0.3585          1.0695            1.0694         3.49
IVFOPQ-m64-nl111-np1                                  11_681.35       559.80    12_241.14       0.4592          1.0462            1.0426         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_731.62       464.09     9_195.72       0.3380          1.2121            1.0714         5.98
IVFOPQ-m32-nl111-np2                                   7_424.09       510.10     7_934.19       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np2                                  11_681.35       648.49    12_329.84       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_731.62       519.21     9_250.84       0.3596          1.0702            1.0692         5.98
IVFOPQ-m32-nl111-np4                                   7_424.09       603.36     8_027.45       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np4                                  11_681.35       830.26    12_511.61       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_731.62       625.07     9_356.69       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np5                                   7_424.09       650.43     8_074.53       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np5                                  11_681.35       920.86    12_602.20       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_731.62       667.03     9_398.65       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np8                                   7_424.09       794.96     8_219.06       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np8                                  11_681.35     1_210.78    12_892.13       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_731.62       840.08     9_571.70       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np10                                  7_424.09       897.63     8_321.72       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np10                                 11_681.35     1_407.54    13_088.89       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_731.62       949.40     9_681.02       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl158-np1                                   8_021.65       461.85     8_483.50       0.3639          1.0672            1.0680         3.84
IVFOPQ-m64-nl158-np1                                  12_347.33       555.37    12_902.70       0.4702          1.0432            1.0408         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         9_137.47       464.23     9_601.70       0.3358          1.1637            1.0708         6.32
IVFOPQ-m32-nl158-np2                                   8_021.65       518.72     8_540.37       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np2                                  12_347.33       650.34    12_997.67       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         9_137.47       517.93     9_655.40       0.3674          1.0667            1.0675         6.32
IVFOPQ-m32-nl158-np4                                   8_021.65       608.38     8_630.03       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np4                                  12_347.33       826.66    13_173.99       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         9_137.47       613.36     9_750.83       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np7                                   8_021.65       741.90     8_763.55       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np7                                  12_347.33     1_081.57    13_428.90       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         9_137.47       752.67     9_890.15       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np8                                   8_021.65       788.28     8_809.93       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np8                                  12_347.33     1_173.58    13_520.91       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         9_137.47       812.03     9_949.50       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np12                                  8_021.65       974.93     8_996.58       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np12                                 12_347.33     1_537.46    13_884.79       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        9_137.47       997.91    10_135.39       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl223-np1                                   8_192.59       438.35     8_630.94       0.3606          1.0674            1.0674         3.96
IVFOPQ-m64-nl223-np1                                  12_892.90       496.34    13_389.24       0.4461          1.0466            1.0437         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_467.44       450.82     9_918.26       0.3478          1.1117            1.0696         6.45
IVFOPQ-m32-nl223-np2                                   8_192.59       491.21     8_683.79       0.3754          1.0628            1.0639         3.96
IVFOPQ-m64-nl223-np2                                  12_892.90       597.90    13_490.80       0.4761          1.0402            1.0390         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_467.44       515.30     9_982.74       0.3741          1.0642            1.0651         6.45
IVFOPQ-m32-nl223-np4                                   8_192.59       590.11     8_782.69       0.3790          1.0620            1.0631         3.96
IVFOPQ-m64-nl223-np4                                  12_892.90       825.06    13_717.96       0.4855          1.0389            1.0374         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_467.44       610.02    10_077.46       0.3778          1.0628            1.0635         6.45
IVFOPQ-m32-nl223-np8                                   8_192.59       781.91     8_974.50       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np8                                  12_892.90     1_152.56    14_045.46       0.4869          1.0387            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_467.44       802.90    10_270.34       0.3793          1.0619            1.0630         6.45
IVFOPQ-m32-nl223-np11                                  8_192.59       932.51     9_125.10       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np11                                 12_892.90     1_427.54    14_320.44       0.4869          1.0387            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_467.44       953.53    10_420.96       0.3794          1.0619            1.0629         6.45
IVFOPQ-m32-nl223-np14                                  8_192.59     1_088.78     9_281.37       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np14                                 12_892.90     1_704.45    14_597.35       0.4869          1.0386            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_467.44     1_116.33    10_583.77       0.3794          1.0619            1.0629         6.45
IVFOPQ-m32-nl316-np1                                   8_232.87       436.22     8_669.09       0.3591          1.0668            1.0666         4.65
IVFOPQ-m64-nl316-np1                                  12_799.95       482.28    13_282.23       0.4345          1.0485            1.0448         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_770.31       446.10    10_216.40       0.3407          1.1111            1.0694         7.13
IVFOPQ-m32-nl316-np2                                   8_232.87       484.75     8_717.61       0.3800          1.0598            1.0615         4.65
IVFOPQ-m64-nl316-np2                                  12_799.95       582.28    13_382.23       0.4771          1.0392            1.0381         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_770.31       500.65    10_270.96       0.3779          1.0616            1.0632         7.13
IVFOPQ-m32-nl316-np4                                   8_232.87       586.16     8_819.02       0.3868          1.0581            1.0600         4.65
IVFOPQ-m64-nl316-np4                                  12_799.95       772.07    13_572.02       0.4945          1.0366            1.0357         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_770.31       605.02    10_375.33       0.3836          1.0596            1.0613         7.13
IVFOPQ-m32-nl316-np8                                   8_232.87       778.50     9_011.36       0.3878          1.0579            1.0598         4.65
IVFOPQ-m64-nl316-np8                                  12_799.95     1_130.55    13_930.50       0.4982          1.0361            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_770.31       811.44    10_581.75       0.3870          1.0583            1.0600         7.13
IVFOPQ-m32-nl316-np15                                  8_232.87     1_136.58     9_369.44       0.3878          1.0579            1.0597         4.65
IVFOPQ-m64-nl316-np15                                 12_799.95     1_778.13    14_578.08       0.4984          1.0360            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_770.31     1_155.49    10_925.79       0.3878          1.0579            1.0597         7.13
IVFOPQ-m32-nl316-np17                                  8_232.87     1_225.49     9_458.35       0.3878          1.0579            1.0597         4.65
IVFOPQ-m64-nl316-np17                                 12_799.95     1_958.65    14_758.60       0.4984          1.0360            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_770.31     1_247.60    11_017.90       0.3878          1.0579            1.0597         7.13
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
Exhaustive (query)                                        69.25     1_259.19     1_328.43       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       9_169.19       499.14     9_668.33       0.3367          1.1614            1.0707         6.32
SOAROPQ-near-np2                                       9_169.19       517.90     9_687.09       0.3676          1.0662            1.0675         6.32
SOAROPQ-near-np4                                       9_169.19       613.99     9_783.19       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np7                                       9_169.19       751.26     9_920.45       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np8                                       9_169.19       799.71     9_968.90       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np12                                      9_169.19       991.36    10_160.55       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np1                                   9_173.25       469.53     9_642.78       0.3360          1.1627            1.0708         6.32
SOAROPQ-shift0.3-np2                                   9_173.25       517.44     9_690.69       0.3674          1.0665            1.0675         6.32
SOAROPQ-shift0.3-np4                                   9_173.25       612.52     9_785.77       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np7                                   9_173.25       753.13     9_926.38       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np8                                   9_173.25       803.21     9_976.45       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np12                                  9_173.25     1_001.91    10_175.16       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np1                                   9_265.57       473.01     9_738.58       0.3358          1.1639            1.0708         6.32
SOAROPQ-shift0.7-np2                                   9_265.57       536.26     9_801.83       0.3673          1.0671            1.0675         6.32
SOAROPQ-shift0.7-np4                                   9_265.57       625.64     9_891.21       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np7                                   9_265.57       756.51    10_022.08       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np8                                   9_265.57       803.64    10_069.21       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np12                                  9_265.57       991.75    10_257.32       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np1                                      9_170.95       468.11     9_639.07       0.3371          1.1616            1.0706         6.32
SOAROPQ-orth1-np2                                      9_170.95       515.84     9_686.80       0.3678          1.0658            1.0674         6.32
SOAROPQ-orth1-np4                                      9_170.95       611.35     9_782.31       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np7                                      9_170.95       751.72     9_922.67       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np8                                      9_170.95       801.91     9_972.87       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np12                                     9_170.95       995.40    10_166.36       0.3680          1.0655            1.0674         6.32
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
Exhaustive (query)                                        68.56     1_339.71     1_408.27       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_477.46       463.01     7_940.47       0.6648          1.0298            1.0250         3.49
IVFOPQ-m64-nl111-np1                                  11_791.64       548.82    12_340.46       0.7593          1.0165            1.0112         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_665.55       462.89     9_128.44       0.6656          1.0330            1.0252         5.98
IVFOPQ-m32-nl111-np2                                   7_477.46       504.93     7_982.38       0.6764          1.0260            1.0244         3.49
IVFOPQ-m64-nl111-np2                                  11_791.64       647.63    12_439.27       0.7734          1.0125            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_665.55       514.03     9_179.58       0.6764          1.0261            1.0245         5.98
IVFOPQ-m32-nl111-np4                                   7_477.46       602.89     8_080.34       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np4                                  11_791.64       821.69    12_613.32       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_665.55       624.22     9_289.77       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np5                                   7_477.46       655.07     8_132.52       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np5                                  11_791.64       916.24    12_707.88       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_665.55       668.32     9_333.87       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np8                                   7_477.46       787.92     8_265.37       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np8                                  11_791.64     1_176.23    12_967.87       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_665.55       829.48     9_495.02       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np10                                  7_477.46       880.55     8_358.00       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np10                                 11_791.64     1_359.50    13_151.14       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_665.55       940.30     9_605.85       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl158-np1                                   7_847.36       461.43     8_308.79       0.6634          1.0312            1.0245         3.84
IVFOPQ-m64-nl158-np1                                  12_244.31       554.76    12_799.07       0.7543          1.0184            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         9_139.57       461.68     9_601.25       0.6770          1.0274            1.0242         6.32
IVFOPQ-m32-nl158-np2                                   7_847.36       504.48     8_351.85       0.6818          1.0253            1.0235         3.84
IVFOPQ-m64-nl158-np2                                  12_244.31       636.72    12_881.03       0.7762          1.0122            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         9_139.57       511.72     9_651.30       0.6834          1.0249            1.0235         6.32
IVFOPQ-m32-nl158-np4                                   7_847.36       596.90     8_444.26       0.6838          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np4                                  12_244.31       810.50    13_054.81       0.7788          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         9_139.57       606.50     9_746.07       0.6839          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np7                                   7_847.36       736.71     8_584.07       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np7                                  12_244.31     1_069.62    13_313.93       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         9_139.57       747.55     9_887.12       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np8                                   7_847.36       778.61     8_625.97       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np8                                  12_244.31     1_157.81    13_402.12       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         9_139.57       796.92     9_936.49       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np12                                  7_847.36       974.82     8_822.19       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np12                                 12_244.31     1_521.83    13_766.14       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        9_139.57       995.60    10_135.18       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl223-np1                                   8_336.08       440.95     8_777.03       0.4945          1.0652            1.0581         3.96
IVFOPQ-m64-nl223-np1                                  12_740.37       495.73    13_236.10       0.5346          1.0541            1.0467         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_372.29       473.27     9_845.57       0.5990          1.0416            1.0356         6.45
IVFOPQ-m32-nl223-np2                                   8_336.08       491.55     8_827.63       0.6125          1.0373            1.0324         3.96
IVFOPQ-m64-nl223-np2                                  12_740.37       600.61    13_340.99       0.6813          1.0254            1.0187         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_372.29       508.77     9_881.07       0.6610          1.0285            1.0263         6.45
IVFOPQ-m32-nl223-np4                                   8_336.08       593.60     8_929.68       0.6718          1.0266            1.0247         3.96
IVFOPQ-m64-nl223-np4                                  12_740.37       797.74    13_538.11       0.7590          1.0143            1.0119         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_372.29       608.26     9_980.56       0.6852          1.0243            1.0230         6.45
IVFOPQ-m32-nl223-np8                                   8_336.08       781.34     9_117.42       0.6888          1.0237            1.0225         3.96
IVFOPQ-m64-nl223-np8                                  12_740.37     1_153.34    13_893.71       0.7817          1.0112            1.0101         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_372.29       805.18    10_177.48       0.6898          1.0236            1.0224         6.45
IVFOPQ-m32-nl223-np11                                  8_336.08       940.13     9_276.21       0.6898          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np11                                 12_740.37     1_431.24    14_171.61       0.7833          1.0110            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_372.29       967.94    10_340.24       0.6898          1.0235            1.0223         6.45
IVFOPQ-m32-nl223-np14                                  8_336.08     1_073.22     9_409.30       0.6898          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np14                                 12_740.37     1_690.21    14_430.58       0.7833          1.0110            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_372.29     1_101.78    10_474.07       0.6898          1.0235            1.0223         6.45
IVFOPQ-m32-nl316-np1                                   8_652.12       437.34     9_089.46       0.4099          1.0855            1.0797         4.65
IVFOPQ-m64-nl316-np1                                  13_930.74       481.50    14_412.24       0.4293          1.0753            1.0691         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_526.32       445.11     9_971.43       0.5369          1.0531            1.0492         7.13
IVFOPQ-m32-nl316-np2                                   8_652.12       484.69     9_136.81       0.5493          1.0491            1.0454         4.65
IVFOPQ-m64-nl316-np2                                  13_930.74       581.80    14_512.54       0.5978          1.0379            1.0336         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_526.32       500.98    10_027.30       0.6301          1.0342            1.0315         7.13
IVFOPQ-m32-nl316-np4                                   8_652.12       593.86     9_245.98       0.6436          1.0315            1.0288         4.65
IVFOPQ-m64-nl316-np4                                  13_930.74       773.67    14_704.41       0.7203          1.0196            1.0159         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_526.32       606.02    10_132.34       0.6787          1.0256            1.0241         7.13
IVFOPQ-m32-nl316-np8                                   8_652.12       800.72     9_452.84       0.6876          1.0240            1.0227         4.65
IVFOPQ-m64-nl316-np8                                  13_930.74     1_132.09    15_062.83       0.7776          1.0118            1.0106         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_526.32       795.32    10_321.64       0.6923          1.0233            1.0221         7.13
IVFOPQ-m32-nl316-np15                                  8_652.12     1_131.06     9_783.18       0.6930          1.0231            1.0219         4.65
IVFOPQ-m64-nl316-np15                                 13_930.74     1_757.90    15_688.64       0.7851          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_526.32     1_170.94    10_697.26       0.6930          1.0231            1.0219         7.13
IVFOPQ-m32-nl316-np17                                  8_652.12     1_226.88     9_879.00       0.6930          1.0231            1.0219         4.65
IVFOPQ-m64-nl316-np17                                 13_930.74     1_938.12    15_868.86       0.7851          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_526.32     1_245.92    10_772.24       0.6930          1.0231            1.0219         7.13
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
Exhaustive (query)                                        68.56     1_339.71     1_408.27       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       9_366.60       466.31     9_832.91       0.6770          1.0274            1.0242         6.32
SOAROPQ-near-np2                                       9_366.60       515.97     9_882.57       0.6834          1.0250            1.0235         6.32
SOAROPQ-near-np4                                       9_366.60       616.51     9_983.10       0.6839          1.0247            1.0234         6.32
SOAROPQ-near-np7                                       9_366.60       746.36    10_112.95       0.6840          1.0247            1.0234         6.32
SOAROPQ-near-np8                                       9_366.60       792.75    10_159.35       0.6840          1.0247            1.0234         6.32
SOAROPQ-near-np12                                      9_366.60     1_016.66    10_383.26       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np1                                   9_544.22       474.26    10_018.48       0.6770          1.0274            1.0241         6.32
SOAROPQ-shift0.3-np2                                   9_544.22       540.96    10_085.18       0.6834          1.0250            1.0235         6.32
SOAROPQ-shift0.3-np4                                   9_544.22       648.44    10_192.66       0.6839          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np7                                   9_544.22       753.73    10_297.95       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np8                                   9_544.22       799.55    10_343.78       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np12                                  9_544.22       993.34    10_537.56       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np1                                   9_144.76       467.87     9_612.63       0.6769          1.0275            1.0242         6.32
SOAROPQ-shift0.7-np2                                   9_144.76       512.88     9_657.64       0.6833          1.0250            1.0235         6.32
SOAROPQ-shift0.7-np4                                   9_144.76       608.86     9_753.62       0.6839          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np7                                   9_144.76       748.50     9_893.25       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np8                                   9_144.76       794.98     9_939.74       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np12                                  9_144.76       994.80    10_139.56       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np1                                      9_240.54       464.94     9_705.47       0.6766          1.0276            1.0242         6.32
SOAROPQ-orth1-np2                                      9_240.54       512.38     9_752.92       0.6833          1.0250            1.0235         6.32
SOAROPQ-orth1-np4                                      9_240.54       607.91     9_848.45       0.6838          1.0247            1.0234         6.32
SOAROPQ-orth1-np7                                      9_240.54       768.12    10_008.66       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np8                                      9_240.54       793.95    10_034.48       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np12                                     9_240.54     1_002.13    10_242.66       0.6840          1.0247            1.0234         6.32
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
Exhaustive (query)                                        71.14     1_352.10     1_423.25       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_674.44       436.43     8_110.87       0.7214          1.1714            1.0835         3.49
IVFOPQ-m64-nl111-np1                                  12_065.06       500.59    12_565.65       0.7259          1.1682            1.0806         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_017.94       456.01     9_473.95       0.8433          1.0560            1.0277         5.98
IVFOPQ-m32-nl111-np2                                   7_674.44       502.95     8_177.39       0.8509          1.0487            1.0238         3.49
IVFOPQ-m64-nl111-np2                                  12_065.06       626.24    12_691.30       0.8604          1.0452            1.0197         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_017.94       542.92     9_560.86       0.8849          1.0261            1.0174         5.98
IVFOPQ-m32-nl111-np4                                   7_674.44       633.68     8_308.12       0.8896          1.0220            1.0160         3.49
IVFOPQ-m64-nl111-np4                                  12_065.06       874.45    12_939.51       0.9012          1.0184            1.0124         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_017.94       706.89     9_724.83       0.8919          1.0213            1.0156         5.98
IVFOPQ-m32-nl111-np5                                   7_674.44       687.45     8_361.89       0.8916          1.0209            1.0156         3.49
IVFOPQ-m64-nl111-np5                                  12_065.06     1_004.23    13_069.29       0.9033          1.0173            1.0120         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_017.94       782.30     9_800.24       0.8923          1.0209            1.0155         5.98
IVFOPQ-m32-nl111-np8                                   7_674.44       876.92     8_551.36       0.8929          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np8                                  12_065.06     1_383.10    13_448.16       0.9047          1.0166            1.0117         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_017.94       993.13    10_011.07       0.8928          1.0205            1.0154         5.98
IVFOPQ-m32-nl111-np10                                  7_674.44       998.60     8_673.04       0.8929          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np10                                 12_065.06     1_635.21    13_700.27       0.9048          1.0166            1.0117         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_017.94     1_138.61    10_156.55       0.8929          1.0203            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   8_582.54       439.14     9_021.68       0.7099          1.1800            1.0954         3.84
IVFOPQ-m64-nl158-np1                                  13_967.62       486.96    14_454.58       0.7128          1.1777            1.0937         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         9_990.74       450.19    10_440.93       0.8400          1.0596            1.0281         6.32
IVFOPQ-m32-nl158-np2                                   8_582.54       497.21     9_079.75       0.8497          1.0504            1.0232         3.84
IVFOPQ-m64-nl158-np2                                  13_967.62       618.59    14_586.21       0.8566          1.0478            1.0202         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         9_990.74       524.46    10_515.20       0.8901          1.0245            1.0155         6.32
IVFOPQ-m32-nl158-np4                                   8_582.54       602.95     9_185.49       0.8953          1.0201            1.0139         3.84
IVFOPQ-m64-nl158-np4                                  13_967.62       825.85    14_793.47       0.9046          1.0174            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         9_990.74       658.43    10_649.17       0.8992          1.0187            1.0132         6.32
IVFOPQ-m32-nl158-np7                                   8_582.54       779.82     9_362.36       0.9007          1.0173            1.0129         3.84
IVFOPQ-m64-nl158-np7                                  13_967.62     1_161.25    15_128.87       0.9101          1.0145            1.0102         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         9_990.74       883.49    10_874.23       0.9007          1.0175            1.0130         6.32
IVFOPQ-m32-nl158-np8                                   8_582.54       840.30     9_422.84       0.9009          1.0171            1.0129         3.84
IVFOPQ-m64-nl158-np8                                  13_967.62     1_274.17    15_241.79       0.9104          1.0144            1.0101         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         9_990.74       941.86    10_932.60       0.9008          1.0174            1.0129         6.32
IVFOPQ-m32-nl158-np12                                  8_582.54     1_058.97     9_641.51       0.9011          1.0171            1.0129         3.84
IVFOPQ-m64-nl158-np12                                 13_967.62     1_713.94    15_681.56       0.9106          1.0143            1.0100         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        9_990.74     1_180.88    11_171.62       0.9011          1.0171            1.0129         6.32
IVFOPQ-m32-nl223-np1                                   9_118.54       439.88     9_558.42       0.6953          1.1884            1.1141         3.96
IVFOPQ-m64-nl223-np1                                  14_034.59       478.43    14_513.03       0.6973          1.1861            1.1118         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_520.32       436.99    10_957.31       0.8322          1.0659            1.0325         6.45
IVFOPQ-m32-nl223-np2                                   9_118.54       480.42     9_598.96       0.8446          1.0542            1.0256         3.96
IVFOPQ-m64-nl223-np2                                  14_034.59       579.77    14_614.36       0.8501          1.0515            1.0221         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_520.32       509.53    11_029.85       0.8922          1.0252            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   9_118.54       585.87     9_704.40       0.8997          1.0193            1.0124         3.96
IVFOPQ-m64-nl223-np4                                  14_034.59       780.74    14_815.34       0.9079          1.0164            1.0098         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_520.32       624.06    11_144.37       0.9052          1.0176            1.0115         6.45
IVFOPQ-m32-nl223-np8                                   9_118.54       800.55     9_919.09       0.9076          1.0154            1.0108         3.96
IVFOPQ-m64-nl223-np8                                  14_034.59     1_179.05    15_213.65       0.9164          1.0124            1.0084         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_520.32       889.69    11_410.00       0.9075          1.0158            1.0109         6.45
IVFOPQ-m32-nl223-np11                                  9_118.54       958.85    10_077.39       0.9080          1.0153            1.0107         3.96
IVFOPQ-m64-nl223-np11                                 14_034.59     1_488.12    15_522.72       0.9168          1.0122            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_520.32     1_032.56    11_552.87       0.9079          1.0155            1.0108         6.45
IVFOPQ-m32-nl223-np14                                  9_118.54     1_132.63    10_251.17       0.9081          1.0153            1.0107         3.96
IVFOPQ-m64-nl223-np14                                 14_034.59     1_799.84    15_834.43       0.9169          1.0122            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_520.32     1_220.18    11_740.50       0.9080          1.0153            1.0108         6.45
IVFOPQ-m32-nl316-np1                                   8_570.12       437.89     9_008.01       0.6789          1.2032            1.1314         4.65
IVFOPQ-m64-nl316-np1                                  13_012.01       483.26    13_495.27       0.6805          1.2011            1.1296         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_878.81       438.93    10_317.74       0.8245          1.0715            1.0363         7.13
IVFOPQ-m32-nl316-np2                                   8_570.12       501.41     9_071.53       0.8379          1.0584            1.0277         4.65
IVFOPQ-m64-nl316-np2                                  13_012.01       573.39    13_585.40       0.8415          1.0568            1.0256         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_878.81       501.89    10_380.70       0.8936          1.0251            1.0141         7.13
IVFOPQ-m32-nl316-np4                                   8_570.12       579.20     9_149.32       0.9033          1.0182            1.0111         4.65
IVFOPQ-m64-nl316-np4                                  13_012.01       756.67    13_768.68       0.9089          1.0163            1.0093         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_878.81       603.15    10_481.95       0.9102          1.0161            1.0102         7.13
IVFOPQ-m32-nl316-np8                                   8_570.12       782.32     9_352.44       0.9138          1.0131            1.0092         4.65
IVFOPQ-m64-nl316-np8                                  13_012.01     1_135.42    14_147.43       0.9202          1.0111            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_878.81       823.93    10_702.73       0.9137          1.0137            1.0093         7.13
IVFOPQ-m32-nl316-np15                                  8_570.12     1_147.26     9_717.38       0.9146          1.0127            1.0090         4.65
IVFOPQ-m64-nl316-np15                                 13_012.01     1_806.52    14_818.52       0.9210          1.0108            1.0073         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_878.81     1_218.27    11_097.08       0.9145          1.0129            1.0091         7.13
IVFOPQ-m32-nl316-np17                                  8_570.12     1_266.89     9_837.01       0.9146          1.0127            1.0090         4.65
IVFOPQ-m64-nl316-np17                                 13_012.01     2_011.52    15_023.53       0.9210          1.0108            1.0073         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_878.81     1_353.88    11_232.69       0.9145          1.0128            1.0091         7.13
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
Exhaustive (query)                                        71.14     1_352.10     1_423.25       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       9_750.50       452.95    10_203.45       0.8390          1.0615            1.0261         6.32
SOAROPQ-near-np2                                       9_750.50       521.36    10_271.86       0.8901          1.0242            1.0152         6.32
SOAROPQ-near-np4                                       9_750.50       662.26    10_412.76       0.8995          1.0183            1.0132         6.32
SOAROPQ-near-np7                                       9_750.50       857.92    10_608.42       0.9008          1.0174            1.0129         6.32
SOAROPQ-near-np8                                       9_750.50       922.33    10_672.83       0.9009          1.0173            1.0129         6.32
SOAROPQ-near-np12                                      9_750.50     1_179.96    10_930.46       0.9011          1.0171            1.0129         6.32
SOAROPQ-shift0.3-np1                                   9_765.00       451.30    10_216.30       0.8423          1.0578            1.0271         6.32
SOAROPQ-shift0.3-np2                                   9_765.00       520.06    10_285.06       0.8908          1.0239            1.0153         6.32
SOAROPQ-shift0.3-np4                                   9_765.00       658.12    10_423.12       0.8995          1.0184            1.0132         6.32
SOAROPQ-shift0.3-np7                                   9_765.00       861.23    10_626.23       0.9008          1.0174            1.0129         6.32
SOAROPQ-shift0.3-np8                                   9_765.00       925.17    10_690.17       0.9009          1.0173            1.0129         6.32
SOAROPQ-shift0.3-np12                                  9_765.00     1_168.43    10_933.43       0.9011          1.0171            1.0129         6.32
SOAROPQ-shift0.7-np1                                   9_794.26       450.88    10_245.14       0.8365          1.0625            1.0295         6.32
SOAROPQ-shift0.7-np2                                   9_794.26       518.59    10_312.85       0.8890          1.0254            1.0158         6.32
SOAROPQ-shift0.7-np4                                   9_794.26       674.57    10_468.83       0.8989          1.0190            1.0134         6.32
SOAROPQ-shift0.7-np7                                   9_794.26       855.63    10_649.90       0.9006          1.0176            1.0130         6.32
SOAROPQ-shift0.7-np8                                   9_794.26       921.32    10_715.58       0.9008          1.0175            1.0130         6.32
SOAROPQ-shift0.7-np12                                  9_794.26     1_186.66    10_980.92       0.9011          1.0171            1.0129         6.32
SOAROPQ-orth1-np1                                      9_821.74       470.49    10_292.22       0.8391          1.0608            1.0274         6.32
SOAROPQ-orth1-np2                                      9_821.74       519.56    10_341.30       0.8901          1.0245            1.0154         6.32
SOAROPQ-orth1-np4                                      9_821.74       658.00    10_479.73       0.8994          1.0186            1.0132         6.32
SOAROPQ-orth1-np7                                      9_821.74       858.71    10_680.45       0.9008          1.0175            1.0129         6.32
SOAROPQ-orth1-np8                                      9_821.74       956.28    10_778.02       0.9009          1.0173            1.0129         6.32
SOAROPQ-orth1-np12                                     9_821.74     1_180.62    11_002.35       0.9011          1.0171            1.0129         6.32
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
Exhaustive (query)                                        73.17     1_356.76     1_429.93       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   7_763.90       436.31     8_200.21       0.7751          1.1356            1.0552         3.49
IVFOPQ-m64-nl111-np1                                  12_094.39       482.01    12_576.40       0.7781          1.1337            1.0517         5.02
SOAROPQ-orth1-m32-nl111-np1                            9_096.57       446.17     9_542.74       0.8634          1.0554            1.0260         5.98
IVFOPQ-m32-nl111-np2                                   7_763.90       483.51     8_247.41       0.8795          1.0366            1.0195         3.49
IVFOPQ-m64-nl111-np2                                  12_094.39       620.90    12_715.29       0.8843          1.0344            1.0173         5.02
SOAROPQ-orth1-m32-nl111-np2                            9_096.57       518.55     9_615.11       0.8948          1.0283            1.0173         5.98
IVFOPQ-m32-nl111-np4                                   7_763.90       601.05     8_364.94       0.9011          1.0217            1.0152         3.49
IVFOPQ-m64-nl111-np4                                  12_094.39       827.03    12_921.42       0.9065          1.0196            1.0130         5.02
SOAROPQ-orth1-m32-nl111-np4                            9_096.57       661.98     9_758.54       0.9014          1.0226            1.0153         5.98
IVFOPQ-m32-nl111-np5                                   7_763.90       660.83     8_424.73       0.9022          1.0212            1.0150         3.49
IVFOPQ-m64-nl111-np5                                  12_094.39       962.52    13_056.91       0.9075          1.0191            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np5                            9_096.57       733.86     9_830.43       0.9021          1.0220            1.0151         5.98
IVFOPQ-m32-nl111-np8                                   7_763.90       848.22     8_612.12       0.9028          1.0209            1.0148         3.49
IVFOPQ-m64-nl111-np8                                  12_094.39     1_324.48    13_418.87       0.9081          1.0188            1.0126         5.02
SOAROPQ-orth1-m32-nl111-np8                            9_096.57       943.09    10_039.66       0.9027          1.0212            1.0149         5.98
IVFOPQ-m32-nl111-np10                                  7_763.90       957.95     8_721.85       0.9029          1.0209            1.0148         3.49
IVFOPQ-m64-nl111-np10                                 12_094.39     1_548.91    13_643.30       0.9082          1.0188            1.0126         5.02
SOAROPQ-orth1-m32-nl111-np10                           9_096.57     1_085.78    10_182.34       0.9029          1.0211            1.0148         5.98
IVFOPQ-m32-nl158-np1                                   8_192.19       428.09     8_620.29       0.7588          1.1517            1.0674         3.84
IVFOPQ-m64-nl158-np1                                  12_638.63       479.52    13_118.15       0.7618          1.1491            1.0634         5.36
SOAROPQ-orth1-m32-nl158-np1                            9_441.30       439.54     9_880.84       0.8580          1.0609            1.0275         6.32
IVFOPQ-m32-nl158-np2                                   8_192.19       480.13     8_672.32       0.8754          1.0406            1.0197         3.84
IVFOPQ-m64-nl158-np2                                  12_638.63       587.20    13_225.83       0.8808          1.0379            1.0169         5.36
SOAROPQ-orth1-m32-nl158-np2                            9_441.30       509.53     9_950.83       0.8969          1.0287            1.0165         6.32
IVFOPQ-m32-nl158-np4                                   8_192.19       596.54     8_788.73       0.9056          1.0194            1.0134         3.84
IVFOPQ-m64-nl158-np4                                  12_638.63       803.42    13_442.04       0.9119          1.0168            1.0110         5.36
SOAROPQ-orth1-m32-nl158-np4                            9_441.30       649.71    10_091.01       0.9062          1.0205            1.0135         6.32
IVFOPQ-m32-nl158-np7                                   8_192.19       763.73     8_955.92       0.9081          1.0180            1.0128         3.84
IVFOPQ-m64-nl158-np7                                  12_638.63     1_137.67    13_776.30       0.9146          1.0155            1.0105         5.36
SOAROPQ-orth1-m32-nl158-np7                            9_441.30       840.28    10_281.58       0.9079          1.0185            1.0129         6.32
IVFOPQ-m32-nl158-np8                                   8_192.19       826.29     9_018.48       0.9082          1.0179            1.0128         3.84
IVFOPQ-m64-nl158-np8                                  12_638.63     1_233.37    13_872.00       0.9147          1.0154            1.0104         5.36
SOAROPQ-orth1-m32-nl158-np8                            9_441.30       900.10    10_341.40       0.9080          1.0183            1.0128         6.32
IVFOPQ-m32-nl158-np12                                  8_192.19     1_044.23     9_236.42       0.9084          1.0179            1.0127         3.84
IVFOPQ-m64-nl158-np12                                 12_638.63     1_676.50    14_315.13       0.9148          1.0154            1.0104         5.36
SOAROPQ-orth1-m32-nl158-np12                           9_441.30     1_134.91    10_576.21       0.9083          1.0180            1.0127         6.32
IVFOPQ-m32-nl223-np1                                   8_607.34       422.59     9_029.93       0.7330          1.1767            1.0999         3.96
IVFOPQ-m64-nl223-np1                                  12_843.62       471.06    13_314.68       0.7347          1.1749            1.0978         5.49
SOAROPQ-orth1-m32-nl223-np1                           10_060.96       434.95    10_495.91       0.8494          1.0687            1.0294         6.45
IVFOPQ-m32-nl223-np2                                   8_607.34       475.90     9_083.24       0.8709          1.0451            1.0207         3.96
IVFOPQ-m64-nl223-np2                                  12_843.62       568.95    13_412.57       0.8741          1.0434            1.0185         5.49
SOAROPQ-orth1-m32-nl223-np2                           10_060.96       490.21    10_551.16       0.9000          1.0280            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   8_607.34       578.99     9_186.34       0.9099          1.0179            1.0120         3.96
IVFOPQ-m64-nl223-np4                                  12_843.62       765.24    13_608.86       0.9149          1.0162            1.0103         5.49
SOAROPQ-orth1-m32-nl223-np4                           10_060.96       607.18    10_668.13       0.9108          1.0198            1.0121         6.45
IVFOPQ-m32-nl223-np8                                   8_607.34       795.13     9_402.47       0.9134          1.0162            1.0113         3.96
IVFOPQ-m64-nl223-np8                                  12_843.62     1_142.18    13_985.79       0.9185          1.0144            1.0095         5.49
SOAROPQ-orth1-m32-nl223-np8                           10_060.96       825.24    10_886.20       0.9132          1.0168            1.0114         6.45
IVFOPQ-m32-nl223-np11                                  8_607.34       935.51     9_542.86       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np11                                 12_843.62     1_436.79    14_280.41       0.9187          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np11                          10_060.96       999.94    11_060.90       0.9135          1.0163            1.0113         6.45
IVFOPQ-m32-nl223-np14                                  8_607.34     1_099.83     9_707.17       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np14                                 12_843.62     1_745.00    14_588.61       0.9188          1.0143            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np14                          10_060.96     1_193.50    11_254.46       0.9135          1.0162            1.0112         6.45
IVFOPQ-m32-nl316-np1                                   9_096.61       428.91     9_525.52       0.7080          1.2039            1.1272         4.65
IVFOPQ-m64-nl316-np1                                  13_253.28       472.80    13_726.08       0.7097          1.2018            1.1246         6.17
SOAROPQ-orth1-m32-nl316-np1                           10_507.77       434.31    10_942.08       0.8347          1.0799            1.0371         7.13
IVFOPQ-m32-nl316-np2                                   9_096.61       477.21     9_573.82       0.8590          1.0536            1.0252         4.65
IVFOPQ-m64-nl316-np2                                  13_253.28       565.36    13_818.65       0.8632          1.0515            1.0223         6.17
SOAROPQ-orth1-m32-nl316-np2                           10_507.77       488.14    10_995.91       0.8984          1.0293            1.0154         7.13
IVFOPQ-m32-nl316-np4                                   9_096.61       573.87     9_670.48       0.9114          1.0177            1.0112         4.65
IVFOPQ-m64-nl316-np4                                  13_253.28       749.31    14_002.59       0.9173          1.0157            1.0093         6.17
SOAROPQ-orth1-m32-nl316-np4                           10_507.77       594.16    11_101.93       0.9135          1.0191            1.0112         7.13
IVFOPQ-m32-nl316-np8                                   9_096.61       783.80     9_880.41       0.9170          1.0148            1.0101         4.65
IVFOPQ-m64-nl316-np8                                  13_253.28     1_117.12    14_370.40       0.9234          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np8                           10_507.77       801.79    11_309.56       0.9167          1.0159            1.0103         7.13
IVFOPQ-m32-nl316-np15                                  9_096.61     1_120.27    10_216.89       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np15                                 13_253.28     1_782.87    15_036.15       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np15                          10_507.77     1_183.24    11_691.01       0.9173          1.0148            1.0100         7.13
IVFOPQ-m32-nl316-np17                                  9_096.61     1_240.06    10_336.67       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np17                                 13_253.28     1_984.84    15_238.12       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np17                          10_507.77     1_304.79    11_812.56       0.9174          1.0147            1.0100         7.13
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
Exhaustive (query)                                        73.17     1_356.76     1_429.93       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                       9_413.22       454.39     9_867.61       0.8624          1.0547            1.0259         6.32
SOAROPQ-near-np2                                       9_413.22       508.08     9_921.30       0.8988          1.0257            1.0158         6.32
SOAROPQ-near-np4                                       9_413.22       639.83    10_053.04       0.9068          1.0193            1.0133         6.32
SOAROPQ-near-np7                                       9_413.22       834.73    10_247.95       0.9081          1.0181            1.0128         6.32
SOAROPQ-near-np8                                       9_413.22       905.84    10_319.06       0.9082          1.0180            1.0128         6.32
SOAROPQ-near-np12                                      9_413.22     1_150.31    10_563.53       0.9084          1.0179            1.0127         6.32
SOAROPQ-shift0.3-np1                                   9_649.67       456.00    10_105.68       0.8619          1.0562            1.0271         6.32
SOAROPQ-shift0.3-np2                                   9_649.67       511.26    10_160.93       0.8976          1.0275            1.0163         6.32
SOAROPQ-shift0.3-np4                                   9_649.67       641.43    10_291.10       0.9060          1.0202            1.0135         6.32
SOAROPQ-shift0.3-np7                                   9_649.67       839.94    10_489.61       0.9078          1.0185            1.0129         6.32
SOAROPQ-shift0.3-np8                                   9_649.67       897.27    10_546.94       0.9080          1.0183            1.0129         6.32
SOAROPQ-shift0.3-np12                                  9_649.67     1_138.58    10_788.25       0.9083          1.0180            1.0127         6.32
SOAROPQ-shift0.7-np1                                   9_705.32       444.19    10_149.51       0.8570          1.0624            1.0289         6.32
SOAROPQ-shift0.7-np2                                   9_705.32       506.94    10_212.26       0.8952          1.0306            1.0170         6.32
SOAROPQ-shift0.7-np4                                   9_705.32       660.20    10_365.52       0.9050          1.0219            1.0139         6.32
SOAROPQ-shift0.7-np7                                   9_705.32       837.25    10_542.57       0.9074          1.0191            1.0131         6.32
SOAROPQ-shift0.7-np8                                   9_705.32       908.02    10_613.34       0.9078          1.0187            1.0130         6.32
SOAROPQ-shift0.7-np12                                  9_705.32     1_138.69    10_844.01       0.9083          1.0180            1.0128         6.32
SOAROPQ-orth1-np1                                      9_464.25       442.31     9_906.56       0.8580          1.0609            1.0275         6.32
SOAROPQ-orth1-np2                                      9_464.25       509.25     9_973.50       0.8969          1.0287            1.0165         6.32
SOAROPQ-orth1-np4                                      9_464.25       640.25    10_104.51       0.9062          1.0205            1.0135         6.32
SOAROPQ-orth1-np7                                      9_464.25       890.36    10_354.62       0.9079          1.0185            1.0129         6.32
SOAROPQ-orth1-np8                                      9_464.25       899.47    10_363.72       0.9080          1.0183            1.0128         6.32
SOAROPQ-orth1-np12                                     9_464.25     1_141.18    10_605.43       0.9083          1.0180            1.0127         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.10.1 (commit v0.10.1-18-g25372f1), run on 2026-10-09.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
