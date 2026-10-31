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
Exhaustive (query)                                        11.38       651.09       662.47       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.38     6_357.46     6_368.83       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.95     1_210.36     1_225.31       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    14.95    12_211.36    12_226.31       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              180.47        95.68       276.15       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              180.47       100.58       281.06       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              180.47       135.50       315.97       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    180.47     1_401.76     1_582.24       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              263.09        90.71       353.80       0.9820          1.0001            1.0000         9.20
IVF-BF16-nl387-np27 (query)                              263.09       118.46       381.55       0.9828          1.0001            1.0000         9.20
IVF-BF16-nl387 (self)                                    263.09     1_219.41     1_482.51       0.9798          1.0001            1.0000         9.20
IVF-BF16-nl547-np23 (query)                              548.79        85.77       634.57       0.9776          1.0005            1.0000         9.22
IVF-BF16-nl547-np27 (query)                              548.79        93.18       641.98       0.9817          1.0002            1.0000         9.22
IVF-BF16-nl547-np33 (query)                              548.79       110.74       659.54       0.9828          1.0001            1.0000         9.22
IVF-BF16-nl547 (self)                                    548.79     1_126.47     1_675.27       0.9798          1.0001            1.0000         9.22
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
Exhaustive (query)                                        12.89       732.35       745.24       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.89     7_237.22     7_250.11       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   13.65     1_321.97     1_335.62       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    13.65    13_028.26    13_041.91       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              376.55       104.57       481.13       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              376.55       127.89       504.44       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              376.55       146.52       523.08       0.8871          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    376.55     1_532.84     1_909.40       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              434.26        94.83       529.09       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              434.26       134.93       569.19       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    434.26     1_315.40     1_749.66       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              498.19        87.21       585.40       0.8849          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              498.19        98.97       597.16       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              498.19       117.35       615.55       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    498.19     1_207.42     1_705.62       0.8852          1.0073            1.0020         9.51
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
Exhaustive (query)                                        11.44       658.19       669.63       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.44     6_409.11     6_420.54       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.12     1_226.52     1_240.64       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    14.12    12_053.59    12_067.71       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              526.60        85.47       612.07       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              526.60       102.39       628.99       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              526.60       119.56       646.16       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    526.60     1_182.16     1_708.76       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              404.61        85.28       489.89       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387-np27 (query)                              404.61       107.60       512.20       0.9344          1.0018            1.0011         9.20
IVF-BF16-nl387 (self)                                    404.61     1_060.36     1_464.97       0.9184          1.0030            1.0021         9.20
IVF-BF16-nl547-np23 (query)                              544.64        80.58       625.22       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np27 (query)                              544.64        88.36       632.99       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547-np33 (query)                              544.64        98.40       643.03       0.9344          1.0018            1.0011         9.22
IVF-BF16-nl547 (self)                                    544.64       975.32     1_519.96       0.9184          1.0030            1.0021         9.22
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
Exhaustive (query)                                        12.27       662.38       674.65       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.27     6_344.89     6_357.17       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   16.50     1_199.60     1_216.09       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    16.50    11_896.96    11_913.46       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              307.29        81.32       388.60       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              307.29        89.65       396.94       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              307.29       121.57       428.85       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    307.29     1_195.08     1_502.36       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              307.29        82.00       389.30       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387-np27 (query)                              307.29       108.29       415.58       0.9541          1.0010            1.0003         9.20
IVF-BF16-nl387 (self)                                    307.29     1_045.69     1_352.99       0.9429          1.0017            1.0009         9.20
IVF-BF16-nl547-np23 (query)                              599.19        78.07       677.26       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np27 (query)                              599.19        85.63       684.82       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547-np33 (query)                              599.19        99.11       698.30       0.9541          1.0010            1.0003         9.22
IVF-BF16-nl547 (self)                                    599.19       977.23     1_576.42       0.9429          1.0017            1.0009         9.22
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
Exhaustive (query)                                        50.09     1_235.35     1_285.44       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.09    12_008.75    12_058.84       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   60.49     5_261.18     5_321.67       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    60.49    54_812.59    54_873.08       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                            1_024.68       271.75     1_296.43       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                            1_024.68       316.51     1_341.19       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                            1_024.68       432.33     1_457.01       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                  1_024.68     4_333.88     5_358.56       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              793.62       285.43     1_079.04       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              793.62       374.78     1_168.40       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    793.62     3_674.67     4_468.28       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            1_425.00       265.64     1_690.64       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            1_425.00       312.91     1_737.92       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            1_425.00       360.04     1_785.04       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                  1_425.00     3_390.15     4_815.15       0.9679          1.0005            1.0000        36.89
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
Exhaustive (query)                                        11.56       652.25       663.81       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.56     6_278.67     6_290.23       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    20.57     1_000.29     1_020.86       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     20.57    10_024.79    10_045.36       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               223.50        70.54       294.04       0.9233          1.0021            1.0010         5.76
IVF-SQ8-nl273-np16 (query)                               223.50        76.44       299.94       0.9246          1.0019            1.0009         5.76
IVF-SQ8-nl273-np23 (query)                               223.50        92.36       315.86       0.9249          1.0019            1.0009         5.76
IVF-SQ8-nl273 (self)                                     223.50       921.10     1_144.60       0.9249          1.0018            1.0009         5.76
IVF-SQ8-nl387-np19 (query)                               280.95        64.08       345.02       0.9255          1.0019            1.0009         5.77
IVF-SQ8-nl387-np27 (query)                               280.95       100.09       381.04       0.9260          1.0018            1.0009         5.77
IVF-SQ8-nl387 (self)                                     280.95       820.06     1_101.00       0.9252          1.0018            1.0009         5.77
IVF-SQ8-nl547-np23 (query)                               469.44        59.73       529.17       0.9223          1.0022            1.0010         5.79
IVF-SQ8-nl547-np27 (query)                               469.44        65.05       534.49       0.9250          1.0019            1.0009         5.79
IVF-SQ8-nl547-np33 (query)                               469.44        76.73       546.16       0.9257          1.0018            1.0009         5.79
IVF-SQ8-nl547 (self)                                     469.44       760.43     1_229.86       0.9252          1.0018            1.0009         5.79
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
Exhaustive (query)                                        12.01       744.42       756.43       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.01     7_299.15     7_311.16       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    23.55       928.96       952.51       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     23.55    10_416.34    10_439.89       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               168.83        60.97       229.81       0.7368          1.0365            1.0161         5.76
IVF-SQ8-nl273-np16 (query)                               168.83        71.94       240.77       0.7369          1.0365            1.0161         5.76
IVF-SQ8-nl273-np23 (query)                               168.83        91.43       260.26       0.7369          1.0365            1.0161         5.76
IVF-SQ8-nl273 (self)                                     168.83       958.21     1_127.04       0.7358          1.0368            1.0158         5.76
IVF-SQ8-nl387-np19 (query)                               271.38        60.49       331.87       0.7379          1.0358            1.0161         5.77
IVF-SQ8-nl387-np27 (query)                               271.38        79.20       350.58       0.7380          1.0358            1.0161         5.77
IVF-SQ8-nl387 (self)                                     271.38       824.13     1_095.51       0.7387          1.0356            1.0157         5.77
IVF-SQ8-nl547-np23 (query)                               449.41        56.00       505.41       0.7359          1.0369            1.0163         5.79
IVF-SQ8-nl547-np27 (query)                               449.41        62.89       512.30       0.7362          1.0369            1.0161         5.79
IVF-SQ8-nl547-np33 (query)                               449.41        72.30       521.71       0.7362          1.0369            1.0161         5.79
IVF-SQ8-nl547 (self)                                     449.41       767.20     1_216.61       0.7361          1.0368            1.0159         5.79
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
Exhaustive (query)                                        11.75       652.79       664.54       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.75     6_537.54     6_549.29       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    20.15     1_035.13     1_055.27       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     20.15     9_993.45    10_013.60       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               208.43        60.94       269.37       0.8148          1.0165            1.0145         5.76
IVF-SQ8-nl273-np16 (query)                               208.43        65.45       273.88       0.8148          1.0165            1.0145         5.76
IVF-SQ8-nl273-np23 (query)                               208.43        82.28       290.71       0.8148          1.0165            1.0145         5.76
IVF-SQ8-nl273 (self)                                     208.43       814.58     1_023.01       0.8119          1.0175            1.0155         5.76
IVF-SQ8-nl387-np19 (query)                               279.68        60.89       340.57       0.8150          1.0164            1.0145         5.77
IVF-SQ8-nl387-np27 (query)                               279.68        72.84       352.52       0.8150          1.0164            1.0145         5.77
IVF-SQ8-nl387 (self)                                     279.68       722.20     1_001.88       0.8122          1.0174            1.0155         5.77
IVF-SQ8-nl547-np23 (query)                               452.52        58.56       511.08       0.8155          1.0164            1.0146         5.79
IVF-SQ8-nl547-np27 (query)                               452.52        63.33       515.85       0.8155          1.0164            1.0146         5.79
IVF-SQ8-nl547-np33 (query)                               452.52        69.54       522.06       0.8155          1.0164            1.0146         5.79
IVF-SQ8-nl547 (self)                                     452.52       717.54     1_170.06       0.8122          1.0174            1.0155         5.79
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
Exhaustive (query)                                        11.18       655.64       666.82       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.18     6_297.88     6_309.05       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    49.62       993.36     1_042.98       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     49.62    10_015.21    10_064.83       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                               202.85        55.18       258.03       0.7906          1.0265            1.0241         5.76
IVF-SQ8-nl273-np16 (query)                               202.85        61.96       264.81       0.7906          1.0265            1.0241         5.76
IVF-SQ8-nl273-np23 (query)                               202.85        81.88       284.73       0.7906          1.0265            1.0241         5.76
IVF-SQ8-nl273 (self)                                     202.85       819.48     1_022.34       0.7897          1.0281            1.0256         5.76
IVF-SQ8-nl387-np19 (query)                               281.41        57.42       338.83       0.7899          1.0265            1.0243         5.77
IVF-SQ8-nl387-np27 (query)                               281.41        72.90       354.31       0.7899          1.0265            1.0243         5.77
IVF-SQ8-nl387 (self)                                     281.41       718.11       999.52       0.7901          1.0280            1.0256         5.77
IVF-SQ8-nl547-np23 (query)                               468.23        55.30       523.52       0.7897          1.0264            1.0241         5.79
IVF-SQ8-nl547-np27 (query)                               468.23        60.36       528.59       0.7897          1.0264            1.0241         5.79
IVF-SQ8-nl547-np33 (query)                               468.23        68.50       536.72       0.7897          1.0264            1.0241         5.79
IVF-SQ8-nl547 (self)                                     468.23       678.60     1_146.82       0.7902          1.0280            1.0256         5.79
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
Exhaustive (query)                                        50.10     1_217.81     1_267.91       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.10    11_984.62    12_034.72       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    79.91     1_202.10     1_282.01       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     79.91    12_156.28    12_236.19       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               463.79        79.15       542.94       0.8800          1.0061            1.0050        19.59
IVF-SQ8-nl273-np16 (query)                               463.79        84.11       547.90       0.8800          1.0061            1.0050        19.59
IVF-SQ8-nl273-np23 (query)                               463.79       110.87       574.66       0.8800          1.0061            1.0050        19.59
IVF-SQ8-nl273 (self)                                     463.79       918.17     1_381.96       0.8865          1.0073            1.0059        19.59
IVF-SQ8-nl387-np19 (query)                               732.80        84.60       817.40       0.8800          1.0061            1.0051        19.65
IVF-SQ8-nl387-np27 (query)                               732.80       101.74       834.54       0.8800          1.0061            1.0051        19.65
IVF-SQ8-nl387 (self)                                     732.80       809.77     1_542.57       0.8867          1.0073            1.0059        19.65
IVF-SQ8-nl547-np23 (query)                             1_148.39        88.87     1_237.25       0.8799          1.0061            1.0051        19.73
IVF-SQ8-nl547-np27 (query)                             1_148.39        91.76     1_240.15       0.8799          1.0061            1.0051        19.73
IVF-SQ8-nl547-np33 (query)                             1_148.39       102.41     1_250.80       0.8799          1.0061            1.0051        19.73
IVF-SQ8-nl547 (self)                                   1_148.39       789.05     1_937.44       0.8865          1.0073            1.0059        19.73
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
Exhaustive (query)                                        11.40       658.48       669.87       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.40     6_305.27     6_316.67       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.17     1_083.51     1_100.68       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               871.02        59.41       930.44       0.9305          1.0141            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              871.02        89.83       960.86       0.9644          1.0086            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              871.02       174.13     1_045.16       0.9835          1.0027            1.0000        38.52
HNSW-M16-ef100 (self)                                    871.02       888.12     1_759.15       0.9647          1.0075            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_573.93        61.72     1_635.65       0.9552          1.0723            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_573.93        93.39     1_667.31       0.9798          1.0427            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_573.93       176.45     1_750.37       0.9903          1.0171            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_573.93       941.37     2_515.29       0.9802          1.0396            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_687.71        58.08     1_745.79       0.9691          1.0259            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_687.71       113.14     1_800.84       0.9875          1.0162            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_687.71       185.44     1_873.15       0.9949          1.0060            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_687.71     1_022.96     2_710.66       0.9877          1.0134            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_741.76        61.17     1_802.94       0.9739          1.0053            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_741.76       109.21     1_850.97       0.9899          1.0033            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_741.76       203.72     1_945.48       0.9963          1.0013            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_741.76     1_050.50     2_792.26       0.9904          1.0019            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          716.22        45.61       761.82       0.8772          1.0277            1.0032        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         716.22        65.82       782.04       0.9015          1.0203            1.0020        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         716.22       123.24       839.46       0.9147          1.0071            1.0014        26.32
HNSW-SQ8U-M16-ef100 (self)                               716.22       618.15     1_334.36       0.9017          1.0129            1.0020        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_407.16        38.00     1_445.17       0.8983          1.0175            1.0021        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_407.16        70.44     1_477.60       0.9143          1.0108            1.0014        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_407.16       133.51     1_540.67       0.9206          1.0076            1.0011        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_407.16       672.07     2_079.23       0.9138          1.0112            1.0014        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_521.44        40.98     1_562.42       0.9056          1.0162            1.0017        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_521.44        75.57     1_597.01       0.9180          1.0093            1.0012        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_521.44       140.00     1_661.44       0.9227          1.0037            1.0010        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_521.44       711.82     2_233.26       0.9177          1.0073            1.0012        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_580.69        42.93     1_623.63       0.9090          1.0060            1.0016        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_580.69        79.23     1_659.92       0.9196          1.0040            1.0012        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_580.69       147.87     1_728.56       0.9233          1.0029            1.0010        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_580.69       755.09     2_335.79       0.9192          1.0039            1.0012        44.63
HNSW-SQ8U-drop0 (query)                                1_398.00        73.88     1_471.88       0.8953          1.0070            1.0024        26.32
HNSW-SQ8U-drop0.001 (query)                            1_410.21        68.80     1_479.01       0.9148          1.0064            1.0014        26.32
HNSW-SQ8U-drop0.01 (query)                             1_438.31        82.57     1_520.88       0.8986          1.0069            1.0018        26.32
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
Exhaustive (query)                                        14.61       716.83       731.44       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         14.61     6_989.01     7_003.62       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    22.44       929.29       951.73       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               842.11        56.39       898.50       0.9345          1.0114            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              842.11        91.45       933.56       0.9687          1.0064            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              842.11       167.15     1_009.26       0.9871          1.0036            1.0000        39.09
HNSW-M16-ef100 (self)                                    842.11       895.10     1_737.21       0.9687          1.0080            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_626.86        54.61     1_681.47       0.9643          1.0072            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_626.86        96.89     1_723.75       0.9870          1.0028            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_626.86       176.58     1_803.44       0.9952          1.0012            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_626.86       964.68     2_591.54       0.9872          1.0045            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_762.89        60.40     1_823.30       0.9737          1.0028            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_762.89       112.11     1_875.01       0.9911          1.0010            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_762.89       193.02     1_955.92       0.9970          1.0002            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_762.89     1_051.46     2_814.35       0.9911          1.0010            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_800.58        62.94     1_863.52       0.9756          1.0159            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_800.58       112.80     1_913.38       0.9913          1.0113            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_800.58       197.98     1_998.55       0.9970          1.0039            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_800.58     1_102.17     2_902.75       0.9915          1.0099            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          752.94        36.59       789.53       0.6855          1.0660            1.0288        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         752.94        67.31       820.25       0.7090          1.0484            1.0240        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         752.94       124.38       877.32       0.7227          1.0427            1.0206        26.32
HNSW-SQ8U-M16-ef100 (self)                               752.94       632.31     1_385.25       0.7087          1.0485            1.0239        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_434.29        39.71     1_474.01       0.7071          1.0511            1.0234        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_434.29        70.79     1_505.08       0.7236          1.0447            1.0199        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_434.29       133.43     1_567.73       0.7313          1.0409            1.0185        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_434.29       681.27     2_115.56       0.7232          1.0452            1.0199        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_576.69        43.16     1_619.85       0.7176          1.0639            1.0206        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_576.69        77.76     1_654.45       0.7303          1.0458            1.0185        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_576.69       146.65     1_723.34       0.7355          1.0394            1.0172        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_576.69       754.51     2_331.20       0.7298          1.0437            1.0180        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_631.72        45.48     1_677.20       0.7206          1.1471            1.0197        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_631.72        81.08     1_712.80       0.7317          1.0972            1.0177        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_631.72       154.05     1_785.77       0.7367          1.0381            1.0168        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_631.72       768.10     2_399.82       0.7309          1.1053            1.0175        44.63
HNSW-SQ8U-drop0 (query)                                1_437.75        73.10     1_510.85       0.6640          1.0874            1.0307        26.32
HNSW-SQ8U-drop0.001 (query)                            1_453.89        74.49     1_528.38       0.7237          1.0534            1.0202        26.32
HNSW-SQ8U-drop0.01 (query)                             1_417.36        78.70     1_496.06       0.6895          1.0549            1.0259        26.32
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
Exhaustive (query)                                        11.42       673.49       684.90       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.42     6_696.84     6_708.26       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    22.65     1_106.68     1_129.34       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               802.58        54.35       856.93       0.9751          1.3031            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              802.58        84.28       886.86       0.9769          1.0261            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              802.58       149.90       952.47       0.9771          1.0261            1.0000        38.52
HNSW-M16-ef100 (self)                                    802.58       810.20     1_612.78       0.9772          1.0327            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_464.95        52.22     1_517.17       0.9709         52.5997            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_464.95        96.05     1_561.00       0.9991          1.4096            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_464.95       158.89     1_623.84       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_464.95       896.94     2_361.89       0.9993          1.3552            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_483.51        55.47     1_538.98       0.9992          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_483.51        91.05     1_574.56       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_483.51       164.18     1_647.69       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_483.51       889.92     2_373.43       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_536.73        54.18     1_590.91       0.9992          1.0001            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_536.73        92.08     1_628.81       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_536.73       165.14     1_701.87       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_536.73       887.05     2_423.78       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          723.79        36.65       760.44       0.8132          2.0176            1.0149        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         723.79        64.67       788.46       0.8144          1.0936            1.0148        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         723.79       116.75       840.54       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef100 (self)                               723.79       599.99     1_323.78       0.8118          1.0255            1.0156        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_304.36        38.30     1_342.66       0.8143          1.0166            1.0148        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_304.36        64.52     1_368.88       0.8145          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_304.36       120.47     1_424.82       0.8146          1.0165            1.0148        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_304.36       607.78     1_912.14       0.8119          1.0175            1.0155        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_427.94        48.84     1_476.78       0.8143          1.0166            1.0148        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_427.94        74.07     1_502.01       0.8145          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_427.94       125.17     1_553.11       0.8146          1.0165            1.0148        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_427.94       666.55     2_094.49       0.8119          1.0175            1.0155        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_468.41        41.14     1_509.55       0.8142          1.3556            1.0148        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_468.41        72.15     1_540.55       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_468.41       128.99     1_597.40       0.8146          1.0165            1.0148        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_468.41       660.82     2_129.23       0.8119          1.0175            1.0155        44.63
HNSW-SQ8U-drop0 (query)                                1_301.76        69.37     1_371.13       0.8081          1.0176            1.0157        26.32
HNSW-SQ8U-drop0.001 (query)                            1_308.64        63.64     1_372.28       0.8141          1.1741            1.0148        26.32
HNSW-SQ8U-drop0.01 (query)                             1_306.69        65.67     1_372.37       0.8049          1.0195            1.0163        26.32
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
Exhaustive (query)                                        11.57       686.02       697.59       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.57     6_812.86     6_824.42       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.53     1_100.30     1_119.83       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               889.94        58.59       948.53       0.9980          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              889.94        95.49       985.43       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              889.94       182.09     1_072.03       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    889.94       937.68     1_827.62       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_620.36        60.51     1_680.87       0.9984          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_620.36       102.77     1_723.13       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_620.36       178.81     1_799.17       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_620.36       946.26     2_566.62       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_697.59        69.24     1_766.83       0.9993          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_697.59       111.42     1_809.01       0.9999          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_697.59       189.16     1_886.75       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_697.59     1_056.61     2_754.21       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_780.18        65.18     1_845.36       0.9994          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_780.18       110.70     1_890.88       0.9999          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_780.18       194.14     1_974.32       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_780.18     1_072.93     2_853.12       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          816.75        50.13       866.87       0.7888          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef100-s100 (query)                         816.75        79.30       896.05       0.7892          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100-s200 (query)                         816.75       137.11       953.86       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef100 (self)                               816.75       677.26     1_494.01       0.7896          1.0282            1.0258        26.32
HNSW-SQ8U-M16-ef200-s50 (query)                        1_458.13        37.72     1_495.85       0.7890          1.0267            1.0245        26.32
HNSW-SQ8U-M16-ef200-s100 (query)                       1_458.13        69.77     1_527.90       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200-s200 (query)                       1_458.13       128.52     1_586.65       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-M16-ef200 (self)                             1_458.13       660.91     2_119.05       0.7897          1.0281            1.0258        26.32
HNSW-SQ8U-M24-ef200-s50 (query)                        1_535.01        42.52     1_577.53       0.7892          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s100 (query)                       1_535.01        78.65     1_613.65       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200-s200 (query)                       1_535.01       145.93     1_680.94       0.7893          1.0266            1.0244        35.23
HNSW-SQ8U-M24-ef200 (self)                             1_535.01       758.56     2_293.57       0.7897          1.0281            1.0258        35.23
HNSW-SQ8U-M32-ef200-s50 (query)                        1_626.55        45.61     1_672.16       0.7892          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200-s100 (query)                       1_626.55        80.35     1_706.90       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200-s200 (query)                       1_626.55       147.70     1_774.24       0.7893          1.0266            1.0244        44.63
HNSW-SQ8U-M32-ef200 (self)                             1_626.55       765.34     2_391.89       0.7897          1.0281            1.0258        44.63
HNSW-SQ8U-drop0 (query)                                1_434.03        71.05     1_505.08       0.7860          1.0279            1.0254        26.32
HNSW-SQ8U-drop0.001 (query)                            1_453.67        71.51     1_525.18       0.7893          1.0266            1.0244        26.32
HNSW-SQ8U-drop0.01 (query)                             1_442.33        74.08     1_516.42       0.7830          1.0294            1.0262        26.32
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
Exhaustive (query)                                        50.68     1_248.10     1_298.78       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.68    12_705.77    12_756.46       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    79.34     1_232.64     1_311.97       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_428.30        83.49     1_511.79       0.9928          1.0373            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_428.30       138.51     1_566.81       0.9955          1.0215            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_428.30       251.04     1_679.34       0.9971          1.0120            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_428.30     1_359.92     2_788.22       0.9956          1.0191            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_481.85        84.88     2_566.74       0.9971          1.0167            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_481.85       147.45     2_629.30       0.9984          1.0079            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_481.85       255.98     2_737.84       0.9990          1.0032            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_481.85     1_423.00     3_904.86       0.9978          1.0121            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_618.02        89.56     2_707.58       0.9986          1.0068            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_618.02       153.56     2_771.59       0.9991          1.0049            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_618.02       272.72     2_890.75       0.9997          1.0009            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_618.02     1_512.17     4_130.19       0.9992          1.0035            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_756.64        94.67     2_851.31       0.9993          1.0037            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_756.64       155.84     2_912.47       0.9996          1.0022            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_756.64       271.45     3_028.09       0.9998          1.0012            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_756.64     1_523.60     4_280.23       0.9995          1.0025            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          884.51        40.40       924.91       0.9289          1.0346            1.0038        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         884.51        71.25       955.76       0.9308          1.0223            1.0038        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         884.51       126.30     1_010.81       0.9321          1.0145            1.0037        40.05
HNSW-SQ8U-M16-ef100 (self)                               884.51       666.85     1_551.36       0.9307          1.0230            1.0038        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_577.64        44.51     1_622.14       0.9305          1.0327            1.0037        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_577.64        74.98     1_652.62       0.9320          1.0206            1.0037        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_577.64       133.70     1_711.33       0.9328          1.0147            1.0037        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_577.64       701.41     2_279.05       0.9315          1.0244            1.0037        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_646.33        44.46     1_690.79       0.9324          1.0173            1.0036        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_646.33        77.93     1_724.25       0.9334          1.0114            1.0036        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_646.33       140.59     1_786.92       0.9338          1.0082            1.0036        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_646.33       766.22     2_412.54       0.9332          1.0112            1.0036        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_724.51        45.41     1_769.92       0.9337          1.0091            1.0036        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_724.51        78.93     1_803.44       0.9339          1.0078            1.0036        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_724.51       137.35     1_861.86       0.9340          1.0076            1.0036        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_724.51       766.12     2_490.64       0.9336          1.0088            1.0036        58.36
HNSW-SQ8U-drop0 (query)                                1_533.45        69.70     1_603.15       0.8648          1.0383            1.0220        40.05
HNSW-SQ8U-drop0.001 (query)                            1_528.45        74.13     1_602.57       0.9324          1.0161            1.0036        40.05
HNSW-SQ8U-drop0.01 (query)                             1_518.92        68.72     1_587.64       0.9316          1.0447            1.0020        40.05
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
Exhaustive (query)                                        52.00     1_325.55     1_377.55       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         52.00    13_407.39    13_459.39       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    85.43     1_252.95     1_338.38       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_241.14        73.10     1_314.24       0.9941          1.0909            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_241.14       117.86     1_359.00       0.9962          1.0475            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_241.14       213.20     1_454.34       0.9978          1.0159            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_241.14     1_148.54     2_389.68       0.9965          1.0354            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_297.60        73.86     2_371.46       0.9941          1.1174            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_297.60       125.13     2_422.73       0.9966          1.0578            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_297.60       228.70     2_526.29       0.9981          1.0228            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_297.60     1_216.47     3_514.06       0.9961          1.0680            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_447.03        77.41     2_524.44       0.9986          1.0196            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_447.03       130.72     2_577.75       0.9994          1.0070            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_447.03       232.77     2_679.81       0.9999          1.0007            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_447.03     1_300.41     3_747.45       0.9991          1.0120            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_507.64        77.44     2_585.09       0.9989          1.0258            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_507.64       130.66     2_638.30       0.9996          1.0059            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_507.64       232.72     2_740.37       0.9998          1.0014            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_507.64     1_273.08     3_780.73       0.9994          1.0093            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          813.39        40.79       854.18       0.6646          1.4063            1.1633        40.05
HNSW-SQ8U-M16-ef100-s100 (query)                         813.39        69.27       882.66       0.6656          1.3759            1.1627        40.05
HNSW-SQ8U-M16-ef100-s200 (query)                         813.39       126.25       939.65       0.6663          1.3609            1.1624        40.05
HNSW-SQ8U-M16-ef100 (self)                               813.39       634.18     1_447.58       0.6654          1.3754            1.1630        40.05
HNSW-SQ8U-M16-ef200-s50 (query)                        1_498.22        41.79     1_540.01       0.6625          1.4746            1.1645        40.05
HNSW-SQ8U-M16-ef200-s100 (query)                       1_498.22        74.23     1_572.45       0.6650          1.4027            1.1627        40.05
HNSW-SQ8U-M16-ef200-s200 (query)                       1_498.22       129.35     1_627.57       0.6665          1.3673            1.1617        40.05
HNSW-SQ8U-M16-ef200 (self)                             1_498.22       653.08     2_151.30       0.6648          1.4015            1.1631        40.05
HNSW-SQ8U-M24-ef200-s50 (query)                        1_601.36        42.93     1_644.29       0.6662          1.3681            1.1621        48.96
HNSW-SQ8U-M24-ef200-s100 (query)                       1_601.36        72.96     1_674.32       0.6669          1.3586            1.1616        48.96
HNSW-SQ8U-M24-ef200-s200 (query)                       1_601.36       138.51     1_739.88       0.6672          1.3535            1.1614        48.96
HNSW-SQ8U-M24-ef200 (self)                             1_601.36       674.23     2_275.59       0.6665          1.3615            1.1619        48.96
HNSW-SQ8U-M32-ef200-s50 (query)                        1_682.21        46.63     1_728.84       0.6670          1.3552            1.1616        58.36
HNSW-SQ8U-M32-ef200-s100 (query)                       1_682.21        81.00     1_763.21       0.6673          1.3508            1.1614        58.36
HNSW-SQ8U-M32-ef200-s200 (query)                       1_682.21       139.82     1_822.03       0.6674          1.3494            1.1614        58.36
HNSW-SQ8U-M32-ef200 (self)                             1_682.21       740.72     2_422.93       0.6669          1.3552            1.1617        58.36
HNSW-SQ8U-drop0 (query)                                1_513.26        72.44     1_585.70       0.6192          1.5301            1.2368        40.05
HNSW-SQ8U-drop0.001 (query)                            1_505.52        70.91     1_576.43       0.6649          1.4054            1.1630        40.05
HNSW-SQ8U-drop0.01 (query)                             1_488.28        70.54     1_558.82       0.6795          1.3431            1.1498        40.05
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
Exhaustive (query)                                        32.76       706.92       739.68       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.76     2_330.10     2_362.86       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                688.97       664.50     1_353.46       0.2581          1.1827            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 688.97     2_203.40     2_892.36       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_272.01     1_527.61     2_799.62       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_272.01     5_107.20     6_379.21       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              1_999.62     3_624.71     5_624.33       0.3611          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               1_999.62    12_054.84    14_054.46       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                             894.00       205.62     1_099.62       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np12 (query)                            894.00       320.09     1_214.09       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np17 (query)                            894.00       442.33     1_336.33       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16 (self)                                  894.00     1_480.62     2_374.61       0.3038          1.1285            1.1336         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_372.53       373.40     1_745.93       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_372.53       576.63     1_949.16       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_372.53       800.22     2_172.75       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32 (self)                                1_372.53     2_640.34     4_012.87       0.4075          1.0802            1.0796         1.93
IVF-PQ-nl158-m64-np7 (query)                           1_845.32       679.06     2_524.38       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np12 (query)                          1_845.32     1_026.60     2_871.92       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np17 (query)                          1_845.32     1_406.01     3_251.33       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64 (self)                                1_845.32     4_689.93     6_535.25       0.6325          1.0279            1.0244         3.46
IVF-PQ-nl223-m16-np11 (query)                            894.95       306.30     1_201.25       0.3866          1.0890            1.0897         1.23
IVF-PQ-nl223-m16-np14 (query)                            894.95       372.69     1_267.63       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16-np21 (query)                            894.95       549.98     1_444.93       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16 (self)                                  894.95     1_803.03     2_697.98       0.3106          1.1231            1.1271         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_398.01       535.26     1_933.27       0.4961          1.0568            1.0521         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_398.01       662.42     2_060.43       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_398.01       959.21     2_357.23       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32 (self)                                1_398.01     3_196.48     4_594.50       0.4138          1.0784            1.0759         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_044.04       951.54     2_995.58       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_044.04     1_175.73     3_219.77       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_044.04     1_706.93     3_750.97       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64 (self)                                2_044.04     5_967.11     8_011.15       0.6393          1.0273            1.0234         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_020.38       380.09     1_400.47       0.3990          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_020.38       425.71     1_446.09       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_020.38       606.81     1_627.19       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16 (self)                                1_020.38     2_007.42     3_027.80       0.3170          1.1180            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_486.66       685.70     2_172.36       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_486.66       772.70     2_259.37       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_486.66     1_095.82     2_582.48       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32 (self)                                1_486.66     3_672.56     5_159.23       0.4239          1.0739            1.0729         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_089.11     1_210.00     3_299.11       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_089.11     1_335.58     3_424.69       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_089.11     1_911.48     4_000.59       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64 (self)                                2_089.11     6_354.44     8_443.56       0.6491          1.0248            1.0220         3.61
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
Exhaustive (query)                                        67.97     1_279.03     1_347.00       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.97     4_212.76     4_280.73       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                895.38       679.72     1_575.10       0.2443          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                                 895.38     2_211.73     3_107.12       0.2277          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_306.17     1_538.15     2_844.32       0.2649          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_306.17     5_074.95     6_381.12       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_223.07     3_668.09     5_891.16       0.2958          1.0991            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_223.07    12_077.34    14_300.41       0.2627          1.1103            1.1143         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_173.85       273.92     1_447.77       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_173.85       422.24     1_596.09       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_173.85       582.51     1_756.36       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16 (self)                                1_173.85     2_019.77     3_193.62       0.2613          1.1090            1.1156         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_590.74       409.30     2_000.04       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_590.74       641.03     2_231.78       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_590.74       872.62     2_463.36       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32 (self)                                1_590.74     2_897.88     4_488.62       0.2899          1.0920            1.0958         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_543.59       760.08     3_303.67       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_543.59     1_173.74     3_717.32       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_543.59     1_595.05     4_138.63       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64 (self)                                2_543.59     5_578.03     8_121.62       0.3928          1.0580            1.0570         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_203.81       418.52     1_622.33       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_203.81       509.41     1_713.22       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_203.81       751.65     1_955.46       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16 (self)                                1_203.81     2_462.14     3_665.95       0.2659          1.1043            1.1093         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_612.78       608.60     2_221.38       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_612.78       744.65     2_357.44       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_612.78     1_086.82     2_699.60       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                1_612.78     3_605.12     5_217.90       0.2945          1.0892            1.0915         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_652.57     1_069.49     3_722.06       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_652.57     1_322.02     3_974.59       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_652.57     1_958.41     4_610.99       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64 (self)                                2_652.57     6_397.40     9_049.97       0.3971          1.0575            1.0549         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_369.21       545.14     1_914.35       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_369.21       598.33     1_967.54       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_369.21       854.28     2_223.49       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16 (self)                                1_369.21     2_924.96     4_294.17       0.2713          1.0997            1.1062         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_703.46       774.58     2_478.05       0.3797          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_703.46       862.17     2_565.63       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_703.46     1_403.05     3_106.52       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32 (self)                                1_703.46     4_515.80     6_219.26       0.3003          1.0857            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_335.68     1_376.27     4_711.95       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_335.68     1_556.72     4_892.40       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_335.68     2_226.68     5_562.35       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64 (self)                                3_335.68     7_372.60    10_708.28       0.4062          1.0539            1.0530         4.17
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
Exhaustive (query)                                       102.73     1_881.15     1_983.88       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.73     6_069.23     6_171.96       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_183.70       705.02     1_888.72       0.2345          1.1095            1.1000         1.51
Exhaustive-PQ-m16 (self)                               1_183.70     2_245.94     3_429.65       0.2206          1.1180            1.1048         1.51
Exhaustive-PQ-m32 (query)                              1_696.98     1_581.02     3_278.00       0.2567          1.0943            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_696.98     5_172.16     6_869.14       0.2391          1.1012            1.1021         2.28
Exhaustive-PQ-m64 (query)                              2_719.05     3_686.06     6_405.10       0.2775          1.0855            1.0910         3.80
Exhaustive-PQ-m64 (self)                               2_719.05    12_160.67    14_879.71       0.2515          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_744.27     8_039.63    12_783.90       0.3162          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_744.27    26_648.26    31_392.54       0.2755          1.0817            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_700.03       373.92     2_073.96       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_700.03       564.98     2_265.01       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_700.03       782.83     2_482.86       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16 (self)                                1_700.03     2_594.68     4_294.71       0.2512          1.0937            1.1005         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_112.90       548.71     2_661.60       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_112.90       868.38     2_981.28       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_112.90     1_217.82     3_330.71       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32 (self)                                2_112.90     4_175.75     6_288.65       0.2627          1.0859            1.0912         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_151.84       849.05     4_000.89       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_151.84     1_380.70     4_532.54       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_151.84     1_869.40     5_021.25       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64 (self)                                3_151.84     6_201.80     9_353.64       0.3100          1.0666            1.0678         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_268.31     1_645.04     6_913.35       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_268.31     2_616.35     7_884.66       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_268.31     3_581.45     8_849.77       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128 (self)                               5_268.31    11_836.22    17_104.53       0.4627          1.0347            1.0321         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_574.77       520.35     2_095.13       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_574.77       634.09     2_208.87       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_574.77       920.71     2_495.49       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16 (self)                                1_574.77     3_035.63     4_610.40       0.2554          1.0889            1.0948         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_110.75       763.00     2_873.75       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_110.75       942.86     3_053.61       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_110.75     1_371.81     3_482.56       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32 (self)                                2_110.75     4_491.88     6_602.63       0.2681          1.0815            1.0863         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_115.91     1_183.52     4_299.43       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_115.91     1_467.18     4_583.09       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_115.91     2_143.38     5_259.29       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                3_115.91     7_073.02    10_188.93       0.3139          1.0649            1.0653         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_148.61     2_383.54     7_532.15       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_148.61     2_961.20     8_109.81       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_148.61     4_319.53     9_468.14       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128 (self)                               5_148.61    14_324.53    19_473.13       0.4702          1.0336            1.0308         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_719.59       663.25     2_382.84       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_719.59       735.71     2_455.30       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_719.59     1_063.83     2_783.42       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16 (self)                                1_719.59     3_672.48     5_392.07       0.2591          1.0854            1.0920         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_207.66     1_022.80     3_230.46       0.3367          1.0582            1.0610         3.20
IVF-PQ-nl316-m32-np17 (query)                          2_207.66     1_119.67     3_327.34       0.3367          1.0582            1.0610         3.20
IVF-PQ-nl316-m32-np25 (query)                          2_207.66     1_603.14     3_810.80       0.3367          1.0582            1.0610         3.20
IVF-PQ-nl316-m32 (self)                                2_207.66     5_372.40     7_580.06       0.2701          1.0793            1.0841         3.20
IVF-PQ-nl316-m64-np15 (query)                          3_168.10     1_578.27     4_746.38       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_168.10     1_774.58     4_942.68       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_168.10     2_557.38     5_725.48       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64 (self)                                3_168.10     8_555.05    11_723.15       0.3199          1.0622            1.0632         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_228.92     3_112.64     8_341.56       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_228.92     3_494.21     8_723.13       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_228.92     5_031.91    10_260.83       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128 (self)                               5_228.92    16_661.09    21_890.01       0.4774          1.0316            1.0300         7.78
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
Exhaustive (query)                                        33.17       713.22       746.39       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.17     2_319.18     2_352.34       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                690.38       658.34     1_348.71       0.2932          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 690.38     2_159.69     2_850.07       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                              1_253.70     1_532.87     2_786.57       0.4008          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                               1_253.70     5_064.47     6_318.17       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                              1_698.07     3_627.75     5_325.83       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                               1_698.07    12_050.37    13_748.44       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                             880.86       208.47     1_089.34       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np12 (query)                            880.86       324.85     1_205.72       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np17 (query)                            880.86       442.05     1_322.91       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16 (self)                                  880.86     1_446.33     2_327.19       0.4294          1.1639            1.1599         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_361.14       358.23     1_719.37       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_361.14       567.42     1_928.56       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_361.14       785.61     2_146.75       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32 (self)                                1_361.14     2_601.53     3_962.67       0.6060          1.0690            1.0642         1.93
IVF-PQ-nl158-m64-np7 (query)                           1_996.01       651.84     2_647.85       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          1_996.01     1_017.16     3_013.18       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          1_996.01     1_409.97     3_405.98       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                1_996.01     4_688.57     6_684.58       0.7974          1.0165            1.0143         3.46
IVF-PQ-nl223-m16-np11 (query)                            916.89       298.54     1_215.43       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np14 (query)                            916.89       367.38     1_284.27       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np21 (query)                            916.89       539.56     1_456.45       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16 (self)                                  916.89     1_847.36     2_764.25       0.4242          1.1680            1.1638         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_421.09       527.23     1_948.32       0.6766          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_421.09       652.95     2_074.04       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_421.09       965.95     2_387.04       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32 (self)                                1_421.09     3_205.73     4_626.81       0.6033          1.0702            1.0654         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_134.43       922.24     3_056.67       0.8369          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_134.43     1_158.26     3_292.69       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_134.43     1_724.31     3_858.74       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                2_134.43     5_699.11     7_833.54       0.8007          1.0159            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_010.06       422.78     1_432.83       0.5369          1.0879            1.0854         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_010.06       446.04     1_456.09       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_010.06       647.55     1_657.61       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16 (self)                                1_010.06     2_141.71     3_151.77       0.4164          1.1739            1.1698         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_486.66       689.66     2_176.32       0.6804          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_486.66       786.47     2_273.12       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_486.66     1_139.59     2_626.24       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32 (self)                                1_486.66     3_730.05     5_216.70       0.6013          1.0708            1.0663         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_184.27     1_182.55     3_366.82       0.8382          1.0090            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_184.27     1_325.01     3_509.28       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_184.27     1_943.94     4_128.21       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                2_184.27     6_419.91     8_604.18       0.8022          1.0156            1.0137         3.61
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
Exhaustive (query)                                        67.99     1_279.96     1_347.95       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.99     4_158.31     4_226.30       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                898.56       686.73     1_585.28       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 898.56     2_200.82     3_099.37       0.1772          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_318.27     1_553.76     2_872.04       0.2802          1.1736            1.1699         2.03
Exhaustive-PQ-m32 (self)                               1_318.27     5_111.03     6_429.31       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                              2_262.95     3_654.85     5_917.80       0.3752          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                               2_262.95    12_088.50    14_351.46       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_202.43       264.17     1_466.60       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_202.43       409.90     1_612.34       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_202.43       563.32     1_765.76       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16 (self)                                1_202.43     1_845.60     3_048.04       0.2739          1.2064            1.2097         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_608.05       394.88     2_002.93       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_608.05       619.59     2_227.65       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_608.05       844.26     2_452.31       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32 (self)                                1_608.05     2_781.14     4_389.19       0.3946          1.1241            1.1226         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_605.55       722.89     3_328.44       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_605.55     1_137.51     3_743.06       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_605.55     1_565.98     4_171.52       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64 (self)                                2_605.55     5_154.05     7_759.59       0.5740          1.0544            1.0503         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_243.46       416.19     1_659.65       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_243.46       503.18     1_746.64       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_243.46       736.31     1_979.77       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16 (self)                                1_243.46     2_441.41     3_684.87       0.2680          1.2120            1.2166         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_653.40       596.08     2_249.48       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_653.40       736.13     2_389.53       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_653.40     1_074.96     2_728.36       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32 (self)                                1_653.40     3_571.18     5_224.58       0.3847          1.1293            1.1282         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_676.95     1_061.78     3_738.73       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_676.95     1_311.15     3_988.10       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_676.95     1_932.20     4_609.15       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64 (self)                                2_676.95     6_457.91     9_134.86       0.5727          1.0549            1.0508         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_421.06       550.68     1_971.75       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_421.06       605.89     2_026.96       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_421.06       867.33     2_288.40       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16 (self)                                1_421.06     2_841.12     4_262.18       0.2621          1.2176            1.2225         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_865.70       785.48     2_651.18       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_865.70       887.02     2_752.72       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_865.70     1_240.04     3_105.74       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32 (self)                                1_865.70     4_057.39     5_923.09       0.3740          1.1347            1.1339         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_206.10     1_373.41     4_579.51       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_206.10     1_532.53     4_738.63       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_206.10     2_205.76     5_411.86       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64 (self)                                3_206.10     7_274.56    10_480.66       0.5697          1.0555            1.0517         4.17
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
Exhaustive (query)                                       100.48     1_833.73     1_934.21       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.48     6_056.92     6_157.39       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_345.37       695.34     2_040.71       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                               1_345.37     2_247.35     3_592.72       0.1758          1.3086            1.3090         1.51
Exhaustive-PQ-m32 (query)                              1_727.29     1_566.39     3_293.68       0.2712          1.1686            1.1636         2.28
Exhaustive-PQ-m32 (self)                               1_727.29     5_137.21     6_864.50       0.2191          1.2527            1.2505         2.28
Exhaustive-PQ-m64 (query)                              2_774.53     3_671.16     6_445.69       0.3546          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                               2_774.53    12_147.86    14_922.39       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             4_813.78     8_058.47    12_872.25       0.4597          1.0781            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_813.78    26_659.40    31_473.19       0.3908          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_587.54       377.85     1_965.39       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_587.54       605.02     2_192.57       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_587.54       822.77     2_410.31       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16 (self)                                1_587.54     2_596.72     4_184.26       0.2589          1.2185            1.2225         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_165.01       536.05     2_701.06       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_165.01       848.20     3_013.21       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_165.01     1_175.46     3_340.47       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32 (self)                                2_165.01     3_945.24     6_110.25       0.3677          1.1383            1.1375         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_250.90       836.05     4_086.95       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_250.90     1_327.33     4_578.23       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_250.90     1_836.30     5_087.20       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64 (self)                                3_250.90     6_094.52     9_345.42       0.5182          1.0711            1.0673         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_319.51     1_620.32     6_939.83       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_319.51     2_590.83     7_910.34       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_319.51     3_574.99     8_894.49       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128 (self)                               5_319.51    11_870.94    17_190.45       0.7139          1.0240            1.0194         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_696.61       520.06     2_216.67       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_696.61       632.21     2_328.83       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_696.61       924.93     2_621.54       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16 (self)                                1_696.61     3_069.36     4_765.97       0.2524          1.2254            1.2305         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_155.16       757.68     2_912.83       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_155.16       937.12     3_092.28       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_155.16     1_394.28     3_549.44       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32 (self)                                2_155.16     4_495.89     6_651.04       0.3495          1.1486            1.1485         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_245.07     1_184.70     4_429.77       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_245.07     1_533.50     4_778.56       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_245.07     2_143.27     5_388.34       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64 (self)                                3_245.07     7_062.38    10_307.44       0.5036          1.0764            1.0726         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_359.95     2_372.05     7_732.00       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_359.95     2_952.95     8_312.90       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_359.95     4_350.01     9_709.95       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128 (self)                               5_359.95    14_415.42    19_775.36       0.7136          1.0239            1.0198         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_758.87       666.29     2_425.16       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_758.87       733.66     2_492.53       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_758.87     1_055.26     2_814.14       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16 (self)                                1_758.87     3_508.40     5_267.28       0.2447          1.2328            1.2393         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_265.04     1_008.21     3_273.25       0.4552          1.0787            1.0773         3.20
IVF-PQ-nl316-m32-np17 (query)                          2_265.04     1_117.34     3_382.38       0.4552          1.0787            1.0773         3.20
IVF-PQ-nl316-m32-np25 (query)                          2_265.04     1_611.27     3_876.31       0.4552          1.0787            1.0773         3.20
IVF-PQ-nl316-m32 (self)                                2_265.04     5_356.31     7_621.35       0.3316          1.1584            1.1592         3.20
IVF-PQ-nl316-m64-np15 (query)                          3_299.32     1_564.85     4_864.17       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_299.32     1_764.40     5_063.72       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_299.32     2_530.90     5_830.22       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                3_299.32     8_397.12    11_696.44       0.4868          1.0819            1.0783         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_641.74     3_105.40     8_747.15       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_641.74     3_486.82     9_128.56       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_641.74     5_031.90    10_673.64       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128 (self)                               5_641.74    16_708.68    22_350.42       0.7111          1.0240            1.0202         7.78
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
Exhaustive (query)                                        33.10       721.70       754.80       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.10     2_356.31     2_389.41       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                705.92       664.05     1_369.97       0.7118          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 705.92     2_172.69     2_878.62       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_232.03     1_523.92     2_755.95       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_232.03     5_063.28     6_295.31       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              2_250.43     3_728.53     5_978.96       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               2_250.43    12_104.19    14_354.62       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_036.92       213.68     1_250.59       0.8274          1.0521            1.0445         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_036.92       347.13     1_384.05       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_036.92       478.76     1_515.68       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16 (self)                                1_036.92     1_607.22     2_644.14       0.7673          1.0987            1.0833         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_512.93       396.65     1_909.58       0.8739          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_512.93       665.07     2_178.00       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_512.93       929.94     2_442.86       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32 (self)                                1_512.93     3_067.59     4_580.52       0.8286          1.0513            1.0425         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_018.76       727.88     2_746.64       0.9048          1.0151            1.0116         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_018.76     1_253.94     3_272.70       0.9055          1.0148            1.0113         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_018.76     1_738.38     3_757.14       0.9055          1.0147            1.0113         3.46
IVF-PQ-nl158-m64 (self)                                2_018.76     5_767.72     7_786.47       0.8704          1.0287            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_106.02       305.20     1_411.21       0.8421          1.0435            1.0371         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_106.02       377.93     1_483.95       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_106.02       560.06     1_666.08       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16 (self)                                1_106.02     1_867.31     2_973.33       0.7842          1.0842            1.0703         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_558.01       551.10     2_109.11       0.8836          1.0226            1.0184         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_558.01       696.79     2_254.80       0.8837          1.0225            1.0184         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_558.01     1_032.02     2_590.03       0.8838          1.0225            1.0184         2.00
IVF-PQ-nl223-m32 (self)                                1_558.01     3_444.25     5_002.25       0.8398          1.0442            1.0357         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_163.42       978.14     3_141.56       0.9103          1.0133            1.0100         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_163.42     1_238.37     3_401.79       0.9104          1.0132            1.0100         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_163.42     1_863.55     4_026.97       0.9105          1.0132            1.0100         3.52
IVF-PQ-nl223-m64 (self)                                2_163.42     6_184.00     8_347.42       0.8772          1.0255            1.0197         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_235.49       392.03     1_627.52       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_235.49       441.69     1_677.18       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_235.49       645.77     1_881.26       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16 (self)                                1_235.49     2_257.21     3_492.69       0.7916          1.0787            1.0640         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_702.01       699.64     2_401.66       0.8864          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_702.01       787.85     2_489.86       0.8865          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_702.01     1_144.06     2_846.07       0.8865          1.0213            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_702.01     3_797.00     5_499.01       0.8427          1.0431            1.0342         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_320.29     1_229.10     3_549.39       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_320.29     1_389.67     3_709.96       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_320.29     2_037.29     4_357.58       0.9129          1.0124            1.0094         3.61
IVF-PQ-nl316-m64 (self)                                2_320.29     6_762.06     9_082.35       0.8794          1.0245            1.0188         3.61
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
Exhaustive (query)                                        68.11     1_320.63     1_388.74       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.11     4_230.45     4_298.56       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                              1_023.25       689.51     1_712.76       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                               1_023.25     2_231.03     3_254.28       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_624.91     1_542.90     3_167.82       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_624.91     5_108.53     6_733.44       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_794.03     3_653.78     6_447.81       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_794.03    12_127.62    14_921.65       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_560.00       279.91     1_839.91       0.8367          1.0503            1.0400         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_560.00       447.64     2_007.64       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_560.00       617.26     2_177.26       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16 (self)                                1_560.00     2_025.37     3_585.37       0.7723          1.1013            1.0737         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_085.38       441.70     2_527.09       0.8684          1.0311            1.0245         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_085.38       712.87     2_798.25       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_085.38       997.96     3_083.34       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32 (self)                                2_085.38     3_285.87     5_371.25       0.8159          1.0642            1.0463         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_233.41       812.65     4_046.07       0.8900          1.0215            1.0164         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_233.41     1_367.43     4_600.84       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_233.41     1_919.76     5_153.17       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64 (self)                                3_233.41     6_309.78     9_543.20       0.8456          1.0436            1.0313         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_650.93       423.06     2_073.99       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_650.93       521.84     2_172.77       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_650.93       779.00     2_429.94       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16 (self)                                1_650.93     2_552.72     4_203.66       0.7969          1.0803            1.0562         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_156.58       629.61     2_786.20       0.8796          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_156.58       782.84     2_939.42       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_156.58     1_154.49     3_311.08       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                2_156.58     3_847.32     6_003.90       0.8298          1.0557            1.0378         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_281.27     1_159.00     4_440.27       0.9001          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_281.27     1_483.94     4_765.22       0.9002          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_281.27     2_127.18     5_408.45       0.9002          1.0178            1.0132         3.99
IVF-PQ-nl223-m64 (self)                                3_281.27     7_125.81    10_407.08       0.8569          1.0374            1.0254         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_703.28       566.70     2_269.98       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_703.28       621.28     2_324.56       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_703.28       897.70     2_600.98       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16 (self)                                1_703.28     2_996.27     4_699.55       0.8151          1.0656            1.0459         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_245.69       801.87     3_047.56       0.8921          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_245.69       898.38     3_144.07       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_245.69     1_297.06     3_542.75       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32 (self)                                2_245.69     4_323.90     6_569.59       0.8453          1.0453            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_509.73     1_426.38     4_936.11       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_509.73     1_599.09     5_108.82       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_509.73     2_352.07     5_861.80       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64 (self)                                3_509.73     7_837.87    11_347.60       0.8660          1.0332            1.0218         4.17
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
Exhaustive (query)                                        99.95     1_846.18     1_946.13       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.95     6_058.23     6_158.18       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_223.11       702.20     1_925.31       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_223.11     2_239.61     3_462.72       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                              1_847.80     1_563.70     3_411.50       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                               1_847.80     5_139.47     6_987.27       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                              2_898.69     3_661.21     6_559.90       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                               2_898.69    12_130.71    15_029.41       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             5_835.37     8_177.31    14_012.67       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              5_835.37    26_589.11    32_424.48       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           2_143.55       404.06     2_547.60       0.8519          1.0423            1.0326         1.98
IVF-PQ-nl158-m16-np12 (query)                          2_143.55       606.00     2_749.55       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16-np17 (query)                          2_143.55       846.03     2_989.58       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16 (self)                                2_143.55     2_753.19     4_896.73       0.7910          1.0832            1.0598         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_474.39       569.18     3_043.57       0.9001          1.0207            1.0128         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_474.39       929.20     3_403.60       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_474.39     1_297.13     3_771.52       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32 (self)                                2_474.39     4_318.91     6_793.30       0.8549          1.0435            1.0236         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_606.55       919.70     4_526.25       0.9204          1.0131            1.0070         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_606.55     1_555.93     5_162.48       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_606.55     2_136.20     5_742.75       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64 (self)                                3_606.55     7_099.51    10_706.06       0.8830          1.0284            1.0134         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_505.97     1_813.07     7_319.04       0.9394          1.0072            1.0031         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_505.97     3_028.89     8_534.86       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_505.97     4_225.88     9_731.85       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128 (self)                               5_505.97    14_040.31    19_546.28       0.9073          1.0171            1.0071         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_114.03       531.64     2_645.68       0.8626          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_114.03       643.03     2_757.06       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_114.03       939.65     3_053.68       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16 (self)                                2_114.03     3_092.88     5_206.92       0.8067          1.0699            1.0512         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_598.10       787.85     3_385.95       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_598.10       981.17     3_579.26       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_598.10     1_437.78     4_035.87       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32 (self)                                2_598.10     4_756.43     7_354.53       0.8676          1.0353            1.0194         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_638.89     1_243.32     4_882.20       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_638.89     1_564.63     5_203.52       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_638.89     2_341.34     5_980.23       0.9269          1.0111            1.0056         4.46
IVF-PQ-nl223-m64 (self)                                3_638.89     7_738.07    11_376.96       0.8921          1.0237            1.0111         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_656.49     2_505.38     8_161.87       0.9441          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_656.49     3_167.27     8_823.76       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_656.49     4_711.70    10_368.19       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128 (self)                               5_656.49    15_612.08    21_268.57       0.9134          1.0147            1.0057         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_412.84       697.43     3_110.27       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_412.84       771.12     3_183.96       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_412.84     1_114.31     3_527.15       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16 (self)                                2_412.84     3_762.08     6_174.92       0.8142          1.0645            1.0463         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_922.18     1_047.93     3_970.11       0.9130          1.0153            1.0092         3.20
IVF-PQ-nl316-m32-np17 (query)                          2_922.18     1_172.28     4_094.46       0.9130          1.0153            1.0092         3.20
IVF-PQ-nl316-m32-np25 (query)                          2_922.18     1_709.95     4_632.14       0.9130          1.0153            1.0092         3.20
IVF-PQ-nl316-m32 (self)                                2_922.18     5_735.09     8_657.27       0.8728          1.0330            1.0175         3.20
IVF-PQ-nl316-m64-np15 (query)                          3_986.04     1_625.02     5_611.06       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_986.04     1_825.42     5_811.46       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_986.04     2_877.47     6_863.52       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64 (self)                                3_986.04     8_918.44    12_904.49       0.8963          1.0221            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_038.41     3_251.53     9_289.94       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_038.41     3_644.94     9_683.35       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_038.41     5_355.03    11_393.44       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128 (self)                               6_038.41    17_645.14    23_683.55       0.9166          1.0138            1.0052         7.78
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
Exhaustive (query)                                        33.15       707.56       740.71       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.15     2_262.30     2_295.46       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_763.18       723.91     4_487.09       0.2869          1.1537            1.1325         1.26
Exhaustive-OPQ-m16 (self)                              3_763.18     2_701.02     6_464.20       0.2599          1.1718            1.1489         1.26
Exhaustive-OPQ-m32 (query)                             6_465.57     1_605.65     8_071.22       0.3256          1.1202            1.1163         2.03
Exhaustive-OPQ-m32 (self)                              6_465.57     5_617.56    12_083.12       0.2833          1.1427            1.1373         2.03
Exhaustive-OPQ-m64 (query)                             9_305.38     3_710.44    13_015.81       0.3790          1.0984            1.0958         3.55
Exhaustive-OPQ-m64 (self)                              9_305.38    12_700.99    22_006.37       0.3225          1.1201            1.1170         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_033.27       267.33     4_300.61       0.3862          1.0891            1.0909         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_033.27       378.97     4_412.24       0.3862          1.0891            1.0909         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_033.27       495.66     4_528.93       0.3862          1.0891            1.0909         1.67
IVF-OPQ-nl158-m16 (self)                               4_033.27     1_991.95     6_025.23       0.3172          1.1191            1.1234         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_459.58       445.51     6_905.09       0.4918          1.0564            1.0551         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_459.58       648.59     7_108.17       0.4918          1.0564            1.0551         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_459.58       860.09     7_319.67       0.4918          1.0564            1.0551         2.43
IVF-OPQ-nl158-m32 (self)                               6_459.58     3_222.93     9_682.51       0.4162          1.0764            1.0770         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_303.57       719.99    10_023.56       0.6951          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_303.57     1_135.83    10_439.40       0.6951          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_303.57     1_464.50    10_768.07       0.6951          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64 (self)                               9_303.57     5_248.02    14_551.58       0.6374          1.0263            1.0238         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_885.41       363.32     4_248.73       0.3979          1.0834            1.0848         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_885.41       434.31     4_319.72       0.3978          1.0834            1.0849         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_885.41       644.51     4_529.92       0.3978          1.0834            1.0849         1.73
IVF-OPQ-nl223-m16 (self)                               3_885.41     2_411.54     6_296.95       0.3213          1.1159            1.1202         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_109.94       610.47     6_720.41       0.5058          1.0527            1.0503         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_109.94       734.20     6_844.15       0.5057          1.0528            1.0503         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_109.94     1_039.39     7_149.34       0.5057          1.0528            1.0503         2.50
IVF-OPQ-nl223-m32 (self)                               6_109.94     3_808.58     9_918.53       0.4234          1.0741            1.0734         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_235.29       997.96    10_233.25       0.7027          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_235.29     1_237.31    10_472.60       0.7028          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_235.29     1_760.65    10_995.94       0.7028          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64 (self)                               9_235.29     6_178.99    15_414.28       0.6447          1.0255            1.0227         4.02
IVF-OPQ-nl316-m16-np15 (query)                         3_843.74       454.64     4_298.37       0.4064          1.0799            1.0814         2.07
IVF-OPQ-nl316-m16-np17 (query)                         3_843.74       504.13     4_347.86       0.4064          1.0800            1.0814         2.07
IVF-OPQ-nl316-m16-np25 (query)                         3_843.74       692.39     4_536.13       0.4064          1.0800            1.0814         2.07
IVF-OPQ-nl316-m16 (self)                               3_843.74     2_647.76     6_491.50       0.3253          1.1130            1.1172         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_229.89       763.08     6_992.97       0.5148          1.0489            1.0480         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_229.89       852.74     7_082.63       0.5148          1.0489            1.0480         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_229.89     1_207.05     7_436.94       0.5148          1.0489            1.0480         2.84
IVF-OPQ-nl316-m32 (self)                               6_229.89     4_259.07    10_488.96       0.4317          1.0704            1.0708         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_391.34     1_269.68    10_661.02       0.7103          1.0164            1.0143         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_391.34     1_419.43    10_810.77       0.7103          1.0164            1.0143         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_391.34     2_060.30    11_451.64       0.7103          1.0164            1.0143         4.36
IVF-OPQ-nl316-m64 (self)                               9_391.34     7_041.98    16_433.33       0.6519          1.0238            1.0216         4.36
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
Exhaustive (query)                                        69.15     1_347.52     1_416.66       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.15     4_442.33     4_511.48       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_954.53     1_033.00     6_987.53       0.2632          1.1171            1.1013         2.26
Exhaustive-OPQ-m16 (self)                              5_954.53     4_772.33    10_726.86       0.2438          1.1286            1.1096         2.26
Exhaustive-OPQ-m32 (query)                             8_107.17     1_874.03     9_981.20       0.2877          1.0971            1.0962         3.03
Exhaustive-OPQ-m32 (self)                              8_107.17     7_550.38    15_657.55       0.2609          1.1084            1.1059         3.03
Exhaustive-OPQ-m64 (query)                            12_884.74     3_987.59    16_872.33       0.3201          1.0820            1.0843         4.55
Exhaustive-OPQ-m64 (self)                             12_884.74    14_516.14    27_400.88       0.2792          1.0963            1.0987         4.55
Exhaustive-OPQ-m128 (query)                           20_620.20     9_091.30    29_711.50       0.3710          1.0677            1.0685         7.61
Exhaustive-OPQ-m128 (self)                            20_620.20    29_698.93    50_319.13       0.3165          1.0822            1.0830         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_275.26       647.61     6_922.88       0.3239          1.0791            1.0832         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_275.26       777.22     7_052.49       0.3239          1.0791            1.0832         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_275.26       954.70     7_229.97       0.3239          1.0791            1.0832         3.07
IVF-OPQ-nl158-m16 (self)                               6_275.26     4_514.42    10_789.68       0.2756          1.0983            1.1037         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_362.54       745.50     9_108.04       0.3685          1.0657            1.0674         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_362.54       968.18     9_330.72       0.3685          1.0657            1.0674         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_362.54     1_198.67     9_561.20       0.3685          1.0657            1.0674         3.84
IVF-OPQ-nl158-m32 (self)                               8_362.54     5_456.04    13_818.58       0.3020          1.0858            1.0896         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_004.09     1_078.16    14_082.25       0.4749          1.0412            1.0405         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_004.09     1_508.32    14_512.41       0.4749          1.0412            1.0405         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_004.09     1_946.28    14_950.37       0.4749          1.0412            1.0405         5.36
IVF-OPQ-nl158-m64 (self)                              13_004.09     7_926.64    20_930.73       0.4013          1.0547            1.0553         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_487.96     1_637.43    21_125.39       0.6842          1.0140            1.0118         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_487.96     2_371.54    21_859.49       0.6842          1.0140            1.0118         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_487.96     3_097.95    22_585.91       0.6842          1.0140            1.0118         8.42
IVF-OPQ-nl158-m128 (self)                             19_487.96    11_827.20    31_315.16       0.6265          1.0191            1.0169         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_625.60       720.12     7_345.73       0.3295          1.0765            1.0794         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_625.60       808.04     7_433.64       0.3296          1.0765            1.0794         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_625.60     1_030.03     7_655.63       0.3296          1.0765            1.0794         3.20
IVF-OPQ-nl223-m16 (self)                               6_625.60     4_815.97    11_441.57       0.2761          1.0978            1.1024         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_395.53       930.44     9_325.97       0.3791          1.0617            1.0627         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_395.53     1_060.15     9_455.68       0.3791          1.0617            1.0627         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_395.53     1_421.99     9_817.53       0.3791          1.0617            1.0627         3.96
IVF-OPQ-nl223-m32 (self)                               8_395.53     6_005.76    14_401.29       0.3039          1.0847            1.0880         3.96
IVF-OPQ-nl223-m64-np11 (query)                        13_419.17     1_408.77    14_827.94       0.4881          1.0386            1.0373         5.49
IVF-OPQ-nl223-m64-np14 (query)                        13_419.17     1_664.13    15_083.30       0.4881          1.0386            1.0373         5.49
IVF-OPQ-nl223-m64-np21 (query)                        13_419.17     2_292.12    15_711.29       0.4881          1.0386            1.0373         5.49
IVF-OPQ-nl223-m64 (self)                              13_419.17     8_987.05    22_406.22       0.4074          1.0534            1.0534         5.49
IVF-OPQ-nl223-m128-np11 (query)                       20_414.31     2_205.02    22_619.33       0.6884          1.0135            1.0113         8.54
IVF-OPQ-nl223-m128-np14 (query)                       20_414.31     2_661.16    23_075.47       0.6884          1.0135            1.0113         8.54
IVF-OPQ-nl223-m128-np21 (query)                       20_414.31     3_676.50    24_090.81       0.6884          1.0135            1.0113         8.54
IVF-OPQ-nl223-m128 (self)                             20_414.31    13_961.64    34_375.95       0.6299          1.0189            1.0167         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_820.43       860.81     7_681.24       0.3376          1.0725            1.0763         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_820.43       932.93     7_753.36       0.3376          1.0725            1.0763         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_820.43     1_202.79     8_023.22       0.3376          1.0725            1.0763         3.88
IVF-OPQ-nl316-m16 (self)                               6_820.43     5_318.04    12_138.47       0.2803          1.0946            1.1005         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_540.20     1_103.39     9_643.59       0.3872          1.0581            1.0601         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_540.20     1_189.13     9_729.33       0.3872          1.0581            1.0601         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_540.20     1_550.47    10_090.68       0.3872          1.0581            1.0601         4.65
IVF-OPQ-nl316-m32 (self)                               8_540.20     6_583.83    15_124.03       0.3088          1.0815            1.0854         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_236.00     1_723.36    14_959.36       0.4993          1.0359            1.0349         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_236.00     1_891.17    15_127.18       0.4993          1.0359            1.0349         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_236.00     2_593.64    15_829.64       0.4993          1.0359            1.0349         6.17
IVF-OPQ-nl316-m64 (self)                              13_236.00    10_207.94    23_443.95       0.4157          1.0508            1.0515         6.17
IVF-OPQ-nl316-m128-np15 (query)                       19_704.47     2_727.07    22_431.53       0.6961          1.0125            1.0107         9.23
IVF-OPQ-nl316-m128-np17 (query)                       19_704.47     3_018.14    22_722.61       0.6961          1.0125            1.0107         9.23
IVF-OPQ-nl316-m128-np25 (query)                       19_704.47     4_206.17    23_910.63       0.6961          1.0125            1.0107         9.23
IVF-OPQ-nl316-m128 (self)                             19_704.47    15_575.99    35_280.45       0.6392          1.0174            1.0158         9.23
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
Exhaustive (query)                                       100.49     1_813.51     1_914.00       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.49     6_051.10     6_151.59       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_334.14     1_510.73    10_844.87       0.2561          1.0979            1.0842         3.76
Exhaustive-OPQ-m16 (self)                              9_334.14     8_103.06    17_437.20       0.2398          1.1074            1.0897         3.76
Exhaustive-OPQ-m32 (query)                            11_361.60     2_323.80    13_685.40       0.2815          1.0769            1.0790         4.53
Exhaustive-OPQ-m32 (self)                             11_361.60    11_116.12    22_477.72       0.2582          1.0860            1.0867         4.53
Exhaustive-OPQ-m64 (query)                            15_641.49     4_371.25    20_012.74       0.2982          1.0718            1.0755         6.05
Exhaustive-OPQ-m64 (self)                             15_641.49    17_649.93    33_291.42       0.2661          1.0823            1.0852         6.05
Exhaustive-OPQ-m128 (query)                           23_889.14     8_621.01    32_510.14       0.3284          1.0636            1.0663         9.11
Exhaustive-OPQ-m128 (self)                            23_889.14    31_866.75    55_755.88       0.2828          1.0759            1.0791         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_204.97     1_154.95    10_359.92       0.3014          1.0694            1.0742         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_204.97     1_325.52    10_530.49       0.3014          1.0694            1.0742         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_204.97     1_509.84    10_714.82       0.3014          1.0694            1.0742         4.98
IVF-OPQ-nl158-m16 (self)                               9_204.97     8_200.73    17_405.71       0.2649          1.0830            1.0886         4.98
IVF-OPQ-nl158-m32-np7 (query)                         11_353.45     1_326.43    12_679.87       0.3315          1.0609            1.0638         5.74
IVF-OPQ-nl158-m32-np12 (query)                        11_353.45     1_621.83    12_975.28       0.3315          1.0609            1.0638         5.74
IVF-OPQ-nl158-m32-np17 (query)                        11_353.45     1_918.36    13_271.81       0.3315          1.0609            1.0638         5.74
IVF-OPQ-nl158-m32 (self)                              11_353.45     9_544.15    20_897.60       0.2753          1.0782            1.0827         5.74
IVF-OPQ-nl158-m64-np7 (query)                         15_659.92     1_672.74    17_332.66       0.3897          1.0481            1.0483         7.27
IVF-OPQ-nl158-m64-np12 (query)                        15_659.92     2_152.93    17_812.84       0.3897          1.0481            1.0483         7.27
IVF-OPQ-nl158-m64-np17 (query)                        15_659.92     2_637.75    18_297.67       0.3897          1.0481            1.0483         7.27
IVF-OPQ-nl158-m64 (self)                              15_659.92    12_038.28    27_698.19       0.3200          1.0628            1.0648         7.27
IVF-OPQ-nl158-m128-np7 (query)                        24_657.65     2_473.49    27_131.15       0.5406          1.0248            1.0225        10.32
IVF-OPQ-nl158-m128-np12 (query)                       24_657.65     3_420.56    28_078.21       0.5406          1.0248            1.0225        10.32
IVF-OPQ-nl158-m128-np17 (query)                       24_657.65     4_384.38    29_042.03       0.5406          1.0248            1.0225        10.32
IVF-OPQ-nl158-m128 (self)                             24_657.65    17_801.14    42_458.79       0.4711          1.0322            1.0310        10.32
IVF-OPQ-nl223-m16-np11 (query)                         9_702.70     1_296.13    10_998.83       0.3082          1.0666            1.0707         5.17
IVF-OPQ-nl223-m16-np14 (query)                         9_702.70     1_409.82    11_112.51       0.3082          1.0666            1.0707         5.17
IVF-OPQ-nl223-m16-np21 (query)                         9_702.70     1_685.13    11_387.83       0.3082          1.0666            1.0707         5.17
IVF-OPQ-nl223-m16 (self)                               9_702.70     8_731.95    18_434.65       0.2670          1.0819            1.0875         5.17
IVF-OPQ-nl223-m32-np11 (query)                        11_806.85     1_567.16    13_374.01       0.3386          1.0579            1.0605         5.93
IVF-OPQ-nl223-m32-np14 (query)                        11_806.85     1_742.84    13_549.69       0.3386          1.0579            1.0605         5.93
IVF-OPQ-nl223-m32-np21 (query)                        11_806.85     2_161.80    13_968.65       0.3386          1.0579            1.0605         5.93
IVF-OPQ-nl223-m32 (self)                              11_806.85    10_364.05    22_170.90       0.2765          1.0770            1.0817         5.93
IVF-OPQ-nl223-m64-np11 (query)                        16_232.98     2_046.12    18_279.10       0.4042          1.0442            1.0443         7.46
IVF-OPQ-nl223-m64-np14 (query)                        16_232.98     2_391.36    18_624.34       0.4042          1.0442            1.0443         7.46
IVF-OPQ-nl223-m64-np21 (query)                        16_232.98     3_021.79    19_254.77       0.4042          1.0442            1.0443         7.46
IVF-OPQ-nl223-m64 (self)                              16_232.98    13_298.91    29_531.89       0.3232          1.0612            1.0633         7.46
IVF-OPQ-nl223-m128-np11 (query)                       24_457.63     3_203.42    27_661.05       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128-np14 (query)                       24_457.63     3_790.91    28_248.53       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128-np21 (query)                       24_457.63     5_148.83    29_606.46       0.5535          1.0227            1.0208        10.51
IVF-OPQ-nl223-m128 (self)                             24_457.63    20_318.89    44_776.52       0.4788          1.0309            1.0300        10.51
IVF-OPQ-nl316-m16-np15 (query)                         9_818.13     1_446.65    11_264.78       0.3122          1.0645            1.0692         6.19
IVF-OPQ-nl316-m16-np17 (query)                         9_818.13     1_514.97    11_333.10       0.3123          1.0645            1.0692         6.19
IVF-OPQ-nl316-m16-np25 (query)                         9_818.13     1_800.50    11_618.63       0.3123          1.0645            1.0692         6.19
IVF-OPQ-nl316-m16 (self)                               9_818.13     9_134.68    18_952.81       0.2692          1.0801            1.0861         6.19
IVF-OPQ-nl316-m32-np15 (query)                        12_043.62     1_794.00    13_837.62       0.3440          1.0560            1.0589         6.95
IVF-OPQ-nl316-m32-np17 (query)                        12_043.62     1_903.58    13_947.20       0.3440          1.0560            1.0589         6.95
IVF-OPQ-nl316-m32-np25 (query)                        12_043.62     2_370.59    14_414.21       0.3440          1.0560            1.0589         6.95
IVF-OPQ-nl316-m32 (self)                              12_043.62    11_076.60    23_120.23       0.2782          1.0757            1.0805         6.95
IVF-OPQ-nl316-m64-np15 (query)                        16_359.80     2_413.82    18_773.62       0.4099          1.0424            1.0428         8.48
IVF-OPQ-nl316-m64-np17 (query)                        16_359.80     2_598.03    18_957.83       0.4099          1.0424            1.0428         8.48
IVF-OPQ-nl316-m64-np25 (query)                        16_359.80     3_377.42    19_737.22       0.4099          1.0424            1.0428         8.48
IVF-OPQ-nl316-m64 (self)                              16_359.80    14_468.58    30_828.39       0.3271          1.0598            1.0620         8.48
IVF-OPQ-nl316-m128-np15 (query)                       25_804.87     3_975.08    29_779.94       0.5627          1.0213            1.0199        11.53
IVF-OPQ-nl316-m128-np17 (query)                       25_804.87     4_322.20    30_127.07       0.5627          1.0213            1.0199        11.53
IVF-OPQ-nl316-m128-np25 (query)                       25_804.87     5_863.84    31_668.71       0.5627          1.0213            1.0199        11.53
IVF-OPQ-nl316-m128 (self)                             25_804.87    22_679.57    48_484.44       0.4864          1.0295            1.0290        11.53
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
Exhaustive (query)                                        33.01       728.95       761.96       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.01     2_397.05     2_430.06       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_621.13       734.26     4_355.39       0.3004          1.2504            1.2442         1.26
Exhaustive-OPQ-m16 (self)                              3_621.13     2_705.55     6_326.68       0.2368          1.3776            1.3713         1.26
Exhaustive-OPQ-m32 (query)                             5_950.39     1_581.82     7_532.21       0.4204          1.1525            1.1470         2.03
Exhaustive-OPQ-m32 (self)                              5_950.39     5_538.15    11_488.53       0.3373          1.2482            1.2417         2.03
Exhaustive-OPQ-m64 (query)                             9_241.03     3_707.00    12_948.02       0.5660          1.0764            1.0736         3.55
Exhaustive-OPQ-m64 (self)                              9_241.03    12_641.98    21_883.01       0.4873          1.1289            1.1245         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_735.34       270.31     4_005.64       0.6995          1.0326            1.0311         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_735.34       379.80     4_115.14       0.6995          1.0326            1.0311         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_735.34       503.88     4_239.21       0.6995          1.0326            1.0311         1.67
IVF-OPQ-nl158-m16 (self)                               3_735.34     2_031.32     5_766.65       0.6201          1.0627            1.0600         1.67
IVF-OPQ-nl158-m32-np7 (query)                          5_995.22       430.40     6_425.62       0.7977          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np12 (query)                         5_995.22       646.67     6_641.90       0.7977          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np17 (query)                         5_995.22       886.19     6_881.41       0.7977          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32 (self)                               5_995.22     3_302.98     9_298.21       0.7472          1.0260            1.0239         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_222.90       705.79     9_928.69       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_222.90     1_084.16    10_307.06       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_222.90     1_491.18    10_714.07       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64 (self)                               9_222.90     5_325.70    14_548.59       0.8307          1.0113            1.0097         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_962.11       363.23     4_325.35       0.7072          1.0309            1.0294         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_962.11       432.21     4_394.32       0.7073          1.0309            1.0294         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_962.11       609.43     4_571.55       0.7073          1.0309            1.0294         1.73
IVF-OPQ-nl223-m16 (self)                               3_962.11     2_411.52     6_373.63       0.6292          1.0594            1.0569         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_231.44       594.26     6_825.71       0.8045          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_231.44       725.12     6_956.56       0.8046          1.0131            1.0118         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_231.44     1_044.00     7_275.45       0.8046          1.0131            1.0118         2.50
IVF-OPQ-nl223-m32 (self)                               6_231.44     3_864.71    10_096.15       0.7560          1.0242            1.0223         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_532.79       992.73    10_525.52       0.8638          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_532.79     1_216.38    10_749.17       0.8639          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_532.79     1_771.79    11_304.58       0.8639          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64 (self)                               9_532.79     6_281.02    15_813.81       0.8346          1.0106            1.0092         4.02
IVF-OPQ-nl316-m16-np15 (query)                         3_931.67       465.94     4_397.61       0.7124          1.0298            1.0282         2.07
IVF-OPQ-nl316-m16-np17 (query)                         3_931.67       501.79     4_433.46       0.7125          1.0297            1.0281         2.07
IVF-OPQ-nl316-m16-np25 (query)                         3_931.67       733.55     4_665.22       0.7125          1.0297            1.0281         2.07
IVF-OPQ-nl316-m16 (self)                               3_931.67     2_668.30     6_599.97       0.6350          1.0572            1.0547         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_263.93       761.14     7_025.08       0.8081          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_263.93       845.01     7_108.94       0.8082          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_263.93     1_200.89     7_464.82       0.8082          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32 (self)                               6_263.93     4_341.70    10_605.63       0.7584          1.0236            1.0218         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_425.27     1_266.89    10_692.16       0.8661          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_425.27     1_411.15    10_836.43       0.8662          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_425.27     2_021.83    11_447.11       0.8662          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               9_425.27     7_139.65    16_564.92       0.8373          1.0103            1.0089         4.36
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
Exhaustive (query)                                        68.65     1_320.79     1_389.44       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.65     4_365.34     4_433.99       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_969.85     1_019.60     6_989.45       0.2330          1.2135            1.2091         2.26
Exhaustive-OPQ-m16 (self)                              5_969.85     4_769.15    10_739.00       0.1878          1.2980            1.2982         2.26
Exhaustive-OPQ-m32 (query)                             8_049.41     1_873.98     9_923.39       0.3196          1.1507            1.1475         3.03
Exhaustive-OPQ-m32 (self)                              8_049.41     7_614.43    15_663.84       0.2589          1.2170            1.2154         3.03
Exhaustive-OPQ-m64 (query)                            12_982.05     4_006.53    16_988.58       0.4324          1.0942            1.0922         4.55
Exhaustive-OPQ-m64 (self)                             12_982.05    14_483.45    27_465.49       0.3620          1.1419            1.1397         4.55
Exhaustive-OPQ-m128 (query)                           19_505.18     8_271.39    27_776.57       0.5702          1.0490            1.0477         7.61
Exhaustive-OPQ-m128 (self)                            19_505.18    29_618.73    49_123.91       0.5002          1.0775            1.0756         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_651.92       606.99     7_258.91       0.5393          1.0565            1.0552         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_651.92       763.17     7_415.08       0.5393          1.0565            1.0552         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_651.92       916.84     7_568.75       0.5393          1.0565            1.0552         3.07
IVF-OPQ-nl158-m16 (self)                               6_651.92     4_622.94    11_274.86       0.4365          1.1018            1.0998         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_388.79       742.29     9_131.08       0.6828          1.0247            1.0235         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_388.79       968.39     9_357.18       0.6828          1.0247            1.0235         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_388.79     1_213.02     9_601.81       0.6828          1.0247            1.0235         3.84
IVF-OPQ-nl158-m32 (self)                               8_388.79     5_485.66    13_874.45       0.6083          1.0441            1.0419         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_262.94     1_069.67    14_332.62       0.7785          1.0116            1.0105         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_262.94     1_496.98    14_759.92       0.7785          1.0116            1.0105         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_262.94     1_934.25    15_197.19       0.7785          1.0116            1.0105         5.36
IVF-OPQ-nl158-m64 (self)                              13_262.94     7_855.97    21_118.91       0.7303          1.0199            1.0179         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_657.53     1_616.30    21_273.83       0.8377          1.0062            1.0051         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_657.53     2_371.46    22_028.99       0.8377          1.0062            1.0051         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_657.53     3_076.67    22_734.20       0.8377          1.0062            1.0051         8.42
IVF-OPQ-nl158-m128 (self)                             19_657.53    11_783.75    31_441.28       0.8098          1.0099            1.0081         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_431.55       743.28     7_174.84       0.5484          1.0539            1.0527         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_431.55       813.15     7_244.70       0.5484          1.0539            1.0527         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_431.55     1_035.99     7_467.55       0.5484          1.0539            1.0527         3.20
IVF-OPQ-nl223-m16 (self)                               6_431.55     4_874.26    11_305.81       0.4478          1.0972            1.0954         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_413.95       927.17     9_341.11       0.6905          1.0234            1.0223         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_413.95     1_063.19     9_477.14       0.6905          1.0234            1.0223         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_413.95     1_391.98     9_805.93       0.6905          1.0234            1.0223         3.96
IVF-OPQ-nl223-m32 (self)                               8_413.95     6_061.48    14_475.43       0.6180          1.0417            1.0398         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_996.67     1_400.12    14_396.80       0.7820          1.0111            1.0100         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_996.67     1_660.84    14_657.51       0.7820          1.0111            1.0100         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_996.67     2_303.16    15_299.83       0.7820          1.0111            1.0100         5.49
IVF-OPQ-nl223-m64 (self)                              12_996.67     9_034.61    22_031.29       0.7362          1.0189            1.0172         5.49
IVF-OPQ-nl223-m128-np11 (query)                       19_482.06     2_171.17    21_653.24       0.8405          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np14 (query)                       19_482.06     2_612.00    22_094.07       0.8405          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np21 (query)                       19_482.06     3_689.65    23_171.71       0.8405          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128 (self)                             19_482.06    13_806.86    33_288.92       0.8128          1.0095            1.0079         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_391.02       847.39     7_238.41       0.5509          1.0532            1.0517         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_391.02       907.78     7_298.80       0.5509          1.0532            1.0517         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_391.02     1_158.04     7_549.06       0.5509          1.0532            1.0517         3.88
IVF-OPQ-nl316-m16 (self)                               6_391.02     5_272.77    11_663.79       0.4523          1.0953            1.0936         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_922.75     1_105.57    10_028.32       0.6931          1.0231            1.0220         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_922.75     1_194.81    10_117.56       0.6931          1.0231            1.0220         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_922.75     1_564.02    10_486.77       0.6931          1.0231            1.0220         4.65
IVF-OPQ-nl316-m32 (self)                               8_922.75     6_634.95    15_557.70       0.6208          1.0410            1.0392         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_227.90     1_731.46    14_959.36       0.7852          1.0109            1.0098         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_227.90     1_889.84    15_117.74       0.7852          1.0109            1.0098         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_227.90     2_584.32    15_812.22       0.7852          1.0109            1.0098         6.17
IVF-OPQ-nl316-m64 (self)                              13_227.90    10_029.74    23_257.64       0.7393          1.0184            1.0168         6.17
IVF-OPQ-nl316-m128-np15 (query)                       19_772.49     2_706.93    22_479.42       0.8419          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np17 (query)                       19_772.49     2_996.82    22_769.31       0.8419          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np25 (query)                       19_772.49     4_200.01    23_972.50       0.8419          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128 (self)                             19_772.49    15_570.93    35_343.42       0.8148          1.0092            1.0077         9.23
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
Exhaustive (query)                                        99.31     1_813.84     1_913.14       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.31     5_958.91     6_058.22       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_161.68     1_491.65    10_653.32       0.2295          1.2024            1.1993         3.76
Exhaustive-OPQ-m16 (self)                              9_161.68     8_144.64    17_306.31       0.1867          1.2970            1.2974         3.76
Exhaustive-OPQ-m32 (query)                            11_352.90     2_329.45    13_682.35       0.3131          1.1445            1.1408         4.53
Exhaustive-OPQ-m32 (self)                             11_352.90    10_895.05    22_247.95       0.2579          1.2166            1.2142         4.53
Exhaustive-OPQ-m64 (query)                            15_674.29     4_416.98    20_091.27       0.4088          1.0974            1.0946         6.05
Exhaustive-OPQ-m64 (self)                             15_674.29    17_633.72    33_308.01       0.3476          1.1497            1.1472         6.05
Exhaustive-OPQ-m128 (query)                           24_001.39     8_688.91    32_690.30       0.5284          1.0565            1.0547         9.11
Exhaustive-OPQ-m128 (self)                            24_001.39    31_934.37    55_935.76       0.4657          1.0908            1.0887         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_217.02     1_140.48    10_357.50       0.5291          1.0552            1.0541         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_217.02     1_328.12    10_545.13       0.5291          1.0552            1.0541         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_217.02     1_497.90    10_714.92       0.5291          1.0552            1.0541         4.98
IVF-OPQ-nl158-m16 (self)                               9_217.02     8_158.52    17_375.54       0.4254          1.1060            1.1037         4.98
IVF-OPQ-nl158-m32-np7 (query)                         11_362.69     1_316.45    12_679.14       0.6771          1.0241            1.0229         5.74
IVF-OPQ-nl158-m32-np12 (query)                        11_362.69     1_610.40    12_973.08       0.6771          1.0241            1.0229         5.74
IVF-OPQ-nl158-m32-np17 (query)                        11_362.69     1_903.58    13_266.27       0.6771          1.0241            1.0229         5.74
IVF-OPQ-nl158-m32 (self)                              11_362.69     9_537.33    20_900.02       0.6006          1.0458            1.0435         5.74
IVF-OPQ-nl158-m64-np7 (query)                         15_612.63     1_643.88    17_256.51       0.7716          1.0115            1.0103         7.27
IVF-OPQ-nl158-m64-np12 (query)                        15_612.63     2_130.55    17_743.18       0.7716          1.0115            1.0103         7.27
IVF-OPQ-nl158-m64-np17 (query)                        15_612.63     2_614.29    18_226.92       0.7716          1.0115            1.0103         7.27
IVF-OPQ-nl158-m64 (self)                              15_612.63    11_914.03    27_526.66       0.7237          1.0209            1.0186         7.27
IVF-OPQ-nl158-m128-np7 (query)                        23_771.17     2_437.36    26_208.53       0.8309          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np12 (query)                       23_771.17     3_392.21    27_163.38       0.8309          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np17 (query)                       23_771.17     4_367.04    28_138.21       0.8309          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128 (self)                             23_771.17    17_705.26    41_476.42       0.8029          1.0106            1.0086        10.32
IVF-OPQ-nl223-m16-np11 (query)                         9_725.42     1_286.46    11_011.88       0.5392          1.0524            1.0511         5.17
IVF-OPQ-nl223-m16-np14 (query)                         9_725.42     1_395.23    11_120.66       0.5392          1.0524            1.0511         5.17
IVF-OPQ-nl223-m16-np21 (query)                         9_725.42     1_648.48    11_373.90       0.5392          1.0524            1.0511         5.17
IVF-OPQ-nl223-m16 (self)                               9_725.42     8_666.92    18_392.34       0.4400          1.0998            1.0979         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_093.26     1_553.21    13_646.47       0.6841          1.0228            1.0215         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_093.26     1_728.32    13_821.58       0.6841          1.0228            1.0215         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_093.26     2_153.11    14_246.37       0.6841          1.0228            1.0215         5.93
IVF-OPQ-nl223-m32 (self)                              12_093.26    10_339.88    22_433.14       0.6106          1.0431            1.0412         5.93
IVF-OPQ-nl223-m64-np11 (query)                        16_188.49     2_042.27    18_230.76       0.7780          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np14 (query)                        16_188.49     2_341.18    18_529.67       0.7780          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np21 (query)                        16_188.49     3_050.26    19_238.75       0.7780          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64 (self)                              16_188.49    13_341.33    29_529.82       0.7321          1.0195            1.0177         7.46
IVF-OPQ-nl223-m128-np11 (query)                       24_611.93     3_188.01    27_799.94       0.8360          1.0059            1.0049        10.51
IVF-OPQ-nl223-m128-np14 (query)                       24_611.93     3_771.66    28_383.59       0.8360          1.0059            1.0049        10.51
IVF-OPQ-nl223-m128-np21 (query)                       24_611.93     5_134.56    29_746.48       0.8360          1.0059            1.0049        10.51
IVF-OPQ-nl223-m128 (self)                             24_611.93    20_301.52    44_913.45       0.8077          1.0100            1.0083        10.51
IVF-OPQ-nl316-m16-np15 (query)                         9_919.33     1_432.89    11_352.22       0.5428          1.0513            1.0501         6.19
IVF-OPQ-nl316-m16-np17 (query)                         9_919.33     1_500.65    11_419.98       0.5428          1.0513            1.0501         6.19
IVF-OPQ-nl316-m16-np25 (query)                         9_919.33     1_790.36    11_709.70       0.5428          1.0513            1.0501         6.19
IVF-OPQ-nl316-m16 (self)                               9_919.33     9_131.61    19_050.94       0.4438          1.0981            1.0963         6.19
IVF-OPQ-nl316-m32-np15 (query)                        12_046.31     1_792.83    13_839.13       0.6871          1.0224            1.0212         6.95
IVF-OPQ-nl316-m32-np17 (query)                        12_046.31     1_917.95    13_964.26       0.6871          1.0224            1.0212         6.95
IVF-OPQ-nl316-m32-np25 (query)                        12_046.31     2_373.32    14_419.63       0.6871          1.0224            1.0212         6.95
IVF-OPQ-nl316-m32 (self)                              12_046.31    10_975.80    23_022.11       0.6133          1.0425            1.0407         6.95
IVF-OPQ-nl316-m64-np15 (query)                        16_378.81     2_408.04    18_786.85       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np17 (query)                        16_378.81     2_615.37    18_994.18       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np25 (query)                        16_378.81     3_379.06    19_757.87       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64 (self)                              16_378.81    14_491.23    30_870.04       0.7332          1.0192            1.0176         8.48
IVF-OPQ-nl316-m128-np15 (query)                       24_793.76     3_912.80    28_706.56       0.8368          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128-np17 (query)                       24_793.76     4_590.30    29_384.06       0.8368          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128-np25 (query)                       24_793.76     5_830.26    30_624.02       0.8368          1.0058            1.0049        11.53
IVF-OPQ-nl316-m128 (self)                             24_793.76    22_733.49    47_527.25       0.8083          1.0097            1.0082        11.53
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
Exhaustive (query)                                        33.23       714.86       748.09       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.23     2_418.61     2_451.84       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_611.55       721.91     4_333.45       0.7904          1.0825            1.0692         1.26
Exhaustive-OPQ-m16 (self)                              3_611.55     2_692.34     6_303.89       0.7232          1.1503            1.1257         1.26
Exhaustive-OPQ-m32 (query)                             6_022.62     1_612.52     7_635.14       0.8306          1.0534            1.0420         2.03
Exhaustive-OPQ-m32 (self)                              6_022.62     5_581.66    11_604.29       0.7762          1.0975            1.0768         2.03
Exhaustive-OPQ-m64 (query)                             9_191.94     3_683.66    12_875.61       0.8559          1.0400            1.0294         3.55
Exhaustive-OPQ-m64 (self)                              9_191.94    12_612.60    21_804.54       0.8088          1.0725            1.0535         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_808.95       286.44     4_095.38       0.8906          1.0204            1.0161         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_808.95       419.97     4_228.92       0.8913          1.0200            1.0159         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_808.95       555.93     4_364.87       0.8913          1.0200            1.0159         1.67
IVF-OPQ-nl158-m16 (self)                               3_808.95     2_205.95     6_014.90       0.8484          1.0400            1.0321         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_129.80       468.37     6_598.17       0.9110          1.0133            1.0098         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_129.80       752.06     6_881.86       0.9120          1.0129            1.0096         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_129.80     1_015.44     7_145.23       0.9120          1.0129            1.0096         2.43
IVF-OPQ-nl158-m32 (self)                               6_129.80     3_716.83     9_846.62       0.8774          1.0256            1.0195         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_563.41       790.68    10_354.09       0.9240          1.0098            1.0067         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_563.41     1_314.32    10_877.72       0.9249          1.0094            1.0065         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_563.41     1_792.68    11_356.09       0.9249          1.0094            1.0065         3.96
IVF-OPQ-nl158-m64 (self)                               9_563.41     6_322.38    15_885.79       0.8965          1.0184            1.0132         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_997.39       371.93     4_369.32       0.8978          1.0177            1.0135         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_997.39       454.44     4_451.83       0.8979          1.0177            1.0135         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_997.39       635.61     4_633.00       0.8979          1.0177            1.0135         1.73
IVF-OPQ-nl223-m16 (self)                               3_997.39     2_487.54     6_484.93       0.8578          1.0353            1.0273         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_241.40       622.50     6_863.90       0.9158          1.0118            1.0086         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_241.40       763.30     7_004.70       0.9160          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_241.40     1_128.39     7_369.79       0.9160          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32 (self)                               6_241.40     4_021.92    10_263.32       0.8833          1.0231            1.0174         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_694.65     1_053.96    10_748.60       0.9271          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_694.65     1_312.04    11_006.69       0.9272          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_694.65     1_914.06    11_608.71       0.9273          1.0089            1.0058         4.02
IVF-OPQ-nl223-m64 (self)                               9_694.65     6_741.75    16_436.40       0.8991          1.0176            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_250.78       473.97     4_724.76       0.9016          1.0166            1.0127         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_250.78       507.72     4_758.51       0.9017          1.0166            1.0127         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_250.78       706.68     4_957.47       0.9017          1.0166            1.0127         2.07
IVF-OPQ-nl316-m16 (self)                               4_250.78     2_704.70     6_955.48       0.8625          1.0334            1.0248         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_524.36       807.05     7_331.42       0.9168          1.0114            1.0083         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_524.36       882.19     7_406.55       0.9169          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_524.36     1_259.04     7_783.41       0.9169          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32 (self)                               6_524.36     4_513.67    11_038.03       0.8843          1.0230            1.0168         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_839.47     1_312.42    11_151.88       0.9287          1.0086            1.0057         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_839.47     1_471.40    11_310.87       0.9288          1.0086            1.0057         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_839.47     2_125.96    11_965.42       0.9288          1.0085            1.0057         4.36
IVF-OPQ-nl316-m64 (self)                               9_839.47     7_468.48    17_307.95       0.9010          1.0169            1.0116         4.36
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
Exhaustive (query)                                        68.30     1_330.53     1_398.83       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.30     4_379.02     4_447.32       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_030.49     1_017.50     7_047.99       0.7547          1.1138            1.0980         2.26
Exhaustive-OPQ-m16 (self)                              6_030.49     4_789.98    10_820.47       0.6781          1.2042            1.1744         2.26
Exhaustive-OPQ-m32 (query)                             8_261.48     1_885.78    10_147.27       0.8055          1.0695            1.0574         3.03
Exhaustive-OPQ-m32 (self)                              8_261.48     7_543.64    15_805.12       0.7452          1.1247            1.1020         3.03
Exhaustive-OPQ-m64 (query)                            12_903.26     3_991.87    16_895.13       0.8415          1.0455            1.0362         4.55
Exhaustive-OPQ-m64 (self)                             12_903.26    14_483.93    27_387.19       0.7920          1.0816            1.0653         4.55
Exhaustive-OPQ-m128 (query)                           20_100.98     8_319.53    28_420.51       0.9196          1.0107            1.0070         7.61
Exhaustive-OPQ-m128 (self)                            20_100.98    30_337.62    50_438.61       0.8928          1.0194            1.0140         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_121.35       618.87     6_740.22       0.8883          1.0226            1.0174         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_121.35       781.99     6_903.34       0.8886          1.0224            1.0173         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_121.35       954.24     7_075.59       0.8886          1.0224            1.0173         3.07
IVF-OPQ-nl158-m16 (self)                               6_121.35     4_567.72    10_689.07       0.8435          1.0458            1.0329         3.07
IVF-OPQ-nl158-m32-np7 (query)                          7_783.34       756.56     8_539.91       0.9009          1.0172            1.0129         3.84
IVF-OPQ-nl158-m32-np12 (query)                         7_783.34     1_031.20     8_814.54       0.9013          1.0170            1.0127         3.84
IVF-OPQ-nl158-m32-np17 (query)                         7_783.34     1_311.16     9_094.50       0.9013          1.0170            1.0127         3.84
IVF-OPQ-nl158-m32 (self)                               7_783.34     5_741.62    13_524.96       0.8619          1.0348            1.0247         3.84
IVF-OPQ-nl158-m64-np7 (query)                         11_840.08     1_135.41    12_975.49       0.9106          1.0145            1.0100         5.36
IVF-OPQ-nl158-m64-np12 (query)                        11_840.08     1_681.43    13_521.51       0.9111          1.0143            1.0099         5.36
IVF-OPQ-nl158-m64-np17 (query)                        11_840.08     2_224.01    14_064.09       0.9111          1.0143            1.0099         5.36
IVF-OPQ-nl158-m64 (self)                              11_840.08     8_684.65    20_524.73       0.8750          1.0285            1.0198         5.36
IVF-OPQ-nl158-m128-np7 (query)                        17_413.78     1_775.68    19_189.46       0.9613          1.0029            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                       17_413.78     2_729.90    20_143.68       0.9620          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                       17_413.78     3_695.13    21_108.90       0.9621          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                             17_413.78    13_701.86    31_115.64       0.9450          1.0056            1.0016         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_214.70       720.37     6_935.07       0.8977          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_214.70       809.84     7_024.54       0.8978          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_214.70     1_032.45     7_247.15       0.8978          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16 (self)                               6_214.70     4_814.68    11_029.38       0.8552          1.0393            1.0271         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_029.30       933.07     8_962.37       0.9084          1.0151            1.0108         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_029.30     1_082.34     9_111.63       0.9084          1.0151            1.0107         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_029.30     1_442.40     9_471.70       0.9084          1.0151            1.0107         3.96
IVF-OPQ-nl223-m32 (self)                               8_029.30     6_169.63    14_198.93       0.8703          1.0312            1.0211         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_208.23     1_452.45    13_660.68       0.9168          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_208.23     1_738.97    13_947.20       0.9169          1.0122            1.0082         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_208.23     2_506.72    14_714.95       0.9169          1.0122            1.0082         5.49
IVF-OPQ-nl223-m64 (self)                              12_208.23     9_555.50    21_763.73       0.8832          1.0244            1.0168         5.49
IVF-OPQ-nl223-m128-np11 (query)                       17_887.51     2_277.03    20_164.54       0.9658          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                       17_887.51     2_791.86    20_679.37       0.9659          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                       17_887.51     3_987.81    21_875.32       0.9659          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                             17_887.51    14_733.17    32_620.68       0.9484          1.0050            1.0009         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_279.60       834.21     7_113.81       0.9048          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_279.60       890.64     7_170.24       0.9048          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_279.60     1_143.17     7_422.77       0.9048          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16 (self)                               6_279.60     5_234.15    11_513.75       0.8654          1.0335            1.0228         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_597.75     1_100.66     9_698.41       0.9145          1.0128            1.0091         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_597.75     1_265.21     9_862.96       0.9145          1.0128            1.0091         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_597.75     1_586.78    10_184.53       0.9145          1.0128            1.0091         4.65
IVF-OPQ-nl316-m32 (self)                               8_597.75     6_656.58    15_254.33       0.8790          1.0266            1.0179         4.65
IVF-OPQ-nl316-m64-np15 (query)                        12_382.41     1_755.69    14_138.11       0.9207          1.0109            1.0075         6.17
IVF-OPQ-nl316-m64-np17 (query)                        12_382.41     1_949.11    14_331.52       0.9207          1.0109            1.0074         6.17
IVF-OPQ-nl316-m64-np25 (query)                        12_382.41     2_681.68    15_064.10       0.9207          1.0109            1.0074         6.17
IVF-OPQ-nl316-m64 (self)                              12_382.41    10_244.27    22_626.68       0.8886          1.0221            1.0150         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_494.03     2_760.74    21_254.77       0.9687          1.0018            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_494.03     3_289.07    21_783.10       0.9687          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_494.03     4_387.30    22_881.34       0.9688          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                             18_494.03    15_935.60    34_429.63       0.9518          1.0045            1.0005         9.23
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
Exhaustive (query)                                        99.22     1_816.63     1_915.85       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.22     6_012.96     6_112.18       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_166.29     1_497.59    10_663.88       0.7379          1.1311            1.1122         3.76
Exhaustive-OPQ-m16 (self)                              9_166.29     8_200.13    17_366.42       0.6590          1.2298            1.1965         3.76
Exhaustive-OPQ-m32 (query)                            11_324.43     2_367.00    13_691.43       0.8491          1.0412            1.0320         4.53
Exhaustive-OPQ-m32 (self)                             11_324.43    11_072.99    22_397.42       0.8004          1.0715            1.0582         4.53
Exhaustive-OPQ-m64 (query)                            15_753.14     4_395.11    20_148.25       0.8798          1.0254            1.0185         6.05
Exhaustive-OPQ-m64 (self)                             15_753.14    17_595.94    33_349.08       0.8412          1.0442            1.0343         6.05
Exhaustive-OPQ-m128 (query)                           24_401.00     9_201.01    33_602.01       0.9047          1.0148            1.0107         9.11
Exhaustive-OPQ-m128 (self)                            24_401.00    31_821.39    56_222.39       0.8745          1.0262            1.0199         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_608.39     1_166.83    10_775.23       0.8916          1.0223            1.0163         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_608.39     1_384.34    10_992.73       0.8918          1.0223            1.0162         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_608.39     1_588.71    11_197.10       0.8918          1.0223            1.0162         4.98
IVF-OPQ-nl158-m16 (self)                               9_608.39     8_421.62    18_030.02       0.8470          1.0438            1.0303         4.98
IVF-OPQ-nl158-m32-np7 (query)                         11_696.42     1_358.65    13_055.07       0.9371          1.0085            1.0035         5.74
IVF-OPQ-nl158-m32-np12 (query)                        11_696.42     1_706.99    13_403.41       0.9373          1.0084            1.0034         5.74
IVF-OPQ-nl158-m32-np17 (query)                        11_696.42     2_059.27    13_755.70       0.9373          1.0084            1.0034         5.74
IVF-OPQ-nl158-m32 (self)                              11_696.42     9_990.22    21_686.65       0.9080          1.0182            1.0074         5.74
IVF-OPQ-nl158-m64-np7 (query)                         15_961.27     1_746.90    17_708.17       0.9511          1.0052            1.0013         7.27
IVF-OPQ-nl158-m64-np12 (query)                        15_961.27     2_367.84    18_329.11       0.9513          1.0051            1.0012         7.27
IVF-OPQ-nl158-m64-np17 (query)                        15_961.27     2_961.68    18_922.96       0.9513          1.0051            1.0012         7.27
IVF-OPQ-nl158-m64 (self)                              15_961.27    13_230.21    29_191.48       0.9268          1.0116            1.0036         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_005.38     2_628.07    27_633.44       0.9606          1.0033            1.0001        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_005.38     3_845.57    28_850.94       0.9609          1.0032            1.0001        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_005.38     5_029.10    30_034.48       0.9609          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                             25_005.38    20_058.95    45_064.33       0.9396          1.0077            1.0017        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_295.27     1_316.00    11_611.26       0.8996          1.0188            1.0136         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_295.27     1_420.80    11_716.07       0.8997          1.0188            1.0136         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_295.27     1_704.60    11_999.86       0.8997          1.0188            1.0136         5.17
IVF-OPQ-nl223-m16 (self)                              10_295.27     8_801.95    19_097.21       0.8592          1.0363            1.0252         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_391.97     1_587.62    13_979.58       0.9423          1.0071            1.0026         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_391.97     1_790.27    14_182.24       0.9424          1.0071            1.0026         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_391.97     2_242.59    14_634.55       0.9424          1.0071            1.0026         5.93
IVF-OPQ-nl223-m32 (self)                              12_391.97    10_667.95    23_059.91       0.9159          1.0147            1.0059         5.93
IVF-OPQ-nl223-m64-np11 (query)                        16_749.01     2_108.63    18_857.64       0.9544          1.0046            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                        16_749.01     2_433.39    19_182.39       0.9544          1.0046            1.0008         7.46
IVF-OPQ-nl223-m64-np21 (query)                        16_749.01     3_223.56    19_972.57       0.9545          1.0046            1.0008         7.46
IVF-OPQ-nl223-m64 (self)                              16_749.01    13_862.28    30_611.29       0.9326          1.0098            1.0027         7.46
IVF-OPQ-nl223-m128-np11 (query)                       25_221.63     3_326.22    28_547.86       0.9638          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                       25_221.63     3_983.14    29_204.77       0.9639          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                       25_221.63     5_519.75    30_741.39       0.9639          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                             25_221.63    21_738.71    46_960.34       0.9434          1.0068            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_589.55     1_457.57    12_047.12       0.9044          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_589.55     1_529.81    12_119.36       0.9044          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_589.55     1_857.19    12_446.74       0.9044          1.0171            1.0119         6.19
IVF-OPQ-nl316-m16 (self)                              10_589.55     9_284.22    19_873.77       0.8645          1.0338            1.0231         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_594.69     1_844.06    15_438.75       0.9442          1.0064            1.0023         6.95
IVF-OPQ-nl316-m32-np17 (query)                        13_594.69     1_939.98    15_534.67       0.9442          1.0064            1.0023         6.95
IVF-OPQ-nl316-m32-np25 (query)                        13_594.69     2_431.75    16_026.44       0.9442          1.0064            1.0023         6.95
IVF-OPQ-nl316-m32 (self)                              13_594.69    11_315.36    24_910.05       0.9193          1.0136            1.0052         6.95
IVF-OPQ-nl316-m64-np15 (query)                        19_387.37     2_492.93    21_880.30       0.9567          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np17 (query)                        19_387.37     2_662.01    22_049.38       0.9567          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np25 (query)                        19_387.37     3_500.38    22_887.76       0.9567          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64 (self)                              19_387.37    14_921.23    34_308.60       0.9358          1.0089            1.0023         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_274.09     4_030.18    30_304.27       0.9658          1.0024            1.0000        11.53
IVF-OPQ-nl316-m128-np17 (query)                       26_274.09     4_428.64    30_702.72       0.9658          1.0024            1.0000        11.53
IVF-OPQ-nl316-m128-np25 (query)                       26_274.09     6_103.06    32_377.15       0.9658          1.0024            1.0000        11.53
IVF-OPQ-nl316-m128 (self)                             26_274.09    23_685.23    49_959.31       0.9462          1.0061            1.0009        11.53
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
Exhaustive (query)                                        68.87     1_235.39     1_304.26       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_384.10       125.43     1_509.53       0.3468          1.0755            1.0751         2.24
IVFPQ-m64-nl111-np1                                    2_233.03       224.97     2_458.00       0.4500          1.0504            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_536.16       139.61     1_675.78       0.3219          1.2732            1.0774         4.72
IVFPQ-m32-nl111-np2                                    1_384.10       172.95     1_557.05       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np2                                    2_233.03       308.04     2_541.07       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_536.16       180.06     1_716.23       0.3477          1.0764            1.0749         4.72
IVFPQ-m32-nl111-np4                                    1_384.10       263.81     1_647.90       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np4                                    2_233.03       479.62     2_712.65       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_536.16       274.37     1_810.54       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np5                                    1_384.10       309.12     1_693.22       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np5                                    2_233.03       563.96     2_796.98       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_536.16       321.73     1_857.89       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np8                                    1_384.10       447.40     1_831.50       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np8                                    2_233.03       825.67     3_058.70       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_536.16       477.00     2_013.17       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np10                                   1_384.10       531.64     1_915.74       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np10                                   2_233.03       989.23     3_222.25       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_536.16       575.84     2_112.01       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl158-np1                                    1_475.88       122.51     1_598.39       0.3490          1.0730            1.0736         2.34
IVFPQ-m64-nl158-np1                                    2_274.20       216.15     2_490.35       0.4593          1.0469            1.0428         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_653.89       130.59     1_784.49       0.3165          1.1950            1.0771         4.82
IVFPQ-m32-nl158-np2                                    1_475.88       169.47     1_645.34       0.3526          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np2                                    2_274.20       309.35     2_583.55       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_653.89       183.85     1_837.74       0.3520          1.0726            1.0731         4.82
IVFPQ-m32-nl158-np4                                    1_475.88       257.25     1_733.13       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np4                                    2_274.20       464.32     2_738.52       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_653.89       272.82     1_926.72       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np7                                    1_475.88       389.35     1_865.23       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np7                                    2_274.20       710.20     2_984.40       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_653.89       408.90     2_062.79       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np8                                    1_475.88       430.52     1_906.40       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np8                                    2_274.20       792.38     3_066.58       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_653.89       453.88     2_107.77       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np12                                   1_475.88       611.97     2_087.85       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np12                                   2_274.20     1_132.31     3_406.51       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_653.89       637.89     2_291.78       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl223-np1                                    1_503.73       101.69     1_605.42       0.3524          1.0705            1.0702         2.46
IVFPQ-m64-nl223-np1                                    2_304.98       158.93     2_463.91       0.4383          1.0504            1.0448         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_695.88       112.66     1_808.53       0.3306          1.1705            1.0732         4.94
IVFPQ-m32-nl223-np2                                    1_503.73       150.83     1_654.56       0.3660          1.0664            1.0666         2.46
IVFPQ-m64-nl223-np2                                    2_304.98       253.96     2_558.94       0.4668          1.0444            1.0401         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_695.88       170.97     1_866.84       0.3633          1.0686            1.0686         4.94
IVFPQ-m32-nl223-np4                                    1_503.73       247.49     1_751.22       0.3690          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np4                                    2_304.98       437.16     2_742.14       0.4762          1.0430            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_695.88       269.57     1_965.45       0.3673          1.0667            1.0665         4.94
IVFPQ-m32-nl223-np8                                    1_503.73       429.63     1_933.36       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np8                                    2_304.98       775.88     3_080.86       0.4775          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_695.88       457.16     2_153.03       0.3692          1.0656            1.0657         4.94
IVFPQ-m32-nl223-np11                                   1_503.73       570.20     2_073.93       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np11                                   2_304.98     1_031.84     3_336.82       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_695.88       598.57     2_294.44       0.3693          1.0656            1.0657         4.94
IVFPQ-m32-nl223-np14                                   1_503.73       707.33     2_211.06       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np14                                   2_304.98     1_280.40     3_585.37       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_695.88       740.81     2_436.68       0.3693          1.0656            1.0657         4.94
IVFPQ-m32-nl316-np1                                    1_590.55       100.63     1_691.18       0.3529          1.0694            1.0685         2.65
IVFPQ-m64-nl316-np1                                    2_397.21       145.03     2_542.24       0.4277          1.0517            1.0458         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_802.13       109.37     1_911.50       0.3254          1.1561            1.0727         5.13
IVFPQ-m32-nl316-np2                                    1_590.55       147.85     1_738.40       0.3727          1.0626            1.0636         2.65
IVFPQ-m64-nl316-np2                                    2_397.21       238.64     2_635.85       0.4696          1.0426            1.0391         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_802.13       164.42     1_966.55       0.3692          1.0659            1.0659         5.13
IVFPQ-m32-nl316-np4                                    1_590.55       242.83     1_833.39       0.3787          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np4                                    2_397.21       417.00     2_814.21       0.4855          1.0401            1.0368         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_802.13       265.62     2_067.75       0.3751          1.0632            1.0637         5.13
IVFPQ-m32-nl316-np8                                    1_590.55       429.62     2_020.17       0.3797          1.0610            1.0619         2.65
IVFPQ-m64-nl316-np8                                    2_397.21       761.21     3_158.42       0.4890          1.0397            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_802.13       456.12     2_258.25       0.3789          1.0615            1.0622         5.13
IVFPQ-m32-nl316-np15                                   1_590.55       754.14     2_344.70       0.3797          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np15                                   2_397.21     1_344.32     3_741.52       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_802.13       794.57     2_596.70       0.3797          1.0610            1.0618         5.13
IVFPQ-m32-nl316-np17                                   1_590.55       850.92     2_441.47       0.3798          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np17                                   2_397.21     1_507.96     3_905.16       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_802.13       886.07     2_688.20       0.3798          1.0610            1.0618         5.13
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
Exhaustive (query)                                        68.87     1_235.39     1_304.26       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_661.19       129.54     1_790.72       0.3175          1.1920            1.0770         4.82
SOARPQ-near-np2                                        1_661.19       178.55     1_839.73       0.3523          1.0721            1.0731         4.82
SOARPQ-near-np4                                        1_661.19       268.79     1_929.97       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np7                                        1_661.19       410.37     2_071.56       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np8                                        1_661.19       465.35     2_126.54       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np12                                       1_661.19       650.21     2_311.40       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np1                                    1_673.49       133.42     1_806.91       0.3167          1.1944            1.0771         4.82
SOARPQ-shift0.3-np2                                    1_673.49       178.95     1_852.44       0.3521          1.0725            1.0731         4.82
SOARPQ-shift0.3-np4                                    1_673.49       269.33     1_942.82       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np7                                    1_673.49       411.08     2_084.57       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np8                                    1_673.49       463.43     2_136.92       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np12                                   1_673.49       653.83     2_327.32       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np1                                    1_662.97       129.41     1_792.38       0.3164          1.1957            1.0771         4.82
SOARPQ-shift0.7-np2                                    1_662.97       180.90     1_843.87       0.3519          1.0730            1.0731         4.82
SOARPQ-shift0.7-np4                                    1_662.97       269.28     1_932.25       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np7                                    1_662.97       412.71     2_075.68       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np8                                    1_662.97       460.44     2_123.40       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np12                                   1_662.97       652.31     2_315.28       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np1                                       1_683.90       129.30     1_813.20       0.3179          1.1930            1.0769         4.82
SOARPQ-orth1-np2                                       1_683.90       178.73     1_862.63       0.3525          1.0718            1.0730         4.82
SOARPQ-orth1-np4                                       1_683.90       270.09     1_953.98       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np7                                       1_683.90       412.81     2_096.71       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np8                                       1_683.90       461.72     2_145.62       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np12                                      1_683.90       666.77     2_350.67       0.3527          1.0715            1.0730         4.82
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
Exhaustive (query)                                        67.62     1_314.86     1_382.49       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_324.21       118.38     1_442.60       0.4790          1.0776            1.0739         2.24
IVFPQ-m64-nl111-np1                                    2_107.46       210.43     2_317.89       0.6165          1.0395            1.0353         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_491.68       126.05     1_617.73       0.4659          1.0980            1.0769         4.72
IVFPQ-m32-nl111-np2                                    1_324.21       164.58     1_488.80       0.4838          1.0757            1.0732         2.24
IVFPQ-m64-nl111-np2                                    2_107.46       295.86     2_403.32       0.6244          1.0367            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_491.68       175.67     1_667.35       0.4828          1.0776            1.0736         4.72
IVFPQ-m32-nl111-np4                                    1_324.21       255.08     1_579.30       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np4                                    2_107.46       463.03     2_570.49       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_491.68       274.51     1_766.19       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np5                                    1_324.21       299.84     1_624.05       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np5                                    2_107.46       546.60     2_654.06       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_491.68       328.00     1_819.68       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np8                                    1_324.21       439.95     1_764.17       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np8                                    2_107.46       805.52     2_912.98       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_491.68       479.91     1_971.59       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np10                                   1_324.21       533.11     1_857.32       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np10                                   2_107.46       976.64     3_084.10       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_491.68       590.89     2_082.56       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl158-np1                                    1_488.75       121.24     1_610.00       0.4830          1.0756            1.0721         2.34
IVFPQ-m64-nl158-np1                                    2_294.56       210.95     2_505.52       0.6147          1.0403            1.0347         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_687.40       125.95     1_813.36       0.4858          1.0778            1.0727         4.82
IVFPQ-m32-nl158-np2                                    1_488.75       165.08     1_653.83       0.4914          1.0723            1.0710         2.34
IVFPQ-m64-nl158-np2                                    2_294.56       295.24     2_589.80       0.6281          1.0356            1.0337         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_687.40       174.70     1_862.10       0.4912          1.0727            1.0713         4.82
IVFPQ-m32-nl158-np4                                    1_488.75       253.74     1_742.50       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np4                                    2_294.56       475.19     2_769.76       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_687.40       267.04     1_954.44       0.4919          1.0721            1.0709         4.82
IVFPQ-m32-nl158-np7                                    1_488.75       391.28     1_880.03       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np7                                    2_294.56       709.33     3_003.89       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_687.40       406.70     2_094.11       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np8                                    1_488.75       436.30     1_925.05       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np8                                    2_294.56       795.87     3_090.44       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_687.40       457.19     2_144.59       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np12                                   1_488.75       622.80     2_111.55       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np12                                   2_294.56     1_124.85     3_419.42       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_687.40       657.15     2_344.55       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl223-np1                                    1_509.76       101.00     1_610.76       0.3968          1.1044            1.1004         2.46
IVFPQ-m64-nl223-np1                                    2_328.31       156.17     2_484.48       0.4720          1.0736            1.0668         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_721.86       114.43     1_836.29       0.4490          1.0881            1.0842         4.94
IVFPQ-m32-nl223-np2                                    1_509.76       152.01     1_661.77       0.4582          1.0822            1.0800         2.46
IVFPQ-m64-nl223-np2                                    2_328.31       258.73     2_587.04       0.5701          1.0475            1.0435         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_721.86       171.32     1_893.18       0.4798          1.0760            1.0744         4.94
IVFPQ-m32-nl223-np4                                    1_509.76       248.11     1_757.87       0.4852          1.0739            1.0723         2.46
IVFPQ-m64-nl223-np4                                    2_328.31       434.55     2_762.86       0.6182          1.0373            1.0354         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_721.86       269.61     1_991.47       0.4900          1.0725            1.0712         4.94
IVFPQ-m32-nl223-np8                                    1_509.76       439.50     1_949.26       0.4917          1.0720            1.0707         2.46
IVFPQ-m64-nl223-np8                                    2_328.31       773.36     3_101.67       0.6323          1.0346            1.0332         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_721.86       460.35     2_182.21       0.4919          1.0720            1.0707         4.94
IVFPQ-m32-nl223-np11                                   1_509.76       579.51     2_089.27       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np11                                   2_328.31     1_041.38     3_369.69       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_721.86       602.39     2_324.25       0.4920          1.0719            1.0706         4.94
IVFPQ-m32-nl223-np14                                   1_509.76       722.47     2_232.23       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np14                                   2_328.31     1_294.59     3_622.90       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_721.86       756.75     2_478.61       0.4920          1.0719            1.0706         4.94
IVFPQ-m32-nl316-np1                                    1_615.07       101.67     1_716.73       0.3503          1.1215            1.1185         2.65
IVFPQ-m64-nl316-np1                                    2_425.29       143.18     2_568.48       0.3989          1.0931            1.0885         4.17
SOARPQ-shift0.5-m32-nl316-np1                          1_854.83       108.94     1_963.77       0.4206          1.0956            1.0942         5.13
IVFPQ-m32-nl316-np2                                    1_615.07       148.66     1_763.73       0.4250          1.0922            1.0912         2.65
IVFPQ-m64-nl316-np2                                    2_425.29       236.68     2_661.98       0.5171          1.0589            1.0562         4.17
SOARPQ-shift0.5-m32-nl316-np2                          1_854.83       164.10     2_018.93       0.4612          1.0816            1.0804         5.13
IVFPQ-m32-nl316-np4                                    1_615.07       244.34     1_859.41       0.4685          1.0785            1.0771         2.65
IVFPQ-m64-nl316-np4                                    2_425.29       418.39     2_843.68       0.5954          1.0419            1.0397         4.17
SOARPQ-shift0.5-m32-nl316-np4                          1_854.83       268.46     2_123.30       0.4828          1.0746            1.0734         5.13
IVFPQ-m32-nl316-np8                                    1_615.07       440.02     2_055.09       0.4864          1.0733            1.0718         2.65
IVFPQ-m64-nl316-np8                                    2_425.29       759.38     3_184.67       0.6309          1.0349            1.0331         4.17
SOARPQ-shift0.5-m32-nl316-np8                          1_854.83       459.38     2_314.21       0.4879          1.0731            1.0717         5.13
IVFPQ-m32-nl316-np15                                   1_615.07       761.53     2_376.60       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np15                                   2_425.29     1_342.72     3_768.02       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np15                         1_854.83       792.21     2_647.05       0.4881          1.0728            1.0715         5.13
IVFPQ-m32-nl316-np17                                   1_615.07       861.45     2_476.51       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np17                                   2_425.29     1_512.89     3_938.18       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np17                         1_854.83       890.31     2_745.15       0.4881          1.0728            1.0715         5.13
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
Exhaustive (query)                                        67.62     1_314.86     1_382.49       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_674.25       125.63     1_799.88       0.4863          1.0774            1.0725         4.82
SOARPQ-near-np2                                        1_674.25       173.89     1_848.14       0.4913          1.0726            1.0713         4.82
SOARPQ-near-np4                                        1_674.25       264.25     1_938.50       0.4920          1.0721            1.0709         4.82
SOARPQ-near-np7                                        1_674.25       405.38     2_079.63       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np8                                        1_674.25       455.36     2_129.62       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np12                                       1_674.25       655.61     2_329.86       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np1                                    1_692.20       125.89     1_818.09       0.4861          1.0776            1.0726         4.82
SOARPQ-shift0.3-np2                                    1_692.20       174.43     1_866.63       0.4912          1.0726            1.0712         4.82
SOARPQ-shift0.3-np4                                    1_692.20       276.86     1_969.06       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.3-np7                                    1_692.20       405.03     2_097.24       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np8                                    1_692.20       454.43     2_146.63       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np12                                   1_692.20       653.52     2_345.73       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np1                                    1_682.03       126.42     1_808.45       0.4855          1.0779            1.0727         4.82
SOARPQ-shift0.7-np2                                    1_682.03       174.61     1_856.64       0.4911          1.0727            1.0713         4.82
SOARPQ-shift0.7-np4                                    1_682.03       264.89     1_946.92       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.7-np7                                    1_682.03       405.98     2_088.00       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np8                                    1_682.03       456.72     2_138.75       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np12                                   1_682.03       664.86     2_346.88       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np1                                       1_797.25       126.77     1_924.02       0.4855          1.0780            1.0728         4.82
SOARPQ-orth1-np2                                       1_797.25       180.18     1_977.44       0.4911          1.0728            1.0713         4.82
SOARPQ-orth1-np4                                       1_797.25       265.72     2_062.97       0.4919          1.0722            1.0709         4.82
SOARPQ-orth1-np7                                       1_797.25       406.77     2_204.02       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np8                                       1_797.25       456.27     2_253.53       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np12                                      1_797.25       657.88     2_455.14       0.4921          1.0720            1.0708         4.82
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
Exhaustive (query)                                        67.57     1_360.44     1_428.01       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_496.15        97.97     1_594.11       0.7050          1.1847            1.0985         2.24
IVFPQ-m64-nl111-np1                                    2_350.63       159.49     2_510.12       0.7176          1.1741            1.0872         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_652.32       122.19     1_774.51       0.8135          1.0743            1.0456         4.72
IVFPQ-m32-nl111-np2                                    1_496.15       159.62     1_655.77       0.8205          1.0642            1.0404         2.24
IVFPQ-m64-nl111-np2                                    2_350.63       278.94     2_629.57       0.8435          1.0523            1.0274         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_652.32       204.76     1_857.07       0.8481          1.0449            1.0335         4.72
IVFPQ-m32-nl111-np4                                    1_496.15       286.00     1_782.15       0.8534          1.0387            1.0308         2.24
IVFPQ-m64-nl111-np4                                    2_350.63       546.59     2_897.22       0.8808          1.0260            1.0194         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_652.32       360.32     2_012.64       0.8549          1.0391            1.0309         4.72
IVFPQ-m32-nl111-np5                                    1_496.15       341.01     1_837.16       0.8551          1.0376            1.0303         2.24
IVFPQ-m64-nl111-np5                                    2_350.63       640.65     2_991.28       0.8826          1.0249            1.0190         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_652.32       435.13     2_087.45       0.8554          1.0385            1.0305         4.72
IVFPQ-m32-nl111-np8                                    1_496.15       523.74     2_019.89       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np8                                    2_350.63     1_000.67     3_351.30       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_652.32       644.11     2_296.43       0.8561          1.0374            1.0301         4.72
IVFPQ-m32-nl111-np10                                   1_496.15       646.44     2_142.59       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np10                                   2_350.63     1_243.01     3_593.64       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_652.32       778.61     2_430.93       0.8561          1.0371            1.0300         4.72
IVFPQ-m32-nl158-np1                                    1_710.48        98.10     1_808.58       0.6971          1.1918            1.1067         2.34
IVFPQ-m64-nl158-np1                                    2_504.92       146.15     2_651.07       0.7064          1.1830            1.0984         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_896.24       111.19     2_007.43       0.8151          1.0761            1.0446         4.82
IVFPQ-m32-nl158-np2                                    1_710.48       147.91     1_858.39       0.8240          1.0636            1.0374         2.34
IVFPQ-m64-nl158-np2                                    2_504.92       259.70     2_764.63       0.8414          1.0541            1.0271         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_896.24       181.22     2_077.46       0.8573          1.0417            1.0292         4.82
IVFPQ-m32-nl158-np4                                    1_710.48       259.95     1_970.43       0.8638          1.0338            1.0259         2.34
IVFPQ-m64-nl158-np4                                    2_504.92       469.21     2_974.13       0.8850          1.0243            1.0175         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_896.24       320.61     2_216.85       0.8657          1.0347            1.0259         4.82
IVFPQ-m32-nl158-np7                                    1_710.48       429.67     2_140.15       0.8684          1.0311            1.0245         2.34
IVFPQ-m64-nl158-np7                                    2_504.92       813.44     3_318.36       0.8900          1.0215            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_896.24       520.12     2_416.36       0.8680          1.0321            1.0248         4.82
IVFPQ-m32-nl158-np8                                    1_710.48       491.45     2_201.94       0.8685          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np8                                    2_504.92       899.23     3_404.15       0.8902          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_896.24       580.14     2_476.38       0.8682          1.0317            1.0247         4.82
IVFPQ-m32-nl158-np12                                   1_710.48       717.57     2_428.06       0.8687          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np12                                   2_504.92     1_330.15     3_835.08       0.8903          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_896.24       835.08     2_731.32       0.8686          1.0311            1.0245         4.82
IVFPQ-m32-nl223-np1                                    1_822.26        93.73     1_916.00       0.6865          1.1976            1.1231         2.46
IVFPQ-m64-nl223-np1                                    2_683.79       141.87     2_825.66       0.6934          1.1901            1.1152         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_954.63       102.43     2_057.06       0.8126          1.0799            1.0465         4.94
IVFPQ-m32-nl223-np2                                    1_822.26       147.26     1_969.53       0.8242          1.0647            1.0369         2.46
IVFPQ-m64-nl223-np2                                    2_683.79       239.17     2_922.96       0.8391          1.0564            1.0285         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_954.63       161.22     2_115.85       0.8641          1.0402            1.0265         4.94
IVFPQ-m32-nl223-np4                                    1_822.26       241.51     2_063.77       0.8725          1.0307            1.0222         2.46
IVFPQ-m64-nl223-np4                                    2_683.79       420.88     3_104.67       0.8923          1.0219            1.0149         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_954.63       279.92     2_234.55       0.8759          1.0314            1.0220         4.94
IVFPQ-m32-nl223-np8                                    1_822.26       451.05     2_273.32       0.8793          1.0270            1.0204         2.46
IVFPQ-m64-nl223-np8                                    2_683.79       800.36     3_484.15       0.8997          1.0180            1.0133         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_954.63       517.01     2_471.65       0.8788          1.0281            1.0206         4.94
IVFPQ-m32-nl223-np11                                   1_822.26       603.57     2_425.84       0.8796          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np11                                   2_683.79     1_092.84     3_776.62       0.9001          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_954.63       685.80     2_640.44       0.8794          1.0273            1.0203         4.94
IVFPQ-m32-nl223-np14                                   1_822.26       758.49     2_580.75       0.8797          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np14                                   2_683.79     1_399.32     4_083.11       0.9002          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_954.63       862.59     2_817.22       0.8796          1.0270            1.0203         4.94
IVFPQ-m32-nl316-np1                                    1_867.99        97.30     1_965.29       0.6730          1.2103            1.1374         2.65
IVFPQ-m64-nl316-np1                                    2_675.50       137.19     2_812.69       0.6777          1.2048            1.1331         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_068.89       104.54     2_173.42       0.8095          1.0841            1.0485         5.13
IVFPQ-m32-nl316-np2                                    1_867.99       142.53     2_010.52       0.8234          1.0660            1.0367         2.65
IVFPQ-m64-nl316-np2                                    2_675.50       224.97     2_900.48       0.8337          1.0605            1.0304         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_068.89       156.16     2_225.04       0.8708          1.0386            1.0233         5.13
IVFPQ-m32-nl316-np4                                    1_867.99       236.95     2_104.95       0.8823          1.0266            1.0185         2.65
IVFPQ-m64-nl316-np4                                    2_675.50       405.39     3_080.89       0.8968          1.0204            1.0133         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_068.89       264.01     2_332.89       0.8865          1.0283            1.0182         5.13
IVFPQ-m32-nl316-np8                                    1_867.99       433.53     2_301.52       0.8914          1.0217            1.0162         2.65
IVFPQ-m64-nl316-np8                                    2_675.50       758.77     3_434.27       0.9066          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_068.89       481.25     2_550.14       0.8906          1.0239            1.0167         5.13
IVFPQ-m32-nl316-np15                                   1_867.99       781.66     2_649.66       0.8921          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np15                                   2_675.50     1_387.25     4_062.75       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_068.89       864.85     2_933.74       0.8919          1.0218            1.0162         5.13
IVFPQ-m32-nl316-np17                                   1_867.99       917.38     2_785.37       0.8922          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np17                                   2_675.50     1_568.14     4_243.65       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_068.89       974.92     3_043.80       0.8920          1.0216            1.0161         5.13
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
Exhaustive (query)                                        67.57     1_360.44     1_428.01       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_890.00       111.07     2_001.08       0.8150          1.0767            1.0413         4.82
SOARPQ-near-np2                                        1_890.00       180.48     2_070.48       0.8584          1.0400            1.0282         4.82
SOARPQ-near-np4                                        1_890.00       315.65     2_205.65       0.8665          1.0333            1.0255         4.82
SOARPQ-near-np7                                        1_890.00       511.92     2_401.93       0.8682          1.0316            1.0247         4.82
SOARPQ-near-np8                                        1_890.00       574.33     2_464.33       0.8684          1.0314            1.0246         4.82
SOARPQ-near-np12                                       1_890.00       837.79     2_727.80       0.8687          1.0311            1.0244         4.82
SOARPQ-shift0.3-np1                                    1_882.87       110.81     1_993.68       0.8175          1.0739            1.0429         4.82
SOARPQ-shift0.3-np2                                    1_882.87       179.63     2_062.50       0.8583          1.0407            1.0288         4.82
SOARPQ-shift0.3-np4                                    1_882.87       314.65     2_197.52       0.8660          1.0342            1.0258         4.82
SOARPQ-shift0.3-np7                                    1_882.87       512.07     2_394.94       0.8681          1.0318            1.0248         4.82
SOARPQ-shift0.3-np8                                    1_882.87       573.64     2_456.51       0.8683          1.0315            1.0246         4.82
SOARPQ-shift0.3-np12                                   1_882.87       820.59     2_703.46       0.8686          1.0311            1.0245         4.82
SOARPQ-shift0.7-np1                                    1_878.98       110.18     1_989.16       0.8118          1.0794            1.0463         4.82
SOARPQ-shift0.7-np2                                    1_878.98       179.51     2_058.49       0.8562          1.0430            1.0297         4.82
SOARPQ-shift0.7-np4                                    1_878.98       316.85     2_195.83       0.8653          1.0353            1.0261         4.82
SOARPQ-shift0.7-np7                                    1_878.98       509.69     2_388.67       0.8679          1.0324            1.0249         4.82
SOARPQ-shift0.7-np8                                    1_878.98       576.69     2_455.67       0.8682          1.0319            1.0247         4.82
SOARPQ-shift0.7-np12                                   1_878.98       820.77     2_699.75       0.8686          1.0312            1.0245         4.82
SOARPQ-orth1-np1                                       1_917.22       110.86     2_028.08       0.8149          1.0770            1.0431         4.82
SOARPQ-orth1-np2                                       1_917.22       179.47     2_096.69       0.8579          1.0413            1.0287         4.82
SOARPQ-orth1-np4                                       1_917.22       315.17     2_232.39       0.8661          1.0343            1.0258         4.82
SOARPQ-orth1-np7                                       1_917.22       512.17     2_429.39       0.8681          1.0319            1.0248         4.82
SOARPQ-orth1-np8                                       1_917.22       574.34     2_491.56       0.8683          1.0316            1.0246         4.82
SOARPQ-orth1-np12                                      1_917.22       824.92     2_742.14       0.8686          1.0311            1.0244         4.82
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
Exhaustive (query)                                        71.48     1_331.55     1_403.02       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                    1_469.08        92.20     1_561.28       0.7674          1.1423            1.0643         2.24
IVFPQ-m64-nl111-np1                                    2_264.82       145.64     2_410.47       0.7738          1.1368            1.0562         3.77
SOARPQ-orth1-m32-nl111-np1                             1_653.53       110.34     1_763.87       0.8492          1.0681            1.0350         4.72
IVFPQ-m32-nl111-np2                                    1_469.08       146.93     1_616.01       0.8656          1.0441            1.0265         2.24
IVFPQ-m64-nl111-np2                                    2_264.82       254.31     2_519.13       0.8760          1.0383            1.0207         3.77
SOARPQ-orth1-m32-nl111-np2                             1_653.53       183.91     1_837.44       0.8782          1.0396            1.0248         4.72
IVFPQ-m32-nl111-np4                                    1_469.08       257.59     1_726.66       0.8857          1.0293            1.0218         2.24
IVFPQ-m64-nl111-np4                                    2_264.82       472.43     2_737.25       0.8968          1.0237            1.0165         3.77
SOARPQ-orth1-m32-nl111-np4                             1_653.53       322.43     1_975.96       0.8854          1.0317            1.0223         4.72
IVFPQ-m32-nl111-np5                                    1_469.08       316.55     1_785.63       0.8866          1.0289            1.0215         2.24
IVFPQ-m64-nl111-np5                                    2_264.82       586.76     2_851.59       0.8978          1.0232            1.0163         3.77
SOARPQ-orth1-m32-nl111-np5                             1_653.53       390.47     2_044.00       0.8862          1.0305            1.0219         4.72
IVFPQ-m32-nl111-np8                                    1_469.08       514.52     1_983.59       0.8872          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np8                                    2_264.82       915.43     3_180.25       0.8984          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np8                             1_653.53       606.37     2_259.90       0.8871          1.0290            1.0215         4.72
IVFPQ-m32-nl111-np10                                   1_469.08       637.54     2_106.62       0.8873          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np10                                   2_264.82     1_147.42     3_412.25       0.8985          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np10                            1_653.53       730.65     2_384.18       0.8872          1.0288            1.0214         4.72
IVFPQ-m32-nl158-np1                                    1_766.43        97.17     1_863.61       0.7517          1.1575            1.0756         2.34
IVFPQ-m64-nl158-np1                                    2_599.17       141.55     2_740.72       0.7581          1.1518            1.0676         3.86
SOARPQ-orth1-m32-nl158-np1                             1_967.81       108.21     2_076.02       0.8454          1.0732            1.0355         4.82
IVFPQ-m32-nl158-np2                                    1_766.43       143.48     1_909.92       0.8638          1.0466            1.0258         2.34
IVFPQ-m64-nl158-np2                                    2_599.17       243.17     2_842.34       0.8739          1.0408            1.0199         3.86
SOARPQ-orth1-m32-nl158-np2                             1_967.81       171.96     2_139.77       0.8822          1.0394            1.0227         4.82
IVFPQ-m32-nl158-np4                                    1_766.43       265.83     2_032.27       0.8918          1.0259            1.0187         2.34
IVFPQ-m64-nl158-np4                                    2_599.17       445.60     3_044.77       0.9036          1.0200            1.0138         3.86
SOARPQ-orth1-m32-nl158-np4                             1_967.81       300.90     2_268.71       0.8918          1.0289            1.0191         4.82
IVFPQ-m32-nl158-np7                                    1_766.43       413.79     2_180.23       0.8942          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np7                                    2_599.17       762.42     3_361.59       0.9060          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np7                             1_967.81       495.49     2_463.30       0.8940          1.0254            1.0182         4.82
IVFPQ-m32-nl158-np8                                    1_766.43       468.60     2_235.04       0.8943          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np8                                    2_599.17       866.91     3_466.08       0.9061          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np8                             1_967.81       557.25     2_525.06       0.8941          1.0250            1.0182         4.82
IVFPQ-m32-nl158-np12                                   1_766.43       695.00     2_461.44       0.8944          1.0244            1.0180         2.34
IVFPQ-m64-nl158-np12                                   2_599.17     1_291.62     3_890.80       0.9063          1.0186            1.0132         3.86
SOARPQ-orth1-m32-nl158-np12                            1_967.81       798.15     2_765.96       0.8944          1.0245            1.0180         4.82
IVFPQ-m32-nl223-np1                                    1_861.21        92.25     1_953.46       0.7283          1.1815            1.1041         2.46
IVFPQ-m64-nl223-np1                                    2_668.22       133.76     2_801.98       0.7318          1.1772            1.1004         3.99
SOARPQ-orth1-m32-nl223-np1                             2_106.43        99.42     2_205.85       0.8393          1.0777            1.0371         4.94
IVFPQ-m32-nl223-np2                                    1_861.21       137.56     1_998.77       0.8613          1.0501            1.0263         2.46
IVFPQ-m64-nl223-np2                                    2_668.22       223.89     2_892.11       0.8679          1.0460            1.0217         3.99
SOARPQ-orth1-m32-nl223-np2                             2_106.43       155.35     2_261.78       0.8871          1.0370            1.0205         4.94
IVFPQ-m32-nl223-np4                                    1_861.21       235.43     2_096.65       0.8979          1.0231            1.0163         2.46
IVFPQ-m64-nl223-np4                                    2_668.22       406.34     3_074.56       0.9065          1.0191            1.0133         3.99
SOARPQ-orth1-m32-nl223-np4                             2_106.43       266.63     2_373.06       0.8976          1.0276            1.0170         4.94
IVFPQ-m32-nl223-np8                                    1_861.21       433.37     2_294.58       0.9010          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np8                                    2_668.22       772.56     3_440.78       0.9099          1.0173            1.0126         3.99
SOARPQ-orth1-m32-nl223-np8                             2_106.43       487.28     2_593.71       0.9006          1.0227            1.0160         4.94
IVFPQ-m32-nl223-np11                                   1_861.21       588.20     2_449.41       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np11                                   2_668.22     1_058.44     3_726.66       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np11                            2_106.43       660.87     2_767.30       0.9009          1.0218            1.0158         4.94
IVFPQ-m32-nl223-np14                                   1_861.21       741.70     2_602.92       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np14                                   2_668.22     1_348.47     4_016.69       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np14                            2_106.43       827.98     2_934.41       0.9011          1.0216            1.0157         4.94
IVFPQ-m32-nl316-np1                                    2_085.73        95.36     2_181.08       0.7037          1.2091            1.1328         2.65
IVFPQ-m64-nl316-np1                                    2_918.37       134.19     3_052.56       0.7071          1.2044            1.1272         4.17
SOARPQ-orth1-m32-nl316-np1                             2_377.07       100.91     2_477.98       0.8250          1.0882            1.0452         5.13
IVFPQ-m32-nl316-np2                                    2_085.73       139.58     2_225.30       0.8495          1.0587            1.0313         2.65
IVFPQ-m64-nl316-np2                                    2_918.37       220.96     3_139.34       0.8573          1.0541            1.0258         4.17
SOARPQ-orth1-m32-nl316-np2                             2_377.07       152.48     2_529.55       0.8855          1.0375            1.0212         5.13
IVFPQ-m32-nl316-np4                                    2_085.73       237.79     2_323.52       0.8982          1.0231            1.0160         2.65
IVFPQ-m64-nl316-np4                                    2_918.37       393.71     3_312.09       0.9091          1.0184            1.0119         4.17
SOARPQ-orth1-m32-nl316-np4                             2_377.07       258.85     2_635.92       0.8994          1.0269            1.0166         5.13
IVFPQ-m32-nl316-np8                                    2_085.73       422.85     2_508.58       0.9033          1.0203            1.0148         2.65
IVFPQ-m64-nl316-np8                                    2_918.37       741.65     3_660.03       0.9144          1.0156            1.0109         4.17
SOARPQ-orth1-m32-nl316-np8                             2_377.07       469.73     2_846.79       0.9028          1.0220            1.0152         5.13
IVFPQ-m32-nl316-np15                                   2_085.73       761.36     2_847.09       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np15                                   2_918.37     1_364.40     4_282.78       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np15                            2_377.07       832.31     3_209.37       0.9035          1.0204            1.0148         5.13
IVFPQ-m32-nl316-np17                                   2_085.73       861.43     2_947.16       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np17                                   2_918.37     1_547.23     4_465.60       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np17                            2_377.07       947.36     3_324.42       0.9035          1.0203            1.0148         5.13
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
Exhaustive (query)                                        71.48     1_331.55     1_403.02       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        1_904.20       106.11     2_010.31       0.8501          1.0646            1.0334         4.82
SOARPQ-near-np2                                        1_904.20       170.45     2_074.65       0.8844          1.0345            1.0218         4.82
SOARPQ-near-np4                                        1_904.20       297.12     2_201.32       0.8927          1.0266            1.0188         4.82
SOARPQ-near-np7                                        1_904.20       489.89     2_394.08       0.8942          1.0248            1.0181         4.82
SOARPQ-near-np8                                        1_904.20       548.68     2_452.87       0.8943          1.0246            1.0181         4.82
SOARPQ-near-np12                                       1_904.20       787.79     2_691.99       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.3-np1                                    1_892.76       105.98     1_998.74       0.8492          1.0675            1.0348         4.82
SOARPQ-shift0.3-np2                                    1_892.76       171.27     2_064.03       0.8831          1.0374            1.0226         4.82
SOARPQ-shift0.3-np4                                    1_892.76       297.12     2_189.89       0.8918          1.0282            1.0191         4.82
SOARPQ-shift0.3-np7                                    1_892.76       485.66     2_378.42       0.8940          1.0252            1.0182         4.82
SOARPQ-shift0.3-np8                                    1_892.76       547.16     2_439.92       0.8941          1.0249            1.0182         4.82
SOARPQ-shift0.3-np12                                   1_892.76       785.35     2_678.12       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.7-np1                                    1_878.57       106.18     1_984.74       0.8440          1.0748            1.0368         4.82
SOARPQ-shift0.7-np2                                    1_878.57       179.70     2_058.27       0.8805          1.0417            1.0235         4.82
SOARPQ-shift0.7-np4                                    1_878.57       298.32     2_176.89       0.8906          1.0306            1.0196         4.82
SOARPQ-shift0.7-np7                                    1_878.57       486.89     2_365.46       0.8936          1.0261            1.0184         4.82
SOARPQ-shift0.7-np8                                    1_878.57       549.87     2_428.44       0.8939          1.0256            1.0183         4.82
SOARPQ-shift0.7-np12                                   1_878.57       786.15     2_664.71       0.8943          1.0247            1.0180         4.82
SOARPQ-orth1-np1                                       1_916.53       105.88     2_022.41       0.8454          1.0732            1.0355         4.82
SOARPQ-orth1-np2                                       1_916.53       171.92     2_088.46       0.8822          1.0394            1.0227         4.82
SOARPQ-orth1-np4                                       1_916.53       303.08     2_219.62       0.8918          1.0289            1.0191         4.82
SOARPQ-orth1-np7                                       1_916.53       489.18     2_405.71       0.8940          1.0254            1.0182         4.82
SOARPQ-orth1-np8                                       1_916.53       552.77     2_469.30       0.8941          1.0250            1.0182         4.82
SOARPQ-orth1-np12                                      1_916.53       785.07     2_701.60       0.8944          1.0245            1.0180         4.82
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
Exhaustive (query)                                        67.69     1_225.78     1_293.47       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_049.05       454.17     7_503.21       0.3591          1.0691            1.0690         3.49
IVFOPQ-m64-nl111-np1                                  10_888.53       548.77    11_437.30       0.4589          1.0463            1.0426         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_297.38       456.45     8_753.82       0.3380          1.2238            1.0709         5.97
IVFOPQ-m32-nl111-np2                                   7_049.05       495.92     7_544.97       0.3603          1.0684            1.0688         3.49
IVFOPQ-m64-nl111-np2                                  10_888.53       636.08    11_524.61       0.4610          1.0452            1.0423         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_297.38       512.99     8_810.36       0.3602          1.0697            1.0688         5.97
IVFOPQ-m32-nl111-np4                                   7_049.05       590.65     7_639.70       0.3603          1.0684            1.0688         3.49
IVFOPQ-m64-nl111-np4                                  10_888.53       811.76    11_700.30       0.4610          1.0452            1.0423         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_297.38       606.36     8_903.74       0.3603          1.0685            1.0688         5.97
IVFOPQ-m32-nl111-np5                                   7_049.05       663.94     7_712.99       0.3603          1.0684            1.0688         3.49
IVFOPQ-m64-nl111-np5                                  10_888.53       898.58    11_787.11       0.4610          1.0452            1.0423         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_297.38       659.58     8_956.96       0.3603          1.0685            1.0688         5.97
IVFOPQ-m32-nl111-np8                                   7_049.05       779.09     7_828.13       0.3603          1.0684            1.0688         3.49
IVFOPQ-m64-nl111-np8                                  10_888.53     1_168.30    12_056.83       0.4610          1.0452            1.0423         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_297.38       816.28     9_113.65       0.3603          1.0684            1.0688         5.97
IVFOPQ-m32-nl111-np10                                  7_049.05       871.82     7_920.87       0.3603          1.0684            1.0688         3.49
IVFOPQ-m64-nl111-np10                                 10_888.53     1_331.99    12_220.53       0.4610          1.0452            1.0423         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_297.38       922.94     9_220.32       0.3603          1.0684            1.0688         5.97
IVFOPQ-m32-nl158-np1                                   7_482.16       453.28     7_935.44       0.3644          1.0674            1.0680         3.84
IVFOPQ-m64-nl158-np1                                  11_353.54       544.98    11_898.52       0.4691          1.0434            1.0412         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         8_722.18       461.50     9_183.68       0.3353          1.1675            1.0709         6.32
IVFOPQ-m32-nl158-np2                                   7_482.16       498.56     7_980.72       0.3685          1.0657            1.0674         3.84
IVFOPQ-m64-nl158-np2                                  11_353.54       636.11    11_989.65       0.4749          1.0412            1.0405         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         8_722.18       510.80     9_232.98       0.3679          1.0668            1.0675         6.32
IVFOPQ-m32-nl158-np4                                   7_482.16       591.24     8_073.40       0.3685          1.0657            1.0674         3.84
IVFOPQ-m64-nl158-np4                                  11_353.54       802.31    12_155.85       0.4749          1.0412            1.0405         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         8_722.18       603.50     9_325.68       0.3685          1.0657            1.0674         6.32
IVFOPQ-m32-nl158-np7                                   7_482.16       728.71     8_210.87       0.3685          1.0657            1.0674         3.84
IVFOPQ-m64-nl158-np7                                  11_353.54     1_056.62    12_410.16       0.4749          1.0412            1.0405         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         8_722.18       748.92     9_471.10       0.3685          1.0657            1.0674         6.32
IVFOPQ-m32-nl158-np8                                   7_482.16       772.21     8_254.36       0.3685          1.0657            1.0674         3.84
IVFOPQ-m64-nl158-np8                                  11_353.54     1_142.28    12_495.83       0.4749          1.0412            1.0405         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         8_722.18       793.46     9_515.64       0.3685          1.0657            1.0674         6.32
IVFOPQ-m32-nl158-np12                                  7_482.16       954.25     8_436.41       0.3685          1.0657            1.0674         3.84
IVFOPQ-m64-nl158-np12                                 11_353.54     1_487.83    12_841.37       0.4749          1.0412            1.0405         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        8_722.18       984.93     9_707.11       0.3685          1.0657            1.0674         6.32
IVFOPQ-m32-nl223-np1                                   8_189.51       436.33     8_625.84       0.3605          1.0671            1.0672         3.96
IVFOPQ-m64-nl223-np1                                  12_711.98       488.56    13_200.54       0.4462          1.0466            1.0436         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         8_982.32       439.13     9_421.45       0.3468          1.1127            1.0693         6.44
IVFOPQ-m32-nl223-np2                                   8_189.51       479.77     8_669.28       0.3750          1.0625            1.0637         3.96
IVFOPQ-m64-nl223-np2                                  12_711.98       586.79    13_298.77       0.4767          1.0403            1.0391         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         8_982.32       500.67     9_482.98       0.3734          1.0640            1.0651         6.44
IVFOPQ-m32-nl223-np4                                   8_189.51       580.54     8_770.05       0.3787          1.0618            1.0628         3.96
IVFOPQ-m64-nl223-np4                                  12_711.98       779.35    13_491.33       0.4868          1.0388            1.0376         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         8_982.32       601.30     9_583.62       0.3773          1.0626            1.0634         6.44
IVFOPQ-m32-nl223-np8                                   8_189.51       815.22     9_004.74       0.3791          1.0617            1.0627         3.96
IVFOPQ-m64-nl223-np8                                  12_711.98     1_129.94    13_841.92       0.4881          1.0386            1.0373         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         8_982.32       796.84     9_779.16       0.3790          1.0617            1.0627         6.44
IVFOPQ-m32-nl223-np11                                  8_189.51       910.76     9_100.27       0.3791          1.0617            1.0627         3.96
IVFOPQ-m64-nl223-np11                                 12_711.98     1_382.58    14_094.57       0.4881          1.0386            1.0373         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        8_982.32       938.51     9_920.82       0.3791          1.0617            1.0627         6.44
IVFOPQ-m32-nl223-np14                                  8_189.51     1_044.75     9_234.26       0.3791          1.0617            1.0627         3.96
IVFOPQ-m64-nl223-np14                                 12_711.98     1_651.86    14_363.84       0.4881          1.0386            1.0373         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        8_982.32     1_085.92    10_068.23       0.3791          1.0617            1.0627         6.44
IVFOPQ-m32-nl316-np1                                   7_876.78       430.01     8_306.79       0.3586          1.0668            1.0665         4.65
IVFOPQ-m64-nl316-np1                                  11_831.34       473.47    12_304.81       0.4350          1.0483            1.0447         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_174.29       435.52     9_609.81       0.3396          1.1105            1.0695         7.13
IVFOPQ-m32-nl316-np2                                   7_876.78       474.80     8_351.57       0.3795          1.0599            1.0617         4.65
IVFOPQ-m64-nl316-np2                                  11_831.34       569.70    12_401.05       0.4780          1.0391            1.0380         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_174.29       496.07     9_670.35       0.3775          1.0619            1.0636         7.13
IVFOPQ-m32-nl316-np4                                   7_876.78       575.93     8_452.70       0.3861          1.0583            1.0603         4.65
IVFOPQ-m64-nl316-np4                                  11_831.34       756.68    12_588.03       0.4951          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_174.29       594.97     9_769.25       0.3829          1.0599            1.0616         7.13
IVFOPQ-m32-nl316-np8                                   7_876.78       764.26     8_641.03       0.3871          1.0582            1.0601         4.65
IVFOPQ-m64-nl316-np8                                  11_831.34     1_106.12    12_937.47       0.4989          1.0360            1.0350         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_174.29       789.70     9_963.99       0.3865          1.0585            1.0603         7.13
IVFOPQ-m32-nl316-np15                                  7_876.78     1_085.92     8_962.69       0.3872          1.0581            1.0601         4.65
IVFOPQ-m64-nl316-np15                                 11_831.34     1_767.77    13_599.12       0.4993          1.0359            1.0349         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_174.29     1_119.48    10_293.77       0.3872          1.0582            1.0601         7.13
IVFOPQ-m32-nl316-np17                                  7_876.78     1_187.94     9_064.72       0.3872          1.0581            1.0601         4.65
IVFOPQ-m64-nl316-np17                                 11_831.34     1_880.43    13_711.78       0.4993          1.0359            1.0349         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_174.29     1_245.05    10_419.34       0.3872          1.0581            1.0601         7.13
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
Exhaustive (query)                                        67.69     1_225.78     1_293.47       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       8_735.16       460.57     9_195.73       0.3362          1.1647            1.0708         6.32
SOAROPQ-near-np2                                       8_735.16       511.55     9_246.71       0.3681          1.0664            1.0675         6.32
SOAROPQ-near-np4                                       8_735.16       607.30     9_342.46       0.3685          1.0657            1.0674         6.32
SOAROPQ-near-np7                                       8_735.16       749.41     9_484.57       0.3685          1.0657            1.0674         6.32
SOAROPQ-near-np8                                       8_735.16       794.00     9_529.15       0.3685          1.0657            1.0674         6.32
SOAROPQ-near-np12                                      8_735.16       989.05     9_724.21       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.3-np1                                   8_765.41       460.23     9_225.64       0.3356          1.1661            1.0709         6.32
SOAROPQ-shift0.3-np2                                   8_765.41       513.96     9_279.36       0.3680          1.0667            1.0675         6.32
SOAROPQ-shift0.3-np4                                   8_765.41       604.08     9_369.49       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.3-np7                                   8_765.41       744.22     9_509.63       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.3-np8                                   8_765.41       793.56     9_558.97       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.3-np12                                  8_765.41       983.56     9_748.96       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.7-np1                                   8_748.27       462.94     9_211.21       0.3353          1.1678            1.0709         6.32
SOAROPQ-shift0.7-np2                                   8_748.27       508.30     9_256.57       0.3678          1.0672            1.0675         6.32
SOAROPQ-shift0.7-np4                                   8_748.27       604.25     9_352.51       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.7-np7                                   8_748.27       747.30     9_495.57       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.7-np8                                   8_748.27       793.37     9_541.64       0.3685          1.0657            1.0674         6.32
SOAROPQ-shift0.7-np12                                  8_748.27       984.76     9_733.02       0.3685          1.0657            1.0674         6.32
SOAROPQ-orth1-np1                                      8_784.79       461.53     9_246.32       0.3368          1.1648            1.0707         6.32
SOAROPQ-orth1-np2                                      8_784.79       508.08     9_292.86       0.3683          1.0661            1.0675         6.32
SOAROPQ-orth1-np4                                      8_784.79       605.90     9_390.69       0.3685          1.0657            1.0674         6.32
SOAROPQ-orth1-np7                                      8_784.79       747.45     9_532.24       0.3685          1.0657            1.0674         6.32
SOAROPQ-orth1-np8                                      8_784.79       793.11     9_577.89       0.3685          1.0657            1.0674         6.32
SOAROPQ-orth1-np12                                     8_784.79       982.91     9_767.69       0.3685          1.0657            1.0674         6.32
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
Exhaustive (query)                                        67.58     1_321.67     1_389.26       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_109.11       446.70     7_555.81       0.6656          1.0297            1.0249         3.49
IVFOPQ-m64-nl111-np1                                  10_932.00       537.77    11_469.77       0.7586          1.0166            1.0113         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_327.93       451.77     8_779.70       0.6662          1.0330            1.0251         5.97
IVFOPQ-m32-nl111-np2                                   7_109.11       491.67     7_600.79       0.6771          1.0259            1.0243         3.49
IVFOPQ-m64-nl111-np2                                  10_932.00       627.50    11_559.50       0.7728          1.0125            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_327.93       504.35     8_832.28       0.6773          1.0260            1.0243         5.97
IVFOPQ-m32-nl111-np4                                   7_109.11       585.17     7_694.29       0.6777          1.0258            1.0243         3.49
IVFOPQ-m64-nl111-np4                                  10_932.00       801.17    11_733.17       0.7736          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_327.93       600.36     8_928.29       0.6777          1.0258            1.0243         5.97
IVFOPQ-m32-nl111-np5                                   7_109.11       634.04     7_743.15       0.6777          1.0258            1.0243         3.49
IVFOPQ-m64-nl111-np5                                  10_932.00       966.16    11_898.16       0.7736          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_327.93       654.74     8_982.67       0.6777          1.0258            1.0243         5.97
IVFOPQ-m32-nl111-np8                                   7_109.11       770.30     7_879.41       0.6777          1.0258            1.0243         3.49
IVFOPQ-m64-nl111-np8                                  10_932.00     1_147.51    12_079.51       0.7736          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_327.93       824.32     9_152.26       0.6777          1.0258            1.0243         5.97
IVFOPQ-m32-nl111-np10                                  7_109.11       866.99     7_976.10       0.6777          1.0258            1.0243         3.49
IVFOPQ-m64-nl111-np10                                 10_932.00     1_319.49    12_251.49       0.7736          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_327.93       925.58     9_253.51       0.6777          1.0258            1.0243         5.97
IVFOPQ-m32-nl158-np1                                   7_533.74       456.41     7_990.15       0.6621          1.0313            1.0246         3.84
IVFOPQ-m64-nl158-np1                                  11_423.05       541.22    11_964.26       0.7539          1.0185            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         8_842.02       455.72     9_297.74       0.6751          1.0277            1.0244         6.32
IVFOPQ-m32-nl158-np2                                   7_533.74       491.86     8_025.59       0.6806          1.0253            1.0236         3.84
IVFOPQ-m64-nl158-np2                                  11_423.05       624.45    12_047.49       0.7758          1.0123            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         8_842.02       509.48     9_351.50       0.6820          1.0250            1.0236         6.32
IVFOPQ-m32-nl158-np4                                   7_533.74       586.35     8_120.09       0.6827          1.0248            1.0235         3.84
IVFOPQ-m64-nl158-np4                                  11_423.05       792.71    12_215.75       0.7783          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         8_842.02       597.27     9_439.29       0.6827          1.0248            1.0235         6.32
IVFOPQ-m32-nl158-np7                                   7_533.74       718.30     8_252.03       0.6828          1.0247            1.0235         3.84
IVFOPQ-m64-nl158-np7                                  11_423.05     1_041.18    12_464.23       0.7785          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         8_842.02       737.30     9_579.32       0.6828          1.0247            1.0235         6.32
IVFOPQ-m32-nl158-np8                                   7_533.74       760.93     8_294.67       0.6828          1.0247            1.0235         3.84
IVFOPQ-m64-nl158-np8                                  11_423.05     1_129.52    12_552.57       0.7785          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         8_842.02       782.55     9_624.58       0.6828          1.0247            1.0235         6.32
IVFOPQ-m32-nl158-np12                                  7_533.74       942.98     8_476.71       0.6828          1.0247            1.0235         3.84
IVFOPQ-m64-nl158-np12                                 11_423.05     1_457.93    12_880.98       0.7785          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        8_842.02       976.43     9_818.45       0.6828          1.0247            1.0235         6.32
IVFOPQ-m32-nl223-np1                                   7_775.11       433.49     8_208.61       0.4952          1.0652            1.0577         3.96
IVFOPQ-m64-nl223-np1                                  11_756.43       484.39    12_240.82       0.5345          1.0542            1.0466         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_007.67       439.81     9_447.48       0.6002          1.0414            1.0353         6.44
IVFOPQ-m32-nl223-np2                                   7_775.11       481.34     8_256.46       0.6128          1.0373            1.0324         3.96
IVFOPQ-m64-nl223-np2                                  11_756.43       590.21    12_346.63       0.6817          1.0254            1.0186         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_007.67       497.25     9_504.92       0.6619          1.0284            1.0264         6.44
IVFOPQ-m32-nl223-np4                                   7_775.11       580.82     8_355.94       0.6721          1.0265            1.0248         3.96
IVFOPQ-m64-nl223-np4                                  11_756.43       771.40    12_527.83       0.7584          1.0143            1.0119         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_007.67       597.47     9_605.14       0.6863          1.0242            1.0229         6.44
IVFOPQ-m32-nl223-np8                                   7_775.11       762.15     8_537.26       0.6893          1.0236            1.0225         3.96
IVFOPQ-m64-nl223-np8                                  11_756.43     1_118.69    12_875.11       0.7806          1.0112            1.0102         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_007.67       785.42     9_793.09       0.6902          1.0235            1.0224         6.44
IVFOPQ-m32-nl223-np11                                  7_775.11       899.97     8_675.09       0.6905          1.0234            1.0223         3.96
IVFOPQ-m64-nl223-np11                                 11_756.43     1_371.59    13_128.02       0.7820          1.0111            1.0100         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_007.67       923.72     9_931.39       0.6905          1.0234            1.0223         6.44
IVFOPQ-m32-nl223-np14                                  7_775.11     1_042.49     8_817.61       0.6905          1.0234            1.0223         3.96
IVFOPQ-m64-nl223-np14                                 11_756.43     1_627.82    13_384.24       0.7820          1.0111            1.0100         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_007.67     1_070.11    10_077.78       0.6905          1.0234            1.0223         6.44
IVFOPQ-m32-nl316-np1                                   7_883.39       431.62     8_315.01       0.4092          1.0856            1.0798         4.65
IVFOPQ-m64-nl316-np1                                  12_266.32       471.04    12_737.35       0.4290          1.0753            1.0690         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_197.41       439.47     9_636.88       0.5361          1.0533            1.0491         7.13
IVFOPQ-m32-nl316-np2                                   7_883.39       474.24     8_357.63       0.5490          1.0493            1.0458         4.65
IVFOPQ-m64-nl316-np2                                  12_266.32       569.60    12_835.92       0.5975          1.0379            1.0338         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_197.41       493.19     9_690.61       0.6288          1.0343            1.0317         7.13
IVFOPQ-m32-nl316-np4                                   7_883.39       575.31     8_458.70       0.6437          1.0316            1.0290         4.65
IVFOPQ-m64-nl316-np4                                  12_266.32       774.41    13_040.72       0.7200          1.0196            1.0162         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_197.41       596.21     9_793.62       0.6780          1.0257            1.0243         7.13
IVFOPQ-m32-nl316-np8                                   7_883.39       764.09     8_647.48       0.6874          1.0241            1.0228         4.65
IVFOPQ-m64-nl316-np8                                  12_266.32     1_101.42    13_367.73       0.7778          1.0118            1.0106         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_197.41       781.34     9_978.76       0.6923          1.0233            1.0221         7.13
IVFOPQ-m32-nl316-np15                                  7_883.39     1_077.69     8_961.08       0.6931          1.0231            1.0220         4.65
IVFOPQ-m64-nl316-np15                                 12_266.32     1_800.52    14_066.84       0.7852          1.0109            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_197.41     1_106.03    10_303.44       0.6931          1.0231            1.0220         7.13
IVFOPQ-m32-nl316-np17                                  7_883.39     1_178.97     9_062.36       0.6931          1.0231            1.0220         4.65
IVFOPQ-m64-nl316-np17                                 12_266.32     1_888.20    14_154.52       0.7852          1.0109            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_197.41     1_207.44    10_404.86       0.6931          1.0231            1.0220         7.13
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
Exhaustive (query)                                        67.58     1_321.67     1_389.26       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       8_793.58       457.67     9_251.26       0.6753          1.0276            1.0244         6.32
SOAROPQ-near-np2                                       8_793.58       504.93     9_298.51       0.6820          1.0250            1.0236         6.32
SOAROPQ-near-np4                                       8_793.58       595.96     9_389.54       0.6827          1.0247            1.0235         6.32
SOAROPQ-near-np7                                       8_793.58       734.01     9_527.60       0.6828          1.0247            1.0235         6.32
SOAROPQ-near-np8                                       8_793.58       786.64     9_580.23       0.6828          1.0247            1.0235         6.32
SOAROPQ-near-np12                                      8_793.58       982.54     9_776.12       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.3-np1                                   8_766.18       458.41     9_224.59       0.6752          1.0276            1.0244         6.32
SOAROPQ-shift0.3-np2                                   8_766.18       502.41     9_268.59       0.6820          1.0250            1.0236         6.32
SOAROPQ-shift0.3-np4                                   8_766.18       595.28     9_361.45       0.6827          1.0247            1.0235         6.32
SOAROPQ-shift0.3-np7                                   8_766.18       736.80     9_502.98       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.3-np8                                   8_766.18       783.04     9_549.21       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.3-np12                                  8_766.18       976.05     9_742.22       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.7-np1                                   8_824.39       456.93     9_281.31       0.6750          1.0277            1.0244         6.32
SOAROPQ-shift0.7-np2                                   8_824.39       525.00     9_349.39       0.6820          1.0251            1.0236         6.32
SOAROPQ-shift0.7-np4                                   8_824.39       601.26     9_425.64       0.6827          1.0248            1.0235         6.32
SOAROPQ-shift0.7-np7                                   8_824.39       733.08     9_557.47       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.7-np8                                   8_824.39       780.80     9_605.18       0.6828          1.0247            1.0235         6.32
SOAROPQ-shift0.7-np12                                  8_824.39       974.72     9_799.11       0.6828          1.0247            1.0235         6.32
SOAROPQ-orth1-np1                                      8_777.41       455.53     9_232.94       0.6749          1.0278            1.0244         6.32
SOAROPQ-orth1-np2                                      8_777.41       503.94     9_281.35       0.6819          1.0251            1.0236         6.32
SOAROPQ-orth1-np4                                      8_777.41       596.18     9_373.59       0.6827          1.0248            1.0235         6.32
SOAROPQ-orth1-np7                                      8_777.41       734.49     9_511.90       0.6828          1.0247            1.0235         6.32
SOAROPQ-orth1-np8                                      8_777.41       782.64     9_560.05       0.6828          1.0247            1.0235         6.32
SOAROPQ-orth1-np12                                     8_777.41       973.63     9_751.04       0.6828          1.0247            1.0235         6.32
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
Exhaustive (query)                                        67.50     1_317.32     1_384.82       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_222.56       427.60     7_650.16       0.7214          1.1713            1.0833         3.49
IVFOPQ-m64-nl111-np1                                  11_042.03       487.03    11_529.06       0.7257          1.1682            1.0805         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_434.61       449.50     8_884.11       0.8431          1.0561            1.0278         5.97
IVFOPQ-m32-nl111-np2                                   7_222.56       508.24     7_730.80       0.8513          1.0486            1.0239         3.49
IVFOPQ-m64-nl111-np2                                  11_042.03       611.06    11_653.09       0.8604          1.0451            1.0195         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_434.61       540.60     8_975.21       0.8848          1.0261            1.0173         5.97
IVFOPQ-m32-nl111-np4                                   7_222.56       608.92     7_831.48       0.8899          1.0220            1.0158         3.49
IVFOPQ-m64-nl111-np4                                  11_042.03       855.69    11_897.72       0.9014          1.0183            1.0121         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_434.61       691.14     9_125.75       0.8921          1.0212            1.0154         5.97
IVFOPQ-m32-nl111-np5                                   7_222.56       669.87     7_892.43       0.8918          1.0208            1.0154         3.49
IVFOPQ-m64-nl111-np5                                  11_042.03       978.42    12_020.45       0.9035          1.0171            1.0118         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_434.61       768.61     9_203.22       0.8926          1.0209            1.0153         5.97
IVFOPQ-m32-nl111-np8                                   7_222.56       856.43     8_078.99       0.8931          1.0202            1.0152         3.49
IVFOPQ-m64-nl111-np8                                  11_042.03     1_355.40    12_397.43       0.9049          1.0165            1.0116         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_434.61       975.63     9_410.24       0.8930          1.0204            1.0152         5.97
IVFOPQ-m32-nl111-np10                                  7_222.56       976.39     8_198.95       0.8932          1.0202            1.0151         3.49
IVFOPQ-m64-nl111-np10                                 11_042.03     1_601.82    12_643.85       0.9050          1.0164            1.0115         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_434.61     1_117.36     9_551.97       0.8931          1.0203            1.0152         5.97
IVFOPQ-m32-nl158-np1                                   7_705.58       432.40     8_137.98       0.7098          1.1800            1.0949         3.84
IVFOPQ-m64-nl158-np1                                  11_587.35       473.82    12_061.17       0.7129          1.1777            1.0932         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         8_934.59       450.34     9_384.93       0.8401          1.0597            1.0281         6.32
IVFOPQ-m32-nl158-np2                                   7_705.58       477.51     8_183.08       0.8498          1.0504            1.0231         3.84
IVFOPQ-m64-nl158-np2                                  11_587.35       584.73    12_172.08       0.8567          1.0477            1.0202         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         8_934.59       507.08     9_441.67       0.8903          1.0246            1.0154         6.32
IVFOPQ-m32-nl158-np4                                   7_705.58       587.68     8_293.26       0.8956          1.0200            1.0139         3.84
IVFOPQ-m64-nl158-np4                                  11_587.35       807.37    12_394.72       0.9049          1.0174            1.0109         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         8_934.59       645.43     9_580.01       0.8994          1.0187            1.0132         6.32
IVFOPQ-m32-nl158-np7                                   7_705.58       802.00     8_507.58       0.9009          1.0172            1.0129         3.84
IVFOPQ-m64-nl158-np7                                  11_587.35     1_126.32    12_713.67       0.9106          1.0145            1.0100         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         8_934.59       840.20     9_774.78       0.9008          1.0175            1.0129         6.32
IVFOPQ-m32-nl158-np8                                   7_705.58       817.16     8_522.74       0.9011          1.0171            1.0128         3.84
IVFOPQ-m64-nl158-np8                                  11_587.35     1_237.99    12_825.35       0.9108          1.0144            1.0099         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         8_934.59       903.25     9_837.83       0.9010          1.0174            1.0128         6.32
IVFOPQ-m32-nl158-np12                                  7_705.58     1_031.99     8_737.57       0.9013          1.0170            1.0127         3.84
IVFOPQ-m64-nl158-np12                                 11_587.35     1_665.26    13_252.61       0.9111          1.0143            1.0099         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        8_934.59     1_151.94    10_086.52       0.9013          1.0170            1.0127         6.32
IVFOPQ-m32-nl223-np1                                   8_015.27       425.93     8_441.20       0.6955          1.1883            1.1142         3.96
IVFOPQ-m64-nl223-np1                                  11_952.54       467.56    12_420.10       0.6973          1.1860            1.1115         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_236.85       428.62     9_665.46       0.8323          1.0657            1.0321         6.44
IVFOPQ-m32-nl223-np2                                   8_015.27       472.25     8_487.53       0.8445          1.0542            1.0256         3.96
IVFOPQ-m64-nl223-np2                                  11_952.54       584.26    12_536.80       0.8501          1.0515            1.0221         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_236.85       511.29     9_748.14       0.8925          1.0251            1.0147         6.44
IVFOPQ-m32-nl223-np4                                   8_015.27       570.79     8_586.07       0.9001          1.0191            1.0124         3.96
IVFOPQ-m64-nl223-np4                                  11_952.54       757.63    12_710.17       0.9079          1.0164            1.0097         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_236.85       608.25     9_845.10       0.9055          1.0174            1.0116         6.44
IVFOPQ-m32-nl223-np8                                   8_015.27       774.17     8_789.45       0.9080          1.0153            1.0108         3.96
IVFOPQ-m64-nl223-np8                                  11_952.54     1_146.21    13_098.75       0.9163          1.0124            1.0084         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_236.85       852.58    10_089.42       0.9079          1.0157            1.0109         6.44
IVFOPQ-m32-nl223-np11                                  8_015.27       932.31     8_947.58       0.9084          1.0151            1.0108         3.96
IVFOPQ-m64-nl223-np11                                 11_952.54     1_441.98    13_394.52       0.9168          1.0122            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_236.85     1_013.07    10_249.92       0.9082          1.0153            1.0108         6.44
IVFOPQ-m32-nl223-np14                                  8_015.27     1_082.56     9_097.83       0.9084          1.0151            1.0107         3.96
IVFOPQ-m64-nl223-np14                                 11_952.54     1_735.62    13_688.15       0.9169          1.0122            1.0082         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_236.85     1_179.27    10_416.12       0.9083          1.0152            1.0108         6.44
IVFOPQ-m32-nl316-np1                                   8_085.44       426.81     8_512.25       0.6787          1.2032            1.1307         4.65
IVFOPQ-m64-nl316-np1                                  12_059.77       466.84    12_526.61       0.6803          1.2012            1.1297         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_382.81       427.81     9_810.62       0.8248          1.0715            1.0363         7.13
IVFOPQ-m32-nl316-np2                                   8_085.44       470.46     8_555.90       0.8378          1.0585            1.0279         4.65
IVFOPQ-m64-nl316-np2                                  12_059.77       557.25    12_617.02       0.8413          1.0569            1.0257         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_382.81       485.96     9_868.77       0.8938          1.0251            1.0141         7.13
IVFOPQ-m32-nl316-np4                                   8_085.44       573.63     8_659.08       0.9031          1.0183            1.0112         4.65
IVFOPQ-m64-nl316-np4                                  12_059.77       741.37    12_801.14       0.9086          1.0164            1.0094         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_382.81       593.85     9_976.66       0.9105          1.0160            1.0103         7.13
IVFOPQ-m32-nl316-np8                                   8_085.44       760.63     8_846.07       0.9137          1.0131            1.0093         4.65
IVFOPQ-m64-nl316-np8                                  12_059.77     1_103.97    13_163.74       0.9198          1.0112            1.0076         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_382.81       807.41    10_190.21       0.9137          1.0137            1.0094         7.13
IVFOPQ-m32-nl316-np15                                  8_085.44     1_101.63     9_187.07       0.9145          1.0128            1.0091         4.65
IVFOPQ-m64-nl316-np15                                 12_059.77     1_745.29    13_805.06       0.9207          1.0109            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_382.81     1_181.37    10_564.17       0.9145          1.0129            1.0091         7.13
IVFOPQ-m32-nl316-np17                                  8_085.44     1_205.64     9_291.08       0.9145          1.0128            1.0091         4.65
IVFOPQ-m64-nl316-np17                                 12_059.77     1_937.54    13_997.31       0.9207          1.0109            1.0074         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_382.81     1_290.61    10_673.41       0.9145          1.0129            1.0091         7.13
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
Exhaustive (query)                                        67.50     1_317.32     1_384.82       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       8_966.18       469.66     9_435.84       0.8390          1.0616            1.0262         6.32
SOAROPQ-near-np2                                       8_966.18       508.76     9_474.95       0.8902          1.0243            1.0151         6.32
SOAROPQ-near-np4                                       8_966.18       648.43     9_614.61       0.8998          1.0182            1.0131         6.32
SOAROPQ-near-np7                                       8_966.18       841.12     9_807.30       0.9010          1.0173            1.0128         6.32
SOAROPQ-near-np8                                       8_966.18       903.53     9_869.72       0.9011          1.0172            1.0128         6.32
SOAROPQ-near-np12                                      8_966.18     1_148.60    10_114.78       0.9013          1.0170            1.0127         6.32
SOAROPQ-shift0.3-np1                                   8_943.41       441.27     9_384.68       0.8422          1.0578            1.0271         6.32
SOAROPQ-shift0.3-np2                                   8_943.41       507.41     9_450.82       0.8911          1.0239            1.0152         6.32
SOAROPQ-shift0.3-np4                                   8_943.41       646.09     9_589.50       0.8996          1.0184            1.0132         6.32
SOAROPQ-shift0.3-np7                                   8_943.41       840.42     9_783.83       0.9009          1.0174            1.0128         6.32
SOAROPQ-shift0.3-np8                                   8_943.41       905.04     9_848.45       0.9010          1.0173            1.0128         6.32
SOAROPQ-shift0.3-np12                                  8_943.41     1_149.07    10_092.48       0.9013          1.0170            1.0127         6.32
SOAROPQ-shift0.7-np1                                   8_927.84       447.39     9_375.23       0.8366          1.0626            1.0295         6.32
SOAROPQ-shift0.7-np2                                   8_927.84       514.26     9_442.09       0.8892          1.0254            1.0158         6.32
SOAROPQ-shift0.7-np4                                   8_927.84       644.51     9_572.34       0.8991          1.0189            1.0133         6.32
SOAROPQ-shift0.7-np7                                   8_927.84       841.12     9_768.96       0.9007          1.0176            1.0129         6.32
SOAROPQ-shift0.7-np8                                   8_927.84       903.21     9_831.05       0.9009          1.0174            1.0129         6.32
SOAROPQ-shift0.7-np12                                  8_927.84     1_149.37    10_077.20       0.9012          1.0171            1.0127         6.32
SOAROPQ-orth1-np1                                      8_961.45       442.80     9_404.25       0.8392          1.0608            1.0273         6.32
SOAROPQ-orth1-np2                                      8_961.45       509.32     9_470.77       0.8902          1.0245            1.0153         6.32
SOAROPQ-orth1-np4                                      8_961.45       645.20     9_606.65       0.8996          1.0186            1.0132         6.32
SOAROPQ-orth1-np7                                      8_961.45       845.70     9_807.15       0.9009          1.0174            1.0128         6.32
SOAROPQ-orth1-np8                                      8_961.45       904.85     9_866.31       0.9010          1.0173            1.0128         6.32
SOAROPQ-orth1-np12                                     8_961.45     1_148.17    10_109.62       0.9013          1.0170            1.0127         6.32
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
Exhaustive (query)                                        71.51     1_286.98     1_358.49       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   7_226.30       414.98     7_641.27       0.7752          1.1355            1.0547         3.49
IVFOPQ-m64-nl111-np1                                  10_972.70       473.54    11_446.24       0.7782          1.1336            1.0518         5.02
SOAROPQ-orth1-m32-nl111-np1                            8_426.12       434.97     8_861.10       0.8632          1.0554            1.0262         5.97
IVFOPQ-m32-nl111-np2                                   7_226.30       472.79     7_699.09       0.8796          1.0365            1.0194         3.49
IVFOPQ-m64-nl111-np2                                  10_972.70       587.65    11_560.35       0.8841          1.0345            1.0173         5.02
SOAROPQ-orth1-m32-nl111-np2                            8_426.12       508.52     8_934.65       0.8947          1.0284            1.0173         5.97
IVFOPQ-m32-nl111-np4                                   7_226.30       586.58     7_812.87       0.9011          1.0218            1.0152         3.49
IVFOPQ-m64-nl111-np4                                  10_972.70       811.50    11_784.20       0.9062          1.0196            1.0131         5.02
SOAROPQ-orth1-m32-nl111-np4                            8_426.12       651.59     9_077.72       0.9014          1.0227            1.0153         5.97
IVFOPQ-m32-nl111-np5                                   7_226.30       645.97     7_872.27       0.9022          1.0213            1.0149         3.49
IVFOPQ-m64-nl111-np5                                  10_972.70       935.03    11_907.73       0.9073          1.0191            1.0128         5.02
SOAROPQ-orth1-m32-nl111-np5                            8_426.12       722.27     9_148.40       0.9021          1.0221            1.0151         5.97
IVFOPQ-m32-nl111-np8                                   7_226.30       820.77     8_047.07       0.9028          1.0210            1.0148         3.49
IVFOPQ-m64-nl111-np8                                  10_972.70     1_279.90    12_252.59       0.9080          1.0189            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np8                            8_426.12       932.63     9_358.75       0.9027          1.0213            1.0148         5.97
IVFOPQ-m32-nl111-np10                                  7_226.30       935.82     8_162.12       0.9029          1.0210            1.0147         3.49
IVFOPQ-m64-nl111-np10                                 10_972.70     1_502.14    12_474.84       0.9080          1.0189            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np10                           8_426.12     1_058.81     9_484.94       0.9028          1.0211            1.0148         5.97
IVFOPQ-m32-nl158-np1                                   7_721.84       422.40     8_144.24       0.7586          1.1515            1.0673         3.84
IVFOPQ-m64-nl158-np1                                  11_675.47       478.58    12_154.06       0.7615          1.1491            1.0637         5.36
SOAROPQ-orth1-m32-nl158-np1                            9_106.01       431.09     9_537.10       0.8577          1.0611            1.0275         6.32
IVFOPQ-m32-nl158-np2                                   7_721.84       472.42     8_194.26       0.8755          1.0404            1.0196         3.84
IVFOPQ-m64-nl158-np2                                  11_675.47       575.22    12_250.70       0.8808          1.0379            1.0169         5.36
SOAROPQ-orth1-m32-nl158-np2                            9_106.01       499.41     9_605.42       0.8967          1.0287            1.0165         6.32
IVFOPQ-m32-nl158-np4                                   7_721.84       579.12     8_300.96       0.9055          1.0193            1.0134         3.84
IVFOPQ-m64-nl158-np4                                  11_675.47       787.47    12_462.95       0.9118          1.0169            1.0112         5.36
SOAROPQ-orth1-m32-nl158-np4                            9_106.01       634.62     9_740.63       0.9061          1.0204            1.0135         6.32
IVFOPQ-m32-nl158-np7                                   7_721.84       748.71     8_470.55       0.9079          1.0179            1.0128         3.84
IVFOPQ-m64-nl158-np7                                  11_675.47     1_102.56    12_778.04       0.9143          1.0156            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np7                            9_106.01       824.07     9_930.08       0.9077          1.0184            1.0129         6.32
IVFOPQ-m32-nl158-np8                                   7_721.84       813.36     8_535.20       0.9080          1.0179            1.0127         3.84
IVFOPQ-m64-nl158-np8                                  11_675.47     1_212.98    12_888.45       0.9144          1.0155            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np8                            9_106.01       906.84    10_012.85       0.9079          1.0182            1.0128         6.32
IVFOPQ-m32-nl158-np12                                  7_721.84     1_017.85     8_739.69       0.9082          1.0178            1.0127         3.84
IVFOPQ-m64-nl158-np12                                 11_675.47     1_652.95    13_328.43       0.9146          1.0155            1.0105         5.36
SOAROPQ-orth1-m32-nl158-np12                           9_106.01     1_122.47    10_228.48       0.9082          1.0179            1.0127         6.32
IVFOPQ-m32-nl223-np1                                   8_079.56       420.21     8_499.76       0.7329          1.1767            1.0996         3.96
IVFOPQ-m64-nl223-np1                                  12_059.23       461.85    12_521.08       0.7345          1.1748            1.0972         5.49
SOAROPQ-orth1-m32-nl223-np1                            9_369.37       427.64     9_797.01       0.8490          1.0690            1.0296         6.44
IVFOPQ-m32-nl223-np2                                   8_079.56       465.76     8_545.32       0.8706          1.0452            1.0207         3.96
IVFOPQ-m64-nl223-np2                                  12_059.23       560.01    12_619.24       0.8739          1.0435            1.0184         5.49
SOAROPQ-orth1-m32-nl223-np2                            9_369.37       482.08     9_851.45       0.8995          1.0283            1.0151         6.44
IVFOPQ-m32-nl223-np4                                   8_079.56       559.10     8_638.66       0.9097          1.0180            1.0121         3.96
IVFOPQ-m64-nl223-np4                                  12_059.23       747.65    12_806.88       0.9146          1.0163            1.0103         5.49
SOAROPQ-orth1-m32-nl223-np4                            9_369.37       595.88     9_965.25       0.9105          1.0200            1.0123         6.44
IVFOPQ-m32-nl223-np8                                   8_079.56       751.70     8_831.26       0.9131          1.0162            1.0115         3.96
IVFOPQ-m64-nl223-np8                                  12_059.23     1_117.26    13_176.49       0.9183          1.0145            1.0095         5.49
SOAROPQ-orth1-m32-nl223-np8                            9_369.37       816.03    10_185.40       0.9128          1.0169            1.0115         6.44
IVFOPQ-m32-nl223-np11                                  8_079.56       901.19     8_980.74       0.9133          1.0162            1.0114         3.96
IVFOPQ-m64-nl223-np11                                 12_059.23     1_402.82    13_462.05       0.9185          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np11                           9_369.37       988.35    10_357.72       0.9132          1.0164            1.0115         6.44
IVFOPQ-m32-nl223-np14                                  8_079.56     1_043.88     9_123.44       0.9133          1.0162            1.0114         3.96
IVFOPQ-m64-nl223-np14                                 12_059.23     1_703.44    13_762.67       0.9186          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np14                           9_369.37     1_159.05    10_528.42       0.9132          1.0163            1.0114         6.44
IVFOPQ-m32-nl316-np1                                   8_313.79       418.51     8_732.30       0.7081          1.2041            1.1269         4.65
IVFOPQ-m64-nl316-np1                                  12_526.54       465.23    12_991.77       0.7095          1.2018            1.1247         6.17
SOAROPQ-orth1-m32-nl316-np1                            9_676.05       426.25    10_102.29       0.8349          1.0801            1.0369         7.13
IVFOPQ-m32-nl316-np2                                   8_313.79       466.12     8_779.91       0.8592          1.0535            1.0249         4.65
IVFOPQ-m64-nl316-np2                                  12_526.54       554.12    13_080.66       0.8629          1.0516            1.0225         6.17
SOAROPQ-orth1-m32-nl316-np2                            9_676.05       479.48    10_155.52       0.8984          1.0293            1.0154         7.13
IVFOPQ-m32-nl316-np4                                   8_313.79       558.96     8_872.75       0.9111          1.0177            1.0113         4.65
IVFOPQ-m64-nl316-np4                                  12_526.54       731.44    13_257.98       0.9171          1.0158            1.0094         6.17
SOAROPQ-orth1-m32-nl316-np4                            9_676.05       587.90    10_263.94       0.9131          1.0193            1.0113         7.13
IVFOPQ-m32-nl316-np8                                   8_313.79       743.04     9_056.83       0.9167          1.0149            1.0103         4.65
IVFOPQ-m64-nl316-np8                                  12_526.54     1_100.58    13_627.13       0.9232          1.0129            1.0083         6.17
SOAROPQ-orth1-m32-nl316-np8                            9_676.05       792.42    10_468.47       0.9163          1.0159            1.0104         7.13
IVFOPQ-m32-nl316-np15                                  8_313.79     1_077.39     9_391.18       0.9170          1.0148            1.0102         4.65
IVFOPQ-m64-nl316-np15                                 12_526.54     1_720.62    14_247.16       0.9236          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np15                           9_676.05     1_164.75    10_840.80       0.9170          1.0149            1.0102         7.13
IVFOPQ-m32-nl316-np17                                  8_313.79     1_172.81     9_486.60       0.9170          1.0148            1.0102         4.65
IVFOPQ-m64-nl316-np17                                 12_526.54     1_906.24    14_432.78       0.9236          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np17                           9_676.05     1_275.26    10_951.30       0.9170          1.0148            1.0102         7.13
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
Exhaustive (query)                                        71.51     1_286.98     1_358.49       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                       8_935.85       434.79     9_370.63       0.8620          1.0550            1.0260         6.32
SOAROPQ-near-np2                                       8_935.85       499.19     9_435.04       0.8986          1.0257            1.0159         6.32
SOAROPQ-near-np4                                       8_935.85       661.17     9_597.02       0.9068          1.0191            1.0133         6.32
SOAROPQ-near-np7                                       8_935.85       820.58     9_756.43       0.9080          1.0180            1.0128         6.32
SOAROPQ-near-np8                                       8_935.85       884.50     9_820.35       0.9081          1.0179            1.0128         6.32
SOAROPQ-near-np12                                      8_935.85     1_122.86    10_058.70       0.9082          1.0178            1.0127         6.32
SOAROPQ-shift0.3-np1                                   8_927.34       438.64     9_365.98       0.8617          1.0563            1.0272         6.32
SOAROPQ-shift0.3-np2                                   8_927.34       499.22     9_426.56       0.8974          1.0275            1.0164         6.32
SOAROPQ-shift0.3-np4                                   8_927.34       630.11     9_557.45       0.9060          1.0201            1.0135         6.32
SOAROPQ-shift0.3-np7                                   8_927.34       822.27     9_749.61       0.9077          1.0184            1.0129         6.32
SOAROPQ-shift0.3-np8                                   8_927.34       884.12     9_811.46       0.9079          1.0182            1.0128         6.32
SOAROPQ-shift0.3-np12                                  8_927.34     1_139.22    10_066.56       0.9082          1.0179            1.0127         6.32
SOAROPQ-shift0.7-np1                                   8_922.84       436.76     9_359.60       0.8567          1.0625            1.0288         6.32
SOAROPQ-shift0.7-np2                                   8_922.84       502.85     9_425.69       0.8951          1.0306            1.0170         6.32
SOAROPQ-shift0.7-np4                                   8_922.84       630.17     9_553.01       0.9050          1.0217            1.0139         6.32
SOAROPQ-shift0.7-np7                                   8_922.84       822.41     9_745.25       0.9074          1.0190            1.0130         6.32
SOAROPQ-shift0.7-np8                                   8_922.84       884.23     9_807.07       0.9077          1.0186            1.0130         6.32
SOAROPQ-shift0.7-np12                                  8_922.84     1_124.16    10_047.00       0.9081          1.0180            1.0127         6.32
SOAROPQ-orth1-np1                                      8_997.61       437.54     9_435.15       0.8577          1.0611            1.0275         6.32
SOAROPQ-orth1-np2                                      8_997.61       502.29     9_499.91       0.8967          1.0287            1.0165         6.32
SOAROPQ-orth1-np4                                      8_997.61       634.97     9_632.58       0.9061          1.0204            1.0135         6.32
SOAROPQ-orth1-np7                                      8_997.61       826.08     9_823.69       0.9077          1.0184            1.0129         6.32
SOAROPQ-orth1-np8                                      8_997.61       885.14     9_882.76       0.9079          1.0182            1.0128         6.32
SOAROPQ-orth1-np12                                     8_997.61     1_131.33    10_128.94       0.9082          1.0179            1.0127         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.12.0 (commit v0.11.2-37-ge44af67), run on 2026-10-31.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
