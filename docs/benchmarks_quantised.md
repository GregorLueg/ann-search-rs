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
Exhaustive (query)                                        12.18       654.32       666.51       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.18     6_445.61     6_457.79       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   15.63     1_183.37     1_199.00       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    15.63    12_535.04    12_550.67       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              259.37        97.49       356.86       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              259.37       103.51       362.88       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              259.37       145.23       404.60       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    259.37     1_330.62     1_589.99       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              327.75        88.17       415.92       0.9820          1.0001            1.0000         9.21
IVF-BF16-nl387-np27 (query)                              327.75       114.00       441.76       0.9828          1.0001            1.0000         9.21
IVF-BF16-nl387 (self)                                    327.75     1_159.60     1_487.35       0.9798          1.0001            1.0000         9.21
IVF-BF16-nl547-np23 (query)                              507.19        83.27       590.46       0.9776          1.0005            1.0000         9.23
IVF-BF16-nl547-np27 (query)                              507.19        91.37       598.56       0.9817          1.0002            1.0000         9.23
IVF-BF16-nl547-np33 (query)                              507.19       108.99       616.18       0.9828          1.0001            1.0000         9.23
IVF-BF16-nl547 (self)                                    507.19     1_090.90     1_598.09       0.9798          1.0001            1.0000         9.23
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
Exhaustive (query)                                        12.69       707.62       720.31       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.69     6_993.00     7_005.69       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   13.04     1_243.33     1_256.37       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    13.04    12_465.92    12_478.97       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              176.22        91.12       267.34       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              176.22       104.96       281.18       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              176.22       142.67       318.89       0.8871          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    176.22     1_503.89     1_680.11       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              291.51        93.55       385.06       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              291.51       123.02       414.53       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    291.51     1_280.91     1_572.42       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                              518.85        84.82       603.67       0.8849          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                              518.85        96.67       615.52       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                              518.85       113.00       631.85       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                    518.85     1_167.41     1_686.26       0.8852          1.0073            1.0020         9.51
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
Exhaustive (query)                                        11.29       638.23       649.51       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.29     6_254.04     6_265.32       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.11     1_202.63     1_215.74       0.9344          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    13.11    11_787.02    11_800.13       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              200.55        89.04       289.59       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              200.55        89.32       289.87       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              200.55       117.62       318.16       0.9344          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    200.55     1_167.13     1_367.68       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              301.33        95.36       396.69       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387-np27 (query)                              301.33       113.67       415.00       0.9344          1.0018            1.0011         9.21
IVF-BF16-nl387 (self)                                    301.33     1_032.82     1_334.15       0.9184          1.0030            1.0021         9.21
IVF-BF16-nl547-np23 (query)                              507.83        76.86       584.69       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np27 (query)                              507.83        85.16       592.99       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547-np33 (query)                              507.83       109.38       617.21       0.9344          1.0018            1.0011         9.23
IVF-BF16-nl547 (self)                                    507.83       971.50     1_479.34       0.9184          1.0030            1.0021         9.23
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
Exhaustive (query)                                        11.41       661.36       672.77       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.41     6_188.76     6_200.17       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   13.47     1_170.09     1_183.55       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    13.47    11_789.89    11_803.36       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              195.36        77.14       272.50       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              195.36        86.54       281.90       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              195.36       116.90       312.26       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    195.36     1_193.58     1_388.95       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              303.29        81.05       384.34       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387-np27 (query)                              303.29       102.01       405.30       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387 (self)                                    303.29     1_033.33     1_336.63       0.9429          1.0017            1.0009         9.21
IVF-BF16-nl547-np23 (query)                              540.85        75.79       616.64       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np27 (query)                              540.85        83.65       624.49       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np33 (query)                              540.85        96.08       636.93       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547 (self)                                    540.85       963.26     1_504.11       0.9429          1.0017            1.0009         9.23
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
Exhaustive (query)                                        51.14     1_215.20     1_266.34       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         51.14    11_929.37    11_980.51       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   59.71     5_178.57     5_238.28       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    59.71    53_313.82    53_373.54       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              445.40       262.34       707.74       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              445.40       297.58       742.98       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              445.40       440.60       886.00       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    445.40     4_299.58     4_744.98       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                              725.77       271.90       997.68       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                              725.77       363.77     1_089.55       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                    725.77     3_646.65     4_372.42       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            1_208.06       258.72     1_466.78       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            1_208.06       286.35     1_494.42       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            1_208.06       340.32     1_548.38       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                  1_208.06     3_382.03     4_590.09       0.9679          1.0005            1.0000        36.89
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
Exhaustive (query)                                        11.38       687.20       698.58       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.38     6_212.78     6_224.16       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.12       977.33       994.45       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     17.12     9_896.65     9_913.77       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               182.25        65.36       247.60       0.9233          1.0021            1.0010         6.33
IVF-SQ8-nl273-np16 (query)                               182.25        71.55       253.80       0.9246          1.0019            1.0009         6.33
IVF-SQ8-nl273-np23 (query)                               182.25        95.72       277.96       0.9249          1.0019            1.0009         6.33
IVF-SQ8-nl273 (self)                                     182.25       917.55     1_099.80       0.9249          1.0018            1.0009         6.33
IVF-SQ8-nl387-np19 (query)                               284.38        66.33       350.71       0.9255          1.0019            1.0009         6.35
IVF-SQ8-nl387-np27 (query)                               284.38        84.17       368.55       0.9260          1.0018            1.0009         6.35
IVF-SQ8-nl387 (self)                                     284.38       787.75     1_072.13       0.9252          1.0018            1.0009         6.35
IVF-SQ8-nl547-np23 (query)                               500.20        69.32       569.52       0.9223          1.0022            1.0010         6.37
IVF-SQ8-nl547-np27 (query)                               500.20        68.53       568.73       0.9250          1.0019            1.0009         6.37
IVF-SQ8-nl547-np33 (query)                               500.20        79.94       580.14       0.9257          1.0018            1.0009         6.37
IVF-SQ8-nl547 (self)                                     500.20       740.45     1_240.65       0.9252          1.0018            1.0009         6.37
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
Exhaustive (query)                                        13.26       736.48       749.74       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         13.26     6_958.69     6_971.95       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    19.69       897.66       917.35       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     19.69    10_097.27    10_116.95       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               178.55        63.16       241.70       0.7368          1.0365            1.0161         6.33
IVF-SQ8-nl273-np16 (query)                               178.55        70.87       249.41       0.7369          1.0365            1.0161         6.33
IVF-SQ8-nl273-np23 (query)                               178.55        94.75       273.29       0.7369          1.0365            1.0161         6.33
IVF-SQ8-nl273 (self)                                     178.55       962.27     1_140.81       0.7358          1.0368            1.0158         6.33
IVF-SQ8-nl387-np19 (query)                               296.91        63.95       360.86       0.7379          1.0358            1.0161         6.35
IVF-SQ8-nl387-np27 (query)                               296.91        81.48       378.39       0.7380          1.0358            1.0161         6.35
IVF-SQ8-nl387 (self)                                     296.91       828.06     1_124.96       0.7387          1.0356            1.0157         6.35
IVF-SQ8-nl547-np23 (query)                               510.59        57.64       568.23       0.7359          1.0369            1.0163         6.37
IVF-SQ8-nl547-np27 (query)                               510.59        66.36       576.96       0.7362          1.0369            1.0161         6.37
IVF-SQ8-nl547-np33 (query)                               510.59        76.34       586.93       0.7362          1.0369            1.0161         6.37
IVF-SQ8-nl547 (self)                                     510.59       741.33     1_251.92       0.7361          1.0368            1.0159         6.37
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
Exhaustive (query)                                        11.28       639.00       650.28       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.28     6_332.42     6_343.70       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    16.99       971.33       988.32       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     16.99    10_146.87    10_163.86       0.8119          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               188.25        63.15       251.40       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273-np16 (query)                               188.25        71.63       259.88       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273-np23 (query)                               188.25        98.34       286.59       0.8148          1.0165            1.0145         6.33
IVF-SQ8-nl273 (self)                                     188.25       809.05       997.30       0.8119          1.0175            1.0155         6.33
IVF-SQ8-nl387-np19 (query)                               301.34        63.51       364.85       0.8150          1.0164            1.0145         6.35
IVF-SQ8-nl387-np27 (query)                               301.34        76.40       377.74       0.8150          1.0164            1.0145         6.35
IVF-SQ8-nl387 (self)                                     301.34       724.23     1_025.57       0.8122          1.0174            1.0155         6.35
IVF-SQ8-nl547-np23 (query)                               508.40        58.28       566.67       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547-np27 (query)                               508.40        63.98       572.38       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547-np33 (query)                               508.40        74.28       582.68       0.8155          1.0164            1.0146         6.37
IVF-SQ8-nl547 (self)                                     508.40       666.88     1_175.27       0.8122          1.0174            1.0155         6.37
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
Exhaustive (query)                                        11.15       645.79       656.94       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.15     6_257.39     6_268.54       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    17.00     1_006.48     1_023.48       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     17.00     9_859.83     9_876.83       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                               199.06        58.47       257.53       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273-np16 (query)                               199.06        65.37       264.42       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273-np23 (query)                               199.06        87.82       286.88       0.7906          1.0265            1.0241         6.33
IVF-SQ8-nl273 (self)                                     199.06       827.79     1_026.85       0.7897          1.0281            1.0256         6.33
IVF-SQ8-nl387-np19 (query)                               314.33        60.61       374.95       0.7899          1.0265            1.0243         6.35
IVF-SQ8-nl387-np27 (query)                               314.33        77.41       391.74       0.7899          1.0265            1.0243         6.35
IVF-SQ8-nl387 (self)                                     314.33       710.82     1_025.15       0.7901          1.0280            1.0256         6.35
IVF-SQ8-nl547-np23 (query)                               556.09        58.48       614.56       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547-np27 (query)                               556.09        64.18       620.27       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547-np33 (query)                               556.09        73.78       629.87       0.7897          1.0264            1.0241         6.37
IVF-SQ8-nl547 (self)                                     556.09       684.37     1_240.46       0.7902          1.0280            1.0256         6.37
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
Exhaustive (query)                                        54.15     1_269.49     1_323.64       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         54.15    12_547.69    12_601.83       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    91.23     1_246.54     1_337.78       0.8798          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     91.23    12_244.66    12_335.90       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               541.60        82.54       624.14       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np16 (query)                               541.60        87.96       629.56       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np23 (query)                               541.60       114.00       655.60       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273 (self)                                     541.60       924.60     1_466.20       0.8865          1.0073            1.0059        20.16
IVF-SQ8-nl387-np19 (query)                               830.99        87.93       918.92       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387-np27 (query)                               830.99       104.66       935.65       0.8800          1.0061            1.0051        20.22
IVF-SQ8-nl387 (self)                                     830.99       826.67     1_657.66       0.8867          1.0073            1.0059        20.22
IVF-SQ8-nl547-np23 (query)                             1_292.51        89.03     1_381.54       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np27 (query)                             1_292.51        95.16     1_387.67       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np33 (query)                             1_292.51       106.23     1_398.74       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547 (self)                                   1_292.51       807.83     2_100.33       0.8865          1.0073            1.0059        20.30
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
Exhaustive (query)                                        11.24       685.58       696.82       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.24     6_559.16     6_570.39       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    21.45     1_020.03     1_041.48       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               839.06        48.95       888.01       0.9302          1.0137            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              839.06        87.95       927.01       0.9644          1.0078            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              839.06       169.57     1_008.63       0.9832          1.0036            1.0000        38.52
HNSW-M16-ef100 (self)                                    839.06       861.23     1_700.29       0.9644          1.0069            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_611.48        53.48     1_664.96       0.9577          1.0206            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_611.48        94.35     1_705.84       0.9825          1.0090            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_611.48       175.32     1_786.80       0.9919          1.0055            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_611.48       918.32     2_529.80       0.9828          1.0064            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_764.59        56.43     1_821.02       0.9678          1.0457            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_764.59       103.64     1_868.23       0.9867          1.0259            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_764.59       189.10     1_953.69       0.9948          1.0061            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_764.59     1_017.99     2_782.58       0.9873          1.0184            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_947.72        61.75     2_009.47       0.9727          1.0098            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_947.72       133.90     2_081.62       0.9897          1.0050            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_947.72       198.30     2_146.03       0.9963          1.0003            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_947.72     1_050.28     2_998.00       0.9901          1.0036            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          736.25        37.19       773.44       0.8781          1.0118            1.0032        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         736.25        70.53       806.78       0.9026          1.0078            1.0020        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         736.25       133.60       869.85       0.9153          1.0054            1.0014        26.89
HNSW-SQ8U-M16-ef100 (self)                               736.25       639.55     1_375.80       0.9019          1.0081            1.0020        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_396.25        41.53     1_437.79       0.8995          1.0127            1.0021        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_396.25        72.85     1_469.10       0.9152          1.0061            1.0014        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_396.25       132.45     1_528.71       0.9210          1.0035            1.0011        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_396.25       682.45     2_078.70       0.9148          1.0066            1.0014        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_594.26        56.13     1_650.39       0.9065          1.0091            1.0017        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_594.26        78.93     1_673.19       0.9185          1.0061            1.0012        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_594.26       143.47     1_737.73       0.9230          1.0023            1.0010        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_594.26       773.58     2_367.84       0.9180          1.0050            1.0012        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_646.31        46.75     1_693.06       0.9092          1.0064            1.0016        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_646.31        88.60     1_734.91       0.9195          1.0047            1.0012        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_646.31       155.38     1_801.68       0.9233          1.0030            1.0010        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_646.31       783.53     2_429.83       0.9192          1.0043            1.0012        45.20
HNSW-SQ8U-drop0 (query)                                1_398.89        74.26     1_473.15       0.8951          1.0100            1.0024        26.89
HNSW-SQ8U-drop0.001 (query)                            1_416.48        70.43     1_486.91       0.9145          1.0077            1.0014        26.89
HNSW-SQ8U-drop0.01 (query)                             1_408.17        91.84     1_500.02       0.8979          1.0132            1.0018        26.89
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
Exhaustive (query)                                        12.09       707.76       719.85       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.09     7_037.37     7_049.46       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    23.00       933.23       956.23       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               886.37        70.32       956.69       0.9342          1.0167            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              886.37        94.48       980.85       0.9682          1.0108            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              886.37       182.00     1_068.37       0.9870          1.0053            1.0000        39.09
HNSW-M16-ef100 (self)                                    886.37       927.68     1_814.05       0.9690          1.0091            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_680.07        64.66     1_744.73       0.9643          1.0063            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_680.07        98.61     1_778.68       0.9870          1.0030            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_680.07       199.18     1_879.24       0.9952          1.0009            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_680.07     1_011.20     2_691.27       0.9872          1.0036            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_809.58        79.10     1_888.68       0.9736          1.0122            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_809.58       107.18     1_916.76       0.9909          1.0028            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_809.58       193.76     2_003.34       0.9969          1.0002            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_809.58     1_036.83     2_846.41       0.9910          1.0022            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_840.44        64.69     1_905.13       0.9759          1.0031            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_840.44       120.99     1_961.42       0.9918          1.0018            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_840.44       201.58     2_042.02       0.9974          1.0002            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_840.44     1_083.99     2_924.43       0.9919          1.0011            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          760.30        41.42       801.72       0.6851          1.0544            1.0287        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         760.30        73.32       833.62       0.7090          1.0466            1.0240        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         760.30       132.66       892.96       0.7225          1.0429            1.0208        26.89
HNSW-SQ8U-M16-ef100 (self)                               760.30       671.38     1_431.68       0.7088          1.0472            1.0238        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_451.50        42.26     1_493.75       0.7069          1.0705            1.0237        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_451.50        73.58     1_525.08       0.7237          1.0425            1.0202        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_451.50       135.73     1_587.23       0.7313          1.0391            1.0186        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_451.50       686.99     2_138.49       0.7232          1.0458            1.0200        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_605.76        63.68     1_669.45       0.7184          1.0497            1.0207        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_605.76        79.93     1_685.69       0.7307          1.0426            1.0183        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_605.76       143.54     1_749.31       0.7357          1.0368            1.0173        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_605.76       759.39     2_365.15       0.7299          1.0407            1.0180        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_726.47        48.83     1_775.31       0.7212          1.1300            1.0196        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_726.47        85.49     1_811.97       0.7327          1.0503            1.0175        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_726.47       154.05     1_880.52       0.7368          1.0384            1.0168        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_726.47       823.07     2_549.55       0.7319          1.0526            1.0173        45.20
HNSW-SQ8U-drop0 (query)                                1_452.97        69.57     1_522.53       0.6644          1.0697            1.0306        26.89
HNSW-SQ8U-drop0.001 (query)                            1_498.49        72.16     1_570.65       0.7243          1.0420            1.0201        26.89
HNSW-SQ8U-drop0.01 (query)                             1_409.69        74.74     1_484.43       0.6897          1.0526            1.0260        26.89
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
Exhaustive (query)                                        11.16       713.75       724.91       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.16     7_355.35     7_366.51       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.73     1_048.95     1_067.67       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               856.71        52.26       908.97       0.9946          2.7691            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              856.71        89.21       945.92       0.9989          1.0410            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              856.71       168.67     1_025.38       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    856.71       879.42     1_736.13       0.9991          1.0326            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_473.25        50.97     1_524.23       0.9980          1.2341            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_473.25        91.81     1_565.07       0.9997          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_473.25       165.86     1_639.11       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_473.25       853.53     2_326.78       0.9998          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_562.87        77.73     1_640.60       0.9992          1.0001            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_562.87       104.42     1_667.29       0.9998          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_562.87       175.68     1_738.55       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_562.87       899.10     2_461.97       0.9999          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_715.22        58.37     1_773.59       0.9993          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_715.22       101.71     1_816.92       0.9998          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_715.22       174.32     1_889.53       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_715.22       914.44     2_629.66       0.9999          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          813.47        41.00       854.47       0.8135          1.1866            1.0149        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         813.47        71.82       885.29       0.8144          1.0166            1.0148        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         813.47       124.54       938.01       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef100 (self)                               813.47       659.75     1_473.22       0.8118          1.0176            1.0156        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_346.57        47.73     1_394.30       0.8142          1.0166            1.0148        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_346.57        69.83     1_416.40       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_346.57       122.05     1_468.63       0.8146          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_346.57       637.92     1_984.49       0.8119          1.0175            1.0155        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_452.42        46.67     1_499.10       0.8143          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_452.42        70.24     1_522.66       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_452.42       129.91     1_582.34       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_452.42       668.62     2_121.05       0.8119          1.0175            1.0155        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_571.98        45.53     1_617.51       0.8143          1.0166            1.0148        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_571.98        96.38     1_668.36       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_571.98       138.72     1_710.69       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_571.98       769.13     2_341.11       0.8119          1.0175            1.0155        45.20
HNSW-SQ8U-drop0 (query)                                1_487.12        89.75     1_576.87       0.8081          1.0176            1.0157        26.89
HNSW-SQ8U-drop0.001 (query)                            1_347.82        95.47     1_443.30       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-drop0.01 (query)                             1_380.89       108.05     1_488.93       0.8049          1.0256            1.0163        26.89
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
Exhaustive (query)                                        11.13       709.95       721.08       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.13     7_163.84     7_174.97       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    37.64     1_048.98     1_086.63       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               967.75        55.12     1_022.88       0.9981          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              967.75        98.64     1_066.39       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              967.75       175.87     1_143.62       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    967.75       940.49     1_908.24       0.9998          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_660.33        60.51     1_720.83       0.9985          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_660.33       101.07     1_761.39       0.9999          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_660.33       185.84     1_846.17       1.0000          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_660.33       976.81     2_637.13       0.9999          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_733.67        65.17     1_798.84       0.9994          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_733.67       112.16     1_845.83       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_733.67       198.01     1_931.68       1.0000          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_733.67     1_074.39     2_808.06       1.0000          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_815.89        71.47     1_887.36       0.9994          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_815.89       121.32     1_937.20       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_815.89       203.14     2_019.02       1.0000          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_815.89     1_080.34     2_896.22       1.0000          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          837.45        49.17       886.62       0.7889          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         837.45        78.42       915.87       0.7892          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         837.45       170.08     1_007.53       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100 (self)                               837.45       711.02     1_548.48       0.7896          1.0282            1.0258        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_494.87        49.42     1_544.29       0.7890          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_494.87        74.91     1_569.78       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_494.87       148.91     1_643.78       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_494.87       717.94     2_212.81       0.7897          1.0281            1.0258        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_599.13        45.15     1_644.28       0.7891          1.0267            1.0244        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_599.13        81.20     1_680.33       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_599.13       147.52     1_746.65       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_599.13       792.56     2_391.69       0.7897          1.0281            1.0258        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_737.48        48.90     1_786.38       0.7891          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_737.48        89.14     1_826.62       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_737.48       154.27     1_891.75       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_737.48       834.23     2_571.71       0.7897          1.0281            1.0258        45.20
HNSW-SQ8U-drop0 (query)                                1_500.42        73.56     1_573.98       0.7860          1.0279            1.0254        26.89
HNSW-SQ8U-drop0.001 (query)                            1_594.41        82.19     1_676.60       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-drop0.01 (query)                             1_504.48        75.88     1_580.36       0.7830          1.0294            1.0262        26.89
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
Exhaustive (query)                                        56.28     1_331.51     1_387.79       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         56.28    13_202.39    13_258.68       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    85.91     1_282.59     1_368.49       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_429.69        84.37     1_514.06       0.9934          1.0299            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_429.69       144.12     1_573.81       0.9956          1.0161            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_429.69       258.39     1_688.08       0.9969          1.0088            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_429.69     1_419.29     2_848.98       0.9954          1.0188            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_591.36        91.94     2_683.30       0.9963          1.0249            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_591.36       155.90     2_747.26       0.9976          1.0152            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_591.36       277.03     2_868.39       0.9989          1.0057            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_591.36     1_500.40     4_091.76       0.9977          1.0151            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_760.65        95.15     2_855.80       0.9978          1.0149            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_760.65       167.62     2_928.27       0.9985          1.0096            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_760.65       296.92     3_057.57       0.9994          1.0020            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_760.65     1_577.40     4_338.05       0.9987          1.0087            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_856.44        94.84     2_951.28       0.9980          1.0130            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_856.44       161.47     3_017.91       0.9994          1.0035            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_856.44       296.98     3_153.42       0.9997          1.0018            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_856.44     1_592.18     4_448.62       0.9993          1.0040            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          882.99        42.31       925.31       0.9270          1.0461            1.0038        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         882.99        76.46       959.45       0.9295          1.0292            1.0038        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         882.99       133.21     1_016.20       0.9314          1.0168            1.0038        40.63
HNSW-SQ8U-M16-ef100 (self)                               882.99       697.80     1_580.79       0.9298          1.0257            1.0038        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_596.48        50.18     1_646.66       0.9310          1.0283            1.0037        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_596.48        77.11     1_673.59       0.9325          1.0175            1.0037        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_596.48       139.07     1_735.55       0.9335          1.0097            1.0036        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_596.48       735.34     2_331.82       0.9321          1.0188            1.0037        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_732.79        52.44     1_785.23       0.9324          1.0179            1.0036        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_732.79        81.49     1_814.28       0.9329          1.0141            1.0036        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_732.79       150.46     1_883.25       0.9335          1.0103            1.0036        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_732.79       793.11     2_525.90       0.9329          1.0124            1.0036        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_881.51        50.13     1_931.64       0.9333          1.0114            1.0036        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_881.51        81.65     1_963.15       0.9338          1.0083            1.0036        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_881.51       141.73     2_023.24       0.9339          1.0077            1.0036        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_881.51       797.49     2_679.00       0.9334          1.0096            1.0036        58.94
HNSW-SQ8U-drop0 (query)                                1_610.02        76.46     1_686.48       0.8644          1.0431            1.0220        40.63
HNSW-SQ8U-drop0.001 (query)                            1_586.16        77.95     1_664.10       0.9325          1.0169            1.0036        40.63
HNSW-SQ8U-drop0.01 (query)                             1_585.75        76.27     1_662.03       0.9319          1.0433            1.0020        40.63
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
Exhaustive (query)                                        58.00     1_344.58     1_402.58       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         58.00    13_550.67    13_608.67       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    99.60     1_260.84     1_360.44       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_319.05        77.70     1_396.74       0.9933          1.1132            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_319.05       129.41     1_448.45       0.9970          1.0405            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_319.05       220.28     1_539.32       0.9980          1.0238            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_319.05     1_195.84     2_514.88       0.9967          1.0417            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_456.17        81.21     2_537.37       0.9944          1.1352            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_456.17       132.10     2_588.27       0.9968          1.0698            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_456.17       235.25     2_691.41       0.9987          1.0218            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_456.17     1_266.66     3_722.83       0.9963          1.0770            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_517.23        78.51     2_595.74       0.9987          1.0155            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_517.23       134.84     2_652.06       0.9991          1.0112            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_517.23       243.42     2_760.65       0.9998          1.0017            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_517.23     1_305.71     3_822.93       0.9993          1.0074            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_641.83        80.57     2_722.40       0.9994          1.0092            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_641.83       145.52     2_787.35       0.9997          1.0043            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_641.83       247.96     2_889.78       0.9998          1.0014            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_641.83     1_323.09     3_964.91       0.9997          1.0040            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          856.65        47.44       904.08       0.6634          1.4206            1.1644        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         856.65        72.22       928.86       0.6652          1.3852            1.1628        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         856.65       125.96       982.60       0.6663          1.3604            1.1621        40.63
HNSW-SQ8U-M16-ef100 (self)                               856.65       656.42     1_513.06       0.6652          1.3832            1.1633        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_559.86        48.93     1_608.78       0.6641          1.4186            1.1635        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_559.86        71.63     1_631.48       0.6661          1.3721            1.1621        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_559.86       138.45     1_698.31       0.6668          1.3554            1.1617        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_559.86       702.56     2_262.42       0.6651          1.3927            1.1629        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_649.55        44.69     1_694.25       0.6663          1.3691            1.1620        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_649.55        74.52     1_724.08       0.6667          1.3605            1.1617        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_649.55       134.56     1_784.11       0.6670          1.3546            1.1615        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_649.55       715.88     2_365.44       0.6666          1.3614            1.1619        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_734.98        51.42     1_786.40       0.6669          1.3559            1.1616        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_734.98        81.89     1_816.87       0.6672          1.3498            1.1614        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_734.98       141.10     1_876.08       0.6674          1.3484            1.1614        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_734.98       771.49     2_506.46       0.6670          1.3532            1.1616        58.94
HNSW-SQ8U-drop0 (query)                                1_560.60        92.27     1_652.87       0.6188          1.5391            1.2373        40.63
HNSW-SQ8U-drop0.001 (query)                            1_520.84        71.00     1_591.84       0.6655          1.3818            1.1626        40.63
HNSW-SQ8U-drop0.01 (query)                             1_515.96        69.46     1_585.42       0.6784          1.3446            1.1509        40.63
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
Exhaustive (query)                                        33.32       725.63       758.95       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.32     2_351.62     2_384.94       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                668.75       691.99     1_360.73       0.2581          1.1827            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 668.75     2_241.99     2_910.74       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_247.87     1_557.34     2_805.20       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_247.87     5_210.77     6_458.64       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              2_044.12     3_657.54     5_701.66       0.3611          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               2_044.12    12_228.26    14_272.39       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                             906.01       213.57     1_119.57       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np12 (query)                            906.01       329.69     1_235.70       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16-np17 (query)                            906.01       458.33     1_364.33       0.3704          1.0978            1.0999         1.17
IVF-PQ-nl158-m16 (self)                                  906.01     1_571.32     2_477.32       0.3038          1.1285            1.1336         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_493.79       370.29     1_864.08       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_493.79       595.57     2_089.36       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_493.79       807.22     2_301.01       0.4817          1.0605            1.0575         1.93
IVF-PQ-nl158-m32 (self)                                1_493.79     2_666.94     4_160.72       0.4075          1.0802            1.0796         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_055.38       660.64     2_716.01       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_055.38     1_033.52     3_088.89       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_055.38     1_406.20     3_461.57       0.6905          1.0206            1.0167         3.46
IVF-PQ-nl158-m64 (self)                                2_055.38     4_720.35     6_775.72       0.6325          1.0279            1.0244         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_027.39       307.40     1_334.79       0.3866          1.0890            1.0897         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_027.39       387.99     1_415.38       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_027.39       570.25     1_597.64       0.3866          1.0891            1.0897         1.23
IVF-PQ-nl223-m16 (self)                                1_027.39     1_844.76     2_872.15       0.3106          1.1231            1.1271         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_679.45       559.68     2_239.12       0.4961          1.0568            1.0521         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_679.45       671.93     2_351.38       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_679.45       961.08     2_640.52       0.4961          1.0568            1.0522         2.00
IVF-PQ-nl223-m32 (self)                                1_679.45     3_193.23     4_872.68       0.4138          1.0784            1.0759         2.00
IVF-PQ-nl223-m64-np11 (query)                          1_986.39       957.63     2_944.02       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np14 (query)                          1_986.39     1_146.15     3_132.54       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64-np21 (query)                          1_986.39     1_669.85     3_656.24       0.6965          1.0200            1.0156         3.52
IVF-PQ-nl223-m64 (self)                                1_986.39     5_520.49     7_506.88       0.6393          1.0273            1.0234         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_100.48       391.96     1_492.44       0.3990          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_100.48       442.34     1_542.82       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_100.48       664.76     1_765.24       0.3989          1.0829            1.0850         1.32
IVF-PQ-nl316-m16 (self)                                1_100.48     2_079.82     3_180.30       0.3170          1.1180            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_712.07       687.08     2_399.15       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_712.07       784.44     2_496.51       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_712.07     1_099.11     2_811.18       0.5103          1.0518            1.0488         2.09
IVF-PQ-nl316-m32 (self)                                1_712.07     3_634.65     5_346.73       0.4239          1.0739            1.0729         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_158.66     1_188.43     3_347.09       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_158.66     1_327.15     3_485.82       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_158.66     1_914.94     4_073.61       0.7083          1.0172            1.0145         3.61
IVF-PQ-nl316-m64 (self)                                2_158.66     6_368.99     8_527.65       0.6491          1.0248            1.0220         3.61
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
Exhaustive (query)                                        74.46     1_336.82     1_411.28       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         74.46     4_282.04     4_356.50       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                982.04       703.50     1_685.54       0.2443          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                                 982.04     2_275.31     3_257.35       0.2277          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_351.88     1_577.08     2_928.96       0.2649          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_351.88     5_184.28     6_536.16       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_301.46     3_711.93     6_013.40       0.2958          1.0991            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_301.46    12_315.52    14_616.99       0.2627          1.1103            1.1143         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_372.18       278.73     1_650.91       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_372.18       454.85     1_827.03       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_372.18       598.66     1_970.84       0.3074          1.0883            1.0928         1.57
IVF-PQ-nl158-m16 (self)                                1_372.18     1_981.26     3_353.44       0.2613          1.1090            1.1156         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_798.97       406.82     2_205.79       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_798.97       633.70     2_432.67       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_798.97       864.31     2_663.28       0.3527          1.0715            1.0730         2.34
IVF-PQ-nl158-m32 (self)                                1_798.97     2_849.70     4_648.67       0.2899          1.0920            1.0958         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_640.84       740.45     3_381.29       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_640.84     1_220.29     3_861.13       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_640.84     1_627.93     4_268.77       0.4644          1.0449            1.0422         3.86
IVF-PQ-nl158-m64 (self)                                2_640.84     5_342.07     7_982.91       0.3928          1.0580            1.0570         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_467.60       432.93     1_900.53       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_467.60       514.13     1_981.73       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_467.60       745.39     2_212.99       0.3166          1.0827            1.0851         1.70
IVF-PQ-nl223-m16 (self)                                1_467.60     2_525.16     3_992.76       0.2659          1.1043            1.1093         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_029.40       623.59     2_653.00       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_029.40       756.23     2_785.63       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_029.40     1_118.84     3_148.24       0.3693          1.0656            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                2_029.40     3_605.32     5_634.72       0.2945          1.0892            1.0915         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_914.20     1_096.47     4_010.68       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_914.20     1_378.61     4_292.81       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_914.20     1_931.68     4_845.89       0.4776          1.0428            1.0385         3.99
IVF-PQ-nl223-m64 (self)                                2_914.20     6_489.40     9_403.60       0.3971          1.0575            1.0549         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_518.23       560.53     2_078.75       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_518.23       615.16     2_133.39       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_518.23       895.49     2_413.72       0.3288          1.0760            1.0804         1.88
IVF-PQ-nl316-m16 (self)                                1_518.23     2_968.90     4_487.13       0.2713          1.0997            1.1062         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_009.28       791.89     2_801.17       0.3797          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_009.28       873.94     2_883.23       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_009.28     1_278.58     3_287.86       0.3798          1.0610            1.0618         2.65
IVF-PQ-nl316-m32 (self)                                2_009.28     4_130.36     6_139.64       0.3003          1.0857            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_777.00     1_371.47     4_148.47       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_777.00     1_535.59     4_312.59       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_777.00     2_215.13     4_992.13       0.4894          1.0396            1.0364         4.17
IVF-PQ-nl316-m64 (self)                                2_777.00     7_346.60    10_123.60       0.4062          1.0539            1.0530         4.17
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
Exhaustive (query)                                       100.67     1_847.67     1_948.35       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.67     6_133.87     6_234.54       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_286.46       745.63     2_032.09       0.2345          1.1095            1.1000         1.51
Exhaustive-PQ-m16 (self)                               1_286.46     2_320.21     3_606.67       0.2206          1.1180            1.1048         1.51
Exhaustive-PQ-m32 (query)                              1_737.07     1_592.70     3_329.77       0.2567          1.0943            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_737.07     5_239.64     6_976.71       0.2391          1.1012            1.1021         2.28
Exhaustive-PQ-m64 (query)                              2_797.69     3_715.33     6_513.02       0.2775          1.0855            1.0910         3.80
Exhaustive-PQ-m64 (self)                               2_797.69    12_296.13    15_093.81       0.2515          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_962.18     8_080.18    13_042.36       0.3162          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_962.18    26_881.89    31_844.07       0.2755          1.0817            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_629.35       364.25     1_993.60       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_629.35       565.34     2_194.69       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_629.35       782.14     2_411.49       0.2852          1.0782            1.0838         1.98
IVF-PQ-nl158-m16 (self)                                1_629.35     2_612.57     4_241.92       0.2512          1.0937            1.1005         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_161.01       551.11     2_712.13       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_161.01       862.92     3_023.93       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_161.01     1_204.32     3_365.34       0.3148          1.0680            1.0713         2.74
IVF-PQ-nl158-m32 (self)                                2_161.01     3_998.09     6_159.10       0.2627          1.0859            1.0912         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_176.05       850.64     4_026.69       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_176.05     1_341.25     4_517.29       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_176.05     1_848.56     5_024.61       0.3771          1.0524            1.0514         4.27
IVF-PQ-nl158-m64 (self)                                3_176.05     6_133.19     9_309.23       0.3100          1.0666            1.0678         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_420.27     1_622.77     7_043.03       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_420.27     2_567.00     7_987.27       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_420.27     3_507.85     8_928.12       0.5325          1.0279            1.0233         7.32
IVF-PQ-nl158-m128 (self)                               5_420.27    11_651.61    17_071.88       0.4627          1.0347            1.0321         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_598.15       537.50     2_135.64       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_598.15       644.21     2_242.36       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_598.15       928.94     2_527.09       0.2962          1.0724            1.0762         2.17
IVF-PQ-nl223-m16 (self)                                1_598.15     3_064.41     4_662.55       0.2554          1.0889            1.0948         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_106.75       762.72     2_869.47       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_106.75       940.40     3_047.15       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_106.75     1_436.55     3_543.30       0.3311          1.0610            1.0634         2.93
IVF-PQ-nl223-m32 (self)                                2_106.75     4_513.62     6_620.38       0.2681          1.0815            1.0863         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_195.47     1_183.49     4_378.96       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_195.47     1_468.18     4_663.64       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_195.47     2_128.51     5_323.98       0.3932          1.0479            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                3_195.47     7_012.35    10_207.82       0.3139          1.0649            1.0653         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_259.58     2_321.15     7_580.73       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_259.58     2_887.42     8_147.00       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_259.58     4_230.03     9_489.61       0.5463          1.0257            1.0212         7.51
IVF-PQ-nl223-m128 (self)                               5_259.58    13_957.89    19_217.47       0.4702          1.0336            1.0308         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_713.48       692.02     2_405.50       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_713.48       738.97     2_452.45       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_713.48     1_065.66     2_779.14       0.3050          1.0680            1.0732         2.44
IVF-PQ-nl316-m16 (self)                                1_713.48     3_502.06     5_215.54       0.2591          1.0854            1.0920         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_231.44       992.24     3_223.67       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_231.44     1_110.52     3_341.95       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_231.44     1_589.86     3_821.30       0.3367          1.0582            1.0610         3.21
IVF-PQ-nl316-m32 (self)                                2_231.44     5_270.04     7_501.47       0.2701          1.0793            1.0841         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_373.07     1_547.25     4_920.31       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_373.07     1_730.45     5_103.52       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_373.07     2_486.94     5_860.01       0.4027          1.0453            1.0435         4.73
IVF-PQ-nl316-m64 (self)                                3_373.07     8_226.58    11_599.65       0.3199          1.0622            1.0632         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_508.82     3_020.48     8_529.30       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_508.82     3_393.11     8_901.93       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_508.82     4_899.88    10_408.70       0.5528          1.0238            1.0205         7.78
IVF-PQ-nl316-m128 (self)                               5_508.82    16_184.53    21_693.35       0.4774          1.0316            1.0300         7.78
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
Exhaustive (query)                                        35.94       737.22       773.16       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         35.94     2_384.08     2_420.02       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                714.26       681.19     1_395.45       0.2932          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 714.26     2_273.06     2_987.32       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                              1_692.79     1_582.81     3_275.60       0.4008          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                               1_692.79     5_202.55     6_895.34       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                              2_035.24     3_736.01     5_771.25       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                               2_035.24    12_376.33    14_411.58       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                             949.05       199.95     1_149.00       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np12 (query)                            949.05       314.80     1_263.85       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16-np17 (query)                            949.05       427.13     1_376.18       0.5346          1.0884            1.0854         1.17
IVF-PQ-nl158-m16 (self)                                  949.05     1_383.69     2_332.73       0.4294          1.1639            1.1599         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_432.56       368.71     1_801.27       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_432.56       585.45     2_018.00       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_432.56       785.20     2_217.76       0.6743          1.0397            1.0375         1.93
IVF-PQ-nl158-m32 (self)                                1_432.56     2_592.87     4_025.43       0.6060          1.0690            1.0642         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_045.37       661.02     2_706.39       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_045.37     1_008.18     3_053.55       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_045.37     1_399.47     3_444.84       0.8335          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                2_045.37     4_730.79     6_776.16       0.7974          1.0165            1.0143         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_019.55       295.61     1_315.15       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_019.55       389.01     1_408.56       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_019.55       534.93     1_554.48       0.5367          1.0874            1.0846         1.23
IVF-PQ-nl223-m16 (self)                                1_019.55     1_804.90     2_824.45       0.4242          1.1680            1.1638         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_628.56       525.65     2_154.21       0.6766          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_628.56       657.03     2_285.59       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_628.56       966.77     2_595.33       0.6767          1.0391            1.0371         2.00
IVF-PQ-nl223-m32 (self)                                1_628.56     3_238.40     4_866.96       0.6033          1.0702            1.0654         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_023.28       917.10     2_940.37       0.8369          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_023.28     1_135.84     3_159.11       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_023.28     1_743.22     3_766.49       0.8370          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                2_023.28     5_607.16     7_630.44       0.8007          1.0159            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_062.26       416.65     1_478.91       0.5369          1.0879            1.0854         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_062.26       445.63     1_507.89       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_062.26       634.42     1_696.68       0.5369          1.0879            1.0853         1.32
IVF-PQ-nl316-m16 (self)                                1_062.26     2_113.91     3_176.17       0.4164          1.1739            1.1698         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_515.29       698.40     2_213.69       0.6804          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_515.29       768.85     2_284.13       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_515.29     1_121.94     2_637.23       0.6805          1.0384            1.0363         2.09
IVF-PQ-nl316-m32 (self)                                1_515.29     3_704.51     5_219.80       0.6013          1.0708            1.0663         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_104.48     1_195.71     3_300.18       0.8382          1.0090            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_104.48     1_321.56     3_426.04       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_104.48     1_931.50     4_035.98       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                2_104.48     6_388.08     8_492.56       0.8022          1.0156            1.0137         3.61
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
Exhaustive (query)                                        68.11     1_335.92     1_404.03       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.11     4_245.64     4_313.75       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                934.08       698.93     1_633.01       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 934.08     2_273.10     3_207.19       0.1772          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_422.40     1_570.26     2_992.66       0.2802          1.1736            1.1699         2.03
Exhaustive-PQ-m32 (self)                               1_422.40     5_173.14     6_595.54       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                              2_356.38     3_704.13     6_060.51       0.3752          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                               2_356.38    12_312.68    14_669.05       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_240.87       269.12     1_509.99       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_240.87       417.90     1_658.76       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_240.87       576.92     1_817.78       0.3795          1.1177            1.1174         1.57
IVF-PQ-nl158-m16 (self)                                1_240.87     1_911.39     3_152.26       0.2739          1.2064            1.2097         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_762.98       411.58     2_174.56       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_762.98       624.80     2_387.79       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_762.98       859.64     2_622.62       0.4921          1.0720            1.0708         2.34
IVF-PQ-nl158-m32 (self)                                1_762.98     2_872.38     4_635.36       0.3946          1.1241            1.1226         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_711.41       726.44     3_437.85       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_711.41     1_259.46     3_970.88       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_711.41     1_679.98     4_391.39       0.6294          1.0352            1.0336         3.86
IVF-PQ-nl158-m64 (self)                                2_711.41     5_301.57     8_012.99       0.5740          1.0544            1.0503         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_253.81       422.25     1_676.05       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_253.81       544.49     1_798.30       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_253.81       741.95     1_995.76       0.3793          1.1177            1.1177         1.70
IVF-PQ-nl223-m16 (self)                                1_253.81     2_477.80     3_731.60       0.2680          1.2120            1.2166         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_728.50       614.27     2_342.77       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_728.50       752.18     2_480.69       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_728.50     1_100.96     2_829.46       0.4920          1.0719            1.0706         2.46
IVF-PQ-nl223-m32 (self)                                1_728.50     3_581.40     5_309.90       0.3847          1.1293            1.1282         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_653.01     1_055.13     3_708.14       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_653.01     1_307.91     3_960.92       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_653.01     1_924.98     4_577.99       0.6330          1.0344            1.0330         3.99
IVF-PQ-nl223-m64 (self)                                2_653.01     6_510.80     9_163.81       0.5727          1.0549            1.0508         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_581.40       540.32     2_121.72       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_581.40       598.08     2_179.49       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_581.40       851.48     2_432.89       0.3778          1.1185            1.1188         1.88
IVF-PQ-nl316-m16 (self)                                1_581.40     2_831.22     4_412.62       0.2621          1.2176            1.2225         1.88
IVF-PQ-nl316-m32-np15 (query)                          1_702.80       774.47     2_477.27       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np17 (query)                          1_702.80       861.47     2_564.27       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32-np25 (query)                          1_702.80     1_232.81     2_935.62       0.4881          1.0728            1.0715         2.65
IVF-PQ-nl316-m32 (self)                                1_702.80     4_045.60     5_748.41       0.3740          1.1347            1.1339         2.65
IVF-PQ-nl316-m64-np15 (query)                          2_568.03     1_352.25     3_920.28       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np17 (query)                          2_568.03     1_518.07     4_086.10       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64-np25 (query)                          2_568.03     2_189.22     4_757.25       0.6352          1.0341            1.0323         4.17
IVF-PQ-nl316-m64 (self)                                2_568.03     7_303.99     9_872.02       0.5697          1.0555            1.0517         4.17
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
Exhaustive (query)                                       100.24     1_855.55     1_955.80       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.24     6_045.88     6_146.12       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_257.04       731.03     1_988.07       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                               1_257.04     2_329.30     3_586.34       0.1758          1.3086            1.3090         1.51
Exhaustive-PQ-m32 (query)                              1_727.76     1_593.92     3_321.68       0.2712          1.1686            1.1636         2.28
Exhaustive-PQ-m32 (self)                               1_727.76     5_211.87     6_939.63       0.2191          1.2527            1.2505         2.28
Exhaustive-PQ-m64 (query)                              2_835.43     3_745.86     6_581.29       0.3546          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                               2_835.43    12_288.15    15_123.58       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             4_848.05     8_064.10    12_912.15       0.4597          1.0781            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_848.05    26_789.00    31_637.06       0.3908          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           1_608.12       369.39     1_977.51       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np12 (query)                          1_608.12       562.38     2_170.51       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16-np17 (query)                          1_608.12       779.68     2_387.80       0.3625          1.1179            1.1170         1.98
IVF-PQ-nl158-m16 (self)                                1_608.12     2_576.40     4_184.53       0.2589          1.2185            1.2225         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_156.49       539.30     2_695.79       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_156.49       859.78     3_016.26       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_156.49     1_180.68     3_337.16       0.4634          1.0766            1.0750         2.74
IVF-PQ-nl158-m32 (self)                                2_156.49     3_967.14     6_123.63       0.3677          1.1383            1.1375         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_212.34       835.44     4_047.78       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_212.34     1_328.42     4_540.75       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_212.34     1_831.81     5_044.15       0.5773          1.0441            1.0423         4.27
IVF-PQ-nl158-m64 (self)                                3_212.34     6_387.78     9_600.12       0.5182          1.0711            1.0673         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_686.95     1_595.80     7_282.75       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_686.95     2_538.80     8_225.76       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_686.95     3_503.46     9_190.41       0.7373          1.0161            1.0143         7.32
IVF-PQ-nl158-m128 (self)                               5_686.95    11_540.70    17_227.65       0.7139          1.0240            1.0194         7.32
IVF-PQ-nl223-m16-np11 (query)                          1_633.82       525.86     2_159.68       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np14 (query)                          1_633.82       630.91     2_264.73       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16-np21 (query)                          1_633.82       922.73     2_556.55       0.3623          1.1174            1.1174         2.17
IVF-PQ-nl223-m16 (self)                                1_633.82     3_097.22     4_731.04       0.2524          1.2254            1.2305         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_206.32       761.84     2_968.17       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_206.32       937.81     3_144.13       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_206.32     1_366.87     3_573.19       0.4603          1.0771            1.0758         2.93
IVF-PQ-nl223-m32 (self)                                2_206.32     4_504.07     6_710.39       0.3495          1.1486            1.1485         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_240.31     1_175.34     4_415.66       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_240.31     1_466.72     4_707.03       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_240.31     2_127.22     5_367.53       0.5749          1.0444            1.0427         4.46
IVF-PQ-nl223-m64 (self)                                3_240.31     7_101.72    10_342.03       0.5036          1.0764            1.0726         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_355.34     2_317.98     7_673.32       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_355.34     2_891.88     8_247.22       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_355.34     4_271.58     9_626.92       0.7405          1.0155            1.0139         7.51
IVF-PQ-nl223-m128 (self)                               5_355.34    13_997.15    19_352.49       0.7136          1.0239            1.0198         7.51
IVF-PQ-nl316-m16-np15 (query)                          1_814.94       692.62     2_507.56       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np17 (query)                          1_814.94       784.23     2_599.17       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16-np25 (query)                          1_814.94     1_117.12     2_932.06       0.3556          1.1201            1.1204         2.44
IVF-PQ-nl316-m16 (self)                                1_814.94     3_543.00     5_357.94       0.2447          1.2328            1.2393         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_362.36       994.43     3_356.80       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_362.36     1_110.45     3_472.81       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_362.36     1_595.63     3_957.99       0.4552          1.0787            1.0773         3.21
IVF-PQ-nl316-m32 (self)                                2_362.36     5_298.77     7_661.13       0.3316          1.1584            1.1592         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_369.68     1_547.06     4_916.75       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_369.68     1_732.86     5_102.54       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_369.68     2_497.67     5_867.35       0.5743          1.0447            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                3_369.68     8_270.04    11_639.72       0.4868          1.0819            1.0783         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_623.09     3_026.26     8_649.35       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_623.09     3_385.72     9_008.82       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_623.09     4_884.90    10_508.00       0.7419          1.0153            1.0138         7.78
IVF-PQ-nl316-m128 (self)                               5_623.09    16_227.63    21_850.72       0.7111          1.0240            1.0202         7.78
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
Exhaustive (query)                                        35.25       737.75       773.00       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         35.25     2_402.72     2_437.97       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                876.99       726.80     1_603.79       0.7118          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 876.99     2_641.80     3_518.79       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_669.78     1_724.11     3_393.89       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_669.78     5_332.82     7_002.60       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              4_296.83     3_688.85     7_985.68       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               4_296.83    12_321.96    16_618.79       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_193.66       223.08     1_416.74       0.8274          1.0521            1.0445         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_193.66       349.23     1_542.88       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_193.66       487.25     1_680.91       0.8279          1.0518            1.0443         1.17
IVF-PQ-nl158-m16 (self)                                1_193.66     1_574.72     2_768.38       0.7673          1.0987            1.0833         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_522.85       406.54     1_929.38       0.8739          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_522.85       675.37     2_198.22       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_522.85       935.28     2_458.13       0.8744          1.0263            1.0216         1.93
IVF-PQ-nl158-m32 (self)                                1_522.85     3_089.99     4_612.83       0.8286          1.0513            1.0425         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_188.70       730.61     2_919.31       0.9048          1.0151            1.0116         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_188.70     1_247.56     3_436.27       0.9055          1.0148            1.0113         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_188.70     1_769.33     3_958.04       0.9055          1.0147            1.0113         3.46
IVF-PQ-nl158-m64 (self)                                2_188.70     5_890.37     8_079.08       0.8704          1.0287            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_243.08       315.01     1_558.09       0.8421          1.0435            1.0371         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_243.08       394.87     1_637.95       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_243.08       575.67     1_818.75       0.8422          1.0435            1.0370         1.23
IVF-PQ-nl223-m16 (self)                                1_243.08     1_968.82     3_211.90       0.7842          1.0842            1.0703         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_596.78       553.75     2_150.53       0.8836          1.0226            1.0184         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_596.78       702.96     2_299.74       0.8837          1.0225            1.0184         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_596.78     1_034.18     2_630.96       0.8838          1.0225            1.0184         2.00
IVF-PQ-nl223-m32 (self)                                1_596.78     3_456.08     5_052.87       0.8398          1.0442            1.0357         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_220.93       989.51     3_210.45       0.9103          1.0133            1.0100         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_220.93     1_270.18     3_491.12       0.9104          1.0132            1.0100         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_220.93     1_862.46     4_083.39       0.9105          1.0132            1.0100         3.52
IVF-PQ-nl223-m64 (self)                                2_220.93     6_224.05     8_444.98       0.8772          1.0255            1.0197         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_285.06       404.85     1_689.91       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_285.06       445.10     1_730.15       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_285.06       642.92     1_927.98       0.8494          1.0393            1.0336         1.32
IVF-PQ-nl316-m16 (self)                                1_285.06     2_143.40     3_428.46       0.7916          1.0787            1.0640         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_706.86       703.91     2_410.77       0.8864          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_706.86       791.78     2_498.64       0.8865          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_706.86     1_159.30     2_866.17       0.8865          1.0213            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_706.86     4_095.54     5_802.40       0.8427          1.0431            1.0342         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_348.08     1_245.94     3_594.02       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_348.08     1_392.28     3_740.36       0.9128          1.0124            1.0094         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_348.08     2_032.39     4_380.47       0.9129          1.0124            1.0094         3.61
IVF-PQ-nl316-m64 (self)                                2_348.08     6_756.87     9_104.95       0.8794          1.0245            1.0188         3.61
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
Exhaustive (query)                                        69.32     1_296.59     1_365.91       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.32     4_223.69     4_293.01       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                938.60       693.26     1_631.86       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 938.60     2_260.56     3_199.16       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_376.20     1_560.78     2_936.99       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_376.20     5_163.82     6_540.02       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_316.05     3_721.81     6_037.86       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_316.05    12_292.02    14_608.07       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           1_440.67       277.84     1_718.51       0.8367          1.0503            1.0400         1.57
IVF-PQ-nl158-m16-np12 (query)                          1_440.67       449.75     1_890.42       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16-np17 (query)                          1_440.67       612.61     2_053.28       0.8370          1.0501            1.0398         1.57
IVF-PQ-nl158-m16 (self)                                1_440.67     2_023.31     3_463.98       0.7723          1.1013            1.0737         1.57
IVF-PQ-nl158-m32-np7 (query)                           1_890.65       432.17     2_322.82       0.8684          1.0311            1.0245         2.34
IVF-PQ-nl158-m32-np12 (query)                          1_890.65       706.35     2_597.00       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32-np17 (query)                          1_890.65       990.55     2_881.19       0.8687          1.0310            1.0244         2.34
IVF-PQ-nl158-m32 (self)                                1_890.65     3_225.87     5_116.52       0.8159          1.0642            1.0463         2.34
IVF-PQ-nl158-m64-np7 (query)                           2_971.31       806.10     3_777.41       0.8900          1.0215            1.0164         3.86
IVF-PQ-nl158-m64-np12 (query)                          2_971.31     1_356.19     4_327.50       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64-np17 (query)                          2_971.31     1_905.00     4_876.31       0.8903          1.0214            1.0164         3.86
IVF-PQ-nl158-m64 (self)                                2_971.31     6_285.69     9_257.00       0.8456          1.0436            1.0313         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_476.67       422.37     1_899.04       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_476.67       513.59     1_990.27       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_476.67       771.22     2_247.89       0.8549          1.0394            1.0308         1.70
IVF-PQ-nl223-m16 (self)                                1_476.67     2_522.23     3_998.90       0.7969          1.0803            1.0562         1.70
IVF-PQ-nl223-m32-np11 (query)                          1_953.24       623.68     2_576.92       0.8796          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          1_953.24       796.72     2_749.96       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          1_953.24     1_153.50     3_106.74       0.8797          1.0268            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                1_953.24     3_817.70     5_770.94       0.8298          1.0557            1.0378         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_864.99     1_125.39     3_990.38       0.9001          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_864.99     1_425.46     4_290.46       0.9002          1.0179            1.0132         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_864.99     2_112.67     4_977.66       0.9002          1.0178            1.0132         3.99
IVF-PQ-nl223-m64 (self)                                2_864.99     6_962.62     9_827.61       0.8569          1.0374            1.0254         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_603.32       554.42     2_157.74       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_603.32       612.51     2_215.83       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_603.32       878.65     2_481.97       0.8697          1.0320            1.0251         1.88
IVF-PQ-nl316-m16 (self)                                1_603.32     2_970.70     4_574.01       0.8151          1.0656            1.0459         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_106.92       792.22     2_899.14       0.8921          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_106.92       884.89     2_991.81       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_106.92     1_292.82     3_399.75       0.8922          1.0214            1.0161         2.65
IVF-PQ-nl316-m32 (self)                                2_106.92     4_229.65     6_336.57       0.8453          1.0453            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_111.07     1_400.93     4_511.99       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_111.07     1_582.35     4_693.42       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_111.07     2_340.55     5_451.62       0.9073          1.0152            1.0111         4.17
IVF-PQ-nl316-m64 (self)                                3_111.07     7_678.88    10_789.95       0.8660          1.0332            1.0218         4.17
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
Exhaustive (query)                                       100.57     1_850.46     1_951.03       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.57     6_126.20     6_226.77       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_238.48       720.70     1_959.18       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_238.48     2_302.18     3_540.66       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                              1_763.08     1_589.83     3_352.91       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                               1_763.08     5_429.26     7_192.34       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                              2_847.70     3_712.24     6_559.94       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                               2_847.70    12_384.80    15_232.50       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             5_013.57     8_182.14    13_195.71       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              5_013.57    26_865.34    31_878.91       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           2_039.77       403.08     2_442.85       0.8519          1.0423            1.0326         1.98
IVF-PQ-nl158-m16-np12 (query)                          2_039.77       605.37     2_645.15       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16-np17 (query)                          2_039.77       858.84     2_898.62       0.8520          1.0422            1.0325         1.98
IVF-PQ-nl158-m16 (self)                                2_039.77     2_731.91     4_771.68       0.7910          1.0832            1.0598         1.98
IVF-PQ-nl158-m32-np7 (query)                           2_587.08       572.59     3_159.67       0.9001          1.0207            1.0128         2.74
IVF-PQ-nl158-m32-np12 (query)                          2_587.08       928.57     3_515.65       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32-np17 (query)                          2_587.08     1_298.49     3_885.58       0.9003          1.0206            1.0127         2.74
IVF-PQ-nl158-m32 (self)                                2_587.08     4_341.77     6_928.86       0.8549          1.0435            1.0236         2.74
IVF-PQ-nl158-m64-np7 (query)                           3_642.70       916.32     4_559.02       0.9204          1.0131            1.0070         4.27
IVF-PQ-nl158-m64-np12 (query)                          3_642.70     1_546.37     5_189.06       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64-np17 (query)                          3_642.70     2_129.99     5_772.69       0.9205          1.0130            1.0070         4.27
IVF-PQ-nl158-m64 (self)                                3_642.70     7_074.81    10_717.51       0.8830          1.0284            1.0134         4.27
IVF-PQ-nl158-m128-np7 (query)                          5_718.18     1_783.29     7_501.46       0.9394          1.0072            1.0031         7.32
IVF-PQ-nl158-m128-np12 (query)                         5_718.18     2_974.98     8_693.15       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128-np17 (query)                         5_718.18     4_153.95     9_872.13       0.9396          1.0071            1.0030         7.32
IVF-PQ-nl158-m128 (self)                               5_718.18    13_741.10    19_459.28       0.9073          1.0171            1.0071         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_141.81       529.86     2_671.66       0.8626          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_141.81       647.19     2_788.99       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_141.81       950.07     3_091.87       0.8627          1.0360            1.0281         2.17
IVF-PQ-nl223-m16 (self)                                2_141.81     3_140.34     5_282.15       0.8067          1.0699            1.0512         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_864.17       809.64     3_673.81       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_864.17       980.66     3_844.84       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_864.17     1_465.56     4_329.73       0.9089          1.0172            1.0105         2.93
IVF-PQ-nl223-m32 (self)                                2_864.17     4_742.21     7_606.39       0.8676          1.0353            1.0194         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_765.69     1_250.73     5_016.42       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_765.69     1_552.72     5_318.41       0.9269          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_765.69     2_292.04     6_057.73       0.9269          1.0111            1.0056         4.46
IVF-PQ-nl223-m64 (self)                                3_765.69     7_573.18    11_338.87       0.8921          1.0237            1.0111         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_758.63     2_456.90     8_215.53       0.9441          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_758.63     3_084.89     8_843.52       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_758.63     4_592.46    10_351.09       0.9442          1.0059            1.0023         7.51
IVF-PQ-nl223-m128 (self)                               5_758.63    15_222.69    20_981.32       0.9134          1.0147            1.0057         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_639.58       689.81     3_329.40       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_639.58       754.61     3_394.19       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_639.58     1_085.28     3_724.86       0.8690          1.0322            1.0252         2.44
IVF-PQ-nl316-m16 (self)                                2_639.58     3_621.64     6_261.23       0.8142          1.0645            1.0463         2.44
IVF-PQ-nl316-m32-np15 (query)                          2_994.01     1_017.28     4_011.29       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32-np17 (query)                          2_994.01     1_143.38     4_137.39       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32-np25 (query)                          2_994.01     1_663.92     4_657.93       0.9130          1.0153            1.0092         3.21
IVF-PQ-nl316-m32 (self)                                2_994.01     5_502.12     8_496.13       0.8728          1.0330            1.0175         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_961.90     1_603.50     5_565.40       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_961.90     1_806.05     5_767.95       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_961.90     2_619.11     6_581.01       0.9306          1.0098            1.0050         4.73
IVF-PQ-nl316-m64 (self)                                3_961.90     8_692.75    12_654.65       0.8963          1.0221            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_230.42     3_141.84     9_372.26       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_230.42     3_529.89     9_760.31       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_230.42     5_159.61    11_390.03       0.9461          1.0054            1.0020         7.78
IVF-PQ-nl316-m128 (self)                               6_230.42    17_109.57    23_339.99       0.9166          1.0138            1.0052         7.78
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
Exhaustive (query)                                        33.11       691.79       724.90       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.11     2_243.11     2_276.22       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_720.03       739.62     4_459.65       0.2865          1.1528            1.1332         1.26
Exhaustive-OPQ-m16 (self)                              3_720.03     2_746.37     6_466.39       0.2585          1.1711            1.1497         1.26
Exhaustive-OPQ-m32 (query)                             6_055.87     1_634.67     7_690.54       0.3260          1.1208            1.1171         2.03
Exhaustive-OPQ-m32 (self)                              6_055.87     5_632.60    11_688.47       0.2831          1.1440            1.1382         2.03
Exhaustive-OPQ-m64 (query)                            10_001.09     3_817.52    13_818.61       0.3797          1.0983            1.0951         3.55
Exhaustive-OPQ-m64 (self)                             10_001.09    12_738.48    22_739.57       0.3219          1.1205            1.1171         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_817.49       270.97     4_088.47       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_817.49       380.36     4_197.86       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_817.49       503.54     4_321.03       0.3868          1.0890            1.0908         1.67
IVF-OPQ-nl158-m16 (self)                               3_817.49     2_013.32     5_830.81       0.3169          1.1190            1.1234         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_103.22       436.90     6_540.12       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_103.22       651.78     6_755.00       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_103.22       862.50     6_965.72       0.4917          1.0563            1.0549         2.43
IVF-OPQ-nl158-m32 (self)                               6_103.22     3_267.62     9_370.84       0.4164          1.0763            1.0768         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_429.07       744.20    10_173.27       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_429.07     1_106.49    10_535.56       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_429.07     1_487.67    10_916.74       0.6952          1.0189            1.0161         3.96
IVF-OPQ-nl158-m64 (self)                               9_429.07     5_235.28    14_664.35       0.6380          1.0262            1.0237         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_002.48       371.58     4_374.05       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_002.48       445.05     4_447.52       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_002.48       649.77     4_652.25       0.3981          1.0836            1.0849         1.73
IVF-OPQ-nl223-m16 (self)                               4_002.48     2_436.13     6_438.61       0.3210          1.1162            1.1203         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_445.92       607.49     7_053.41       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_445.92       743.39     7_189.30       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_445.92     1_058.52     7_504.44       0.5047          1.0531            1.0505         2.50
IVF-OPQ-nl223-m32 (self)                               6_445.92     3_876.53    10_322.45       0.4230          1.0744            1.0737         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_756.14     1_032.83    10_788.97       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_756.14     1_247.69    11_003.83       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_756.14     1_790.29    11_546.43       0.7020          1.0183            1.0152         4.02
IVF-OPQ-nl223-m64 (self)                               9_756.14     6_324.79    16_080.93       0.6451          1.0255            1.0227         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_057.29       461.60     4_518.88       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_057.29       517.39     4_574.68       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_057.29       715.78     4_773.06       0.4071          1.0797            1.0811         2.07
IVF-OPQ-nl316-m16 (self)                               4_057.29     2_735.50     6_792.78       0.3260          1.1127            1.1171         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_569.83       773.46     7_343.30       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_569.83       858.94     7_428.78       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_569.83     1_220.44     7_790.27       0.5171          1.0487            1.0475         2.84
IVF-OPQ-nl316-m32 (self)                               6_569.83     4_381.20    10_951.03       0.4329          1.0702            1.0708         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_837.68     1_307.11    11_144.80       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_837.68     1_434.26    11_271.94       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_837.68     2_044.63    11_882.31       0.7098          1.0164            1.0144         4.36
IVF-OPQ-nl316-m64 (self)                               9_837.68     7_066.45    16_904.13       0.6520          1.0238            1.0217         4.36
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
Exhaustive (query)                                        68.07     1_353.08     1_421.15       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.07     4_459.42     4_527.49       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_077.48     1_041.51     7_119.00       0.2659          1.1129            1.1008         2.26
Exhaustive-OPQ-m16 (self)                              6_077.48     4_865.74    10_943.23       0.2458          1.1245            1.1094         2.26
Exhaustive-OPQ-m32 (query)                             8_273.94     1_935.53    10_209.48       0.2865          1.0971            1.0962         3.03
Exhaustive-OPQ-m32 (self)                              8_273.94     7_637.51    15_911.45       0.2608          1.1083            1.1060         3.03
Exhaustive-OPQ-m64 (query)                            13_307.38     4_053.33    17_360.71       0.3196          1.0822            1.0844         4.55
Exhaustive-OPQ-m64 (self)                             13_307.38    14_609.36    27_916.75       0.2789          1.0967            1.0990         4.55
Exhaustive-OPQ-m128 (query)                           20_176.97     8_372.01    28_548.98       0.3687          1.0680            1.0687         7.61
Exhaustive-OPQ-m128 (self)                            20_176.97    29_310.69    49_487.66       0.3154          1.0825            1.0831         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_273.18       622.74     6_895.92       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_273.18       760.62     7_033.80       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_273.18       926.12     7_199.30       0.3242          1.0787            1.0828         3.07
IVF-OPQ-nl158-m16 (self)                               6_273.18     4_596.10    10_869.28       0.2764          1.0979            1.1033         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_445.85       753.77     9_199.62       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_445.85       977.00     9_422.85       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_445.85     1_228.85     9_674.70       0.3680          1.0655            1.0674         3.84
IVF-OPQ-nl158-m32 (self)                               8_445.85     5_553.50    13_999.34       0.3020          1.0859            1.0898         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_210.16     1_091.05    14_301.22       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_210.16     1_521.16    14_731.32       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_210.16     1_967.28    15_177.44       0.4758          1.0410            1.0402         5.36
IVF-OPQ-nl158-m64 (self)                              13_210.16     8_034.87    21_245.04       0.4021          1.0546            1.0550         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_630.97     1_643.29    21_274.25       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_630.97     2_378.62    22_009.58       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_630.97     3_103.52    22_734.48       0.6832          1.0139            1.0117         8.42
IVF-OPQ-nl158-m128 (self)                             19_630.97    11_906.78    31_537.75       0.6253          1.0192            1.0171         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_412.61       741.43     7_154.04       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_412.61       815.28     7_227.88       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_412.61     1_044.23     7_456.84       0.3304          1.0760            1.0795         3.20
IVF-OPQ-nl223-m16 (self)                               6_412.61     4_876.14    11_288.75       0.2769          1.0973            1.1025         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_683.54       936.17     9_619.70       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_683.54     1_070.08     9_753.62       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_683.54     1_388.99    10_072.52       0.3794          1.0619            1.0629         3.96
IVF-OPQ-nl223-m32 (self)                               8_683.54     6_159.91    14_843.44       0.3038          1.0847            1.0880         3.96
IVF-OPQ-nl223-m64-np11 (query)                        13_583.21     1_415.56    14_998.77       0.4869          1.0387            1.0372         5.49
IVF-OPQ-nl223-m64-np14 (query)                        13_583.21     1_678.88    15_262.09       0.4869          1.0386            1.0372         5.49
IVF-OPQ-nl223-m64-np21 (query)                        13_583.21     2_302.19    15_885.40       0.4869          1.0386            1.0372         5.49
IVF-OPQ-nl223-m64 (self)                              13_583.21     9_039.16    22_622.37       0.4074          1.0533            1.0532         5.49
IVF-OPQ-nl223-m128-np11 (query)                       20_526.90     2_206.86    22_733.76       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128-np14 (query)                       20_526.90     2_653.18    23_180.09       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128-np21 (query)                       20_526.90     3_691.65    24_218.55       0.6893          1.0134            1.0112         8.54
IVF-OPQ-nl223-m128 (self)                             20_526.90    13_913.77    34_440.68       0.6310          1.0188            1.0165         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_487.41       847.17     7_334.58       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_487.41       903.42     7_390.83       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_487.41     1_156.33     7_643.74       0.3363          1.0726            1.0764         3.88
IVF-OPQ-nl316-m16 (self)                               6_487.41     5_269.14    11_756.55       0.2799          1.0946            1.1004         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_841.86     1_141.72     9_983.58       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_841.86     1_202.20    10_044.07       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_841.86     1_565.38    10_407.24       0.3878          1.0579            1.0597         4.65
IVF-OPQ-nl316-m32 (self)                               8_841.86     6_652.73    15_494.59       0.3088          1.0817            1.0856         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_740.08     1_752.05    15_492.13       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_740.08     1_919.65    15_659.73       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_740.08     2_624.00    16_364.08       0.4984          1.0360            1.0352         6.17
IVF-OPQ-nl316-m64 (self)                              13_740.08    10_222.33    23_962.42       0.4153          1.0508            1.0515         6.17
IVF-OPQ-nl316-m128-np15 (query)                       20_560.75     2_735.18    23_295.94       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np17 (query)                       20_560.75     3_033.80    23_594.55       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np25 (query)                       20_560.75     4_186.78    24_747.54       0.6971          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128 (self)                             20_560.75    15_508.51    36_069.26       0.6400          1.0173            1.0157         9.23
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
Exhaustive (query)                                       100.31     1_922.90     2_023.21       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.31     6_324.63     6_424.95       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_836.70     1_650.76    11_487.46       0.2602          1.0925            1.0831         3.76
Exhaustive-OPQ-m16 (self)                              9_836.70     8_460.82    18_297.52       0.2417          1.1021            1.0892         3.76
Exhaustive-OPQ-m32 (query)                            12_428.42     2_443.94    14_872.36       0.2792          1.0787            1.0797         4.53
Exhaustive-OPQ-m32 (self)                             12_428.42    11_223.40    23_651.82       0.2568          1.0877            1.0870         4.53
Exhaustive-OPQ-m64 (query)                            17_660.89     4_573.03    22_233.92       0.2961          1.0725            1.0758         6.05
Exhaustive-OPQ-m64 (self)                             17_660.89    18_424.74    36_085.63       0.2656          1.0825            1.0855         6.05
Exhaustive-OPQ-m128 (query)                           27_750.95     8_908.17    36_659.12       0.3311          1.0632            1.0658         9.11
Exhaustive-OPQ-m128 (self)                            27_750.95    33_023.51    60_774.46       0.2834          1.0754            1.0784         9.11
IVF-OPQ-nl158-m16-np7 (query)                          9_955.22     1_185.04    11_140.26       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16-np12 (query)                         9_955.22     1_383.33    11_338.55       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16-np17 (query)                         9_955.22     1_611.61    11_566.83       0.3030          1.0686            1.0739         4.98
IVF-OPQ-nl158-m16 (self)                               9_955.22     8_492.84    18_448.06       0.2665          1.0821            1.0878         4.98
IVF-OPQ-nl158-m32-np7 (query)                         12_691.61     1_414.80    14_106.41       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32-np12 (query)                        12_691.61     1_666.05    14_357.66       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32-np17 (query)                        12_691.61     1_957.53    14_649.14       0.3301          1.0611            1.0640         5.74
IVF-OPQ-nl158-m32 (self)                              12_691.61     9_833.01    22_524.63       0.2747          1.0784            1.0827         5.74
IVF-OPQ-nl158-m64-np7 (query)                         18_304.13     1_686.69    19_990.82       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64-np12 (query)                        18_304.13     2_178.83    20_482.96       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64-np17 (query)                        18_304.13     2_647.18    20_951.31       0.3910          1.0477            1.0479         7.27
IVF-OPQ-nl158-m64 (self)                              18_304.13    12_898.34    31_202.47       0.3209          1.0625            1.0645         7.27
IVF-OPQ-nl158-m128-np7 (query)                        28_369.35     2_909.34    31_278.69       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128-np12 (query)                       28_369.35     3_580.97    31_950.32       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128-np17 (query)                       28_369.35     4_768.97    33_138.32       0.5418          1.0249            1.0224        10.32
IVF-OPQ-nl158-m128 (self)                             28_369.35    18_052.67    46_422.02       0.4722          1.0322            1.0310        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_398.75     1_331.54    11_730.29       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_398.75     1_444.51    11_843.26       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_398.75     1_690.39    12_089.14       0.3094          1.0662            1.0704         5.17
IVF-OPQ-nl223-m16 (self)                              10_398.75     8_936.81    19_335.56       0.2676          1.0813            1.0870         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_909.17     1_603.17    14_512.34       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_909.17     1_805.15    14_714.31       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_909.17     2_209.22    15_118.38       0.3395          1.0580            1.0603         5.93
IVF-OPQ-nl223-m32 (self)                              12_909.17    10_693.93    23_603.10       0.2763          1.0771            1.0817         5.93
IVF-OPQ-nl223-m64-np11 (query)                        18_619.79     2_057.62    20_677.41       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64-np14 (query)                        18_619.79     2_350.10    20_969.89       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64-np21 (query)                        18_619.79     3_030.99    21_650.78       0.4028          1.0443            1.0444         7.46
IVF-OPQ-nl223-m64 (self)                              18_619.79    13_390.22    32_010.01       0.3239          1.0609            1.0630         7.46
IVF-OPQ-nl223-m128-np11 (query)                       28_795.29     3_236.21    32_031.50       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128-np14 (query)                       28_795.29     3_819.52    32_614.81       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128-np21 (query)                       28_795.29     5_169.09    33_964.39       0.5546          1.0226            1.0208        10.51
IVF-OPQ-nl223-m128 (self)                             28_795.29    20_528.94    49_324.23       0.4799          1.0308            1.0299        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_574.55     1_488.39    12_062.94       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_574.55     1_557.55    12_132.10       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_574.55     1_840.39    12_414.93       0.3131          1.0645            1.0690         6.19
IVF-OPQ-nl316-m16 (self)                              10_574.55     9_447.67    20_022.22       0.2690          1.0802            1.0862         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_126.96     1_848.34    14_975.30       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_126.96     1_959.48    15_086.44       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_126.96     2_429.95    15_556.91       0.3451          1.0557            1.0586         6.96
IVF-OPQ-nl316-m32 (self)                              13_126.96    11_398.83    24_525.80       0.2788          1.0756            1.0804         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_497.61     2_430.91    20_928.52       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_497.61     2_619.17    21_116.78       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_497.61     3_440.34    21_937.95       0.4107          1.0423            1.0426         8.48
IVF-OPQ-nl316-m64 (self)                              18_497.61    14_744.61    33_242.22       0.3275          1.0596            1.0619         8.48
IVF-OPQ-nl316-m128-np15 (query)                       28_855.19     3_951.16    32_806.35       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128-np17 (query)                       28_855.19     4_328.70    33_183.89       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128-np25 (query)                       28_855.19     5_956.48    34_811.67       0.5630          1.0212            1.0198        11.54
IVF-OPQ-nl316-m128 (self)                             28_855.19    22_897.82    51_753.01       0.4862          1.0295            1.0291        11.54
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
Exhaustive (query)                                        33.56       740.95       774.51       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.56     2_428.92     2_462.48       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_783.33       738.90     4_522.23       0.3009          1.2503            1.2433         1.26
Exhaustive-OPQ-m16 (self)                              3_783.33     2_753.09     6_536.42       0.2368          1.3778            1.3714         1.26
Exhaustive-OPQ-m32 (query)                             6_173.21     1_602.95     7_776.16       0.4204          1.1526            1.1474         2.03
Exhaustive-OPQ-m32 (self)                              6_173.21     5_603.47    11_776.68       0.3378          1.2478            1.2416         2.03
Exhaustive-OPQ-m64 (query)                             9_580.62     3_740.99    13_321.61       0.5662          1.0765            1.0733         3.55
Exhaustive-OPQ-m64 (self)                              9_580.62    12_768.05    22_348.67       0.4876          1.1287            1.1241         3.55
IVF-OPQ-nl158-m16-np7 (query)                          3_778.23       271.75     4_049.98       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np12 (query)                         3_778.23       387.33     4_165.57       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np17 (query)                         3_778.23       514.69     4_292.93       0.6992          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16 (self)                               3_778.23     2_050.76     5_829.00       0.6197          1.0627            1.0600         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_251.05       432.06     6_683.10       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_251.05       652.26     6_903.31       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_251.05       883.30     7_134.34       0.7983          1.0140            1.0128         2.43
IVF-OPQ-nl158-m32 (self)                               6_251.05     3_318.54     9_569.58       0.7470          1.0260            1.0240         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_254.93       714.99     9_969.92       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_254.93     1_115.17    10_370.10       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_254.93     1_565.91    10_820.84       0.8603          1.0065            1.0056         3.96
IVF-OPQ-nl158-m64 (self)                               9_254.93     5_355.30    14_610.23       0.8304          1.0113            1.0097         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_966.38       365.01     4_331.39       0.7067          1.0311            1.0296         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_966.38       437.22     4_403.60       0.7068          1.0311            1.0295         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_966.38       624.92     4_591.30       0.7068          1.0311            1.0295         1.73
IVF-OPQ-nl223-m16 (self)                               3_966.38     2_420.45     6_386.83       0.6288          1.0595            1.0570         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_256.23       601.66     6_857.88       0.8046          1.0131            1.0120         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_256.23       738.82     6_995.04       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_256.23     1_067.17     7_323.39       0.8047          1.0131            1.0119         2.50
IVF-OPQ-nl223-m32 (self)                               6_256.23     3_885.40    10_141.63       0.7549          1.0243            1.0225         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_608.40     1_001.64    10_610.03       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_608.40     1_234.71    10_843.11       0.8638          1.0061            1.0051         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_608.40     1_809.89    11_418.28       0.8638          1.0061            1.0051         4.02
IVF-OPQ-nl223-m64 (self)                               9_608.40     6_297.29    15_905.68       0.8352          1.0106            1.0092         4.02
IVF-OPQ-nl316-m16-np15 (query)                         3_955.84       455.26     4_411.10       0.7112          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16-np17 (query)                         3_955.84       524.58     4_480.42       0.7113          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16-np25 (query)                         3_955.84       699.49     4_655.32       0.7113          1.0300            1.0286         2.07
IVF-OPQ-nl316-m16 (self)                               3_955.84     2_697.95     6_653.79       0.6343          1.0576            1.0549         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_584.23       772.34     7_356.57       0.8076          1.0128            1.0115         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_584.23       857.73     7_441.97       0.8077          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_584.23     1_197.36     7_781.59       0.8077          1.0127            1.0115         2.84
IVF-OPQ-nl316-m32 (self)                               6_584.23     4_287.39    10_871.62       0.7581          1.0237            1.0219         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_756.54     1_276.11    11_032.65       0.8662          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_756.54     1_432.66    11_189.20       0.8663          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_756.54     2_064.90    11_821.44       0.8663          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               9_756.54     7_171.66    16_928.20       0.8373          1.0103            1.0090         4.36
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
Exhaustive (query)                                        68.41     1_304.23     1_372.64       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.41     4_353.39     4_421.80       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_280.10     1_063.59     7_343.69       0.2317          1.2142            1.2107         2.26
Exhaustive-OPQ-m16 (self)                              6_280.10     4_887.17    11_167.27       0.1879          1.2983            1.2985         2.26
Exhaustive-OPQ-m32 (query)                             8_520.24     1_946.91    10_467.15       0.3189          1.1505            1.1472         3.03
Exhaustive-OPQ-m32 (self)                              8_520.24     7_650.75    16_170.99       0.2588          1.2171            1.2145         3.03
Exhaustive-OPQ-m64 (query)                            13_479.43     4_054.61    17_534.04       0.4332          1.0939            1.0912         4.55
Exhaustive-OPQ-m64 (self)                             13_479.43    14_605.68    28_085.11       0.3620          1.1417            1.1393         4.55
Exhaustive-OPQ-m128 (query)                           19_775.56     8_387.70    28_163.26       0.5699          1.0489            1.0476         7.61
Exhaustive-OPQ-m128 (self)                            19_775.56    29_391.54    49_167.11       0.4998          1.0773            1.0755         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_224.99       612.00     6_836.99       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_224.99       765.23     6_990.23       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_224.99       926.75     7_151.74       0.5408          1.0562            1.0551         3.07
IVF-OPQ-nl158-m16 (self)                               6_224.99     4_562.87    10_787.86       0.4370          1.1017            1.0996         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_585.01       751.39     9_336.40       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_585.01       973.03     9_558.04       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_585.01     1_216.66     9_801.67       0.6840          1.0247            1.0234         3.84
IVF-OPQ-nl158-m32 (self)                               8_585.01     5_550.07    14_135.08       0.6083          1.0441            1.0419         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_574.04     1_077.28    14_651.33       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_574.04     1_505.70    15_079.74       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_574.04     1_952.38    15_526.42       0.7789          1.0115            1.0105         5.36
IVF-OPQ-nl158-m64 (self)                              13_574.04     8_013.24    21_587.28       0.7308          1.0199            1.0179         5.36
IVF-OPQ-nl158-m128-np7 (query)                        20_177.59     1_632.80    21_810.39       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128-np12 (query)                       20_177.59     2_407.64    22_585.23       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128-np17 (query)                       20_177.59     3_095.03    23_272.62       0.8379          1.0061            1.0052         8.42
IVF-OPQ-nl158-m128 (self)                             20_177.59    11_849.71    32_027.30       0.8095          1.0099            1.0081         8.42
IVF-OPQ-nl223-m16-np11 (query)                         7_745.69       753.87     8_499.56       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16-np14 (query)                         7_745.69       824.01     8_569.70       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16-np21 (query)                         7_745.69     1_033.75     8_779.44       0.5476          1.0540            1.0527         3.20
IVF-OPQ-nl223-m16 (self)                               7_745.69     4_881.39    12_627.08       0.4482          1.0970            1.0949         3.20
IVF-OPQ-nl223-m32-np11 (query)                         9_914.54       942.28    10_856.82       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32-np14 (query)                         9_914.54     1_072.74    10_987.28       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32-np21 (query)                         9_914.54     1_409.37    11_323.91       0.6898          1.0235            1.0223         3.96
IVF-OPQ-nl223-m32 (self)                               9_914.54     6_130.24    16_044.78       0.6178          1.0417            1.0398         3.96
IVF-OPQ-nl223-m64-np11 (query)                        15_071.66     1_426.56    16_498.22       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64-np14 (query)                        15_071.66     1_703.39    16_775.05       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64-np21 (query)                        15_071.66     2_306.56    17_378.22       0.7833          1.0110            1.0099         5.49
IVF-OPQ-nl223-m64 (self)                              15_071.66     9_164.03    24_235.69       0.7366          1.0189            1.0172         5.49
IVF-OPQ-nl223-m128-np11 (query)                       20_743.03     2_481.39    23_224.42       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np14 (query)                       20_743.03     2_824.35    23_567.38       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np21 (query)                       20_743.03     4_181.64    24_924.67       0.8410          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128 (self)                             20_743.03    13_844.80    34_587.83       0.8130          1.0094            1.0079         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_936.40       897.28     7_833.68       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_936.40       919.21     7_855.61       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_936.40     1_167.40     8_103.80       0.5516          1.0531            1.0518         3.88
IVF-OPQ-nl316-m16 (self)                               6_936.40     5_310.86    12_247.26       0.4530          1.0950            1.0932         3.88
IVF-OPQ-nl316-m32-np15 (query)                        10_046.44     1_114.65    11_161.09       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32-np17 (query)                        10_046.44     1_207.96    11_254.40       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32-np25 (query)                        10_046.44     1_619.01    11_665.45       0.6930          1.0231            1.0219         4.65
IVF-OPQ-nl316-m32 (self)                              10_046.44     6_664.33    16_710.77       0.6213          1.0409            1.0391         4.65
IVF-OPQ-nl316-m64-np15 (query)                        15_209.40     1_748.90    16_958.30       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np17 (query)                        15_209.40     1_908.28    17_117.68       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np25 (query)                        15_209.40     2_622.96    17_832.36       0.7851          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64 (self)                              15_209.40    10_194.15    25_403.55       0.7386          1.0185            1.0169         6.17
IVF-OPQ-nl316-m128-np15 (query)                       20_322.03     2_714.36    23_036.38       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np17 (query)                       20_322.03     3_009.52    23_331.54       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np25 (query)                       20_322.03     4_176.99    24_499.01       0.8417          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128 (self)                             20_322.03    15_460.03    35_782.06       0.8143          1.0092            1.0078         9.23
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
Exhaustive (query)                                       100.59     1_880.27     1_980.85       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.59     6_208.21     6_308.79       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_971.85     1_565.17    11_537.03       0.2295          1.2024            1.1985         3.76
Exhaustive-OPQ-m16 (self)                              9_971.85     8_400.50    18_372.36       0.1868          1.2974            1.2971         3.76
Exhaustive-OPQ-m32 (query)                            12_792.17     2_616.97    15_409.15       0.3123          1.1452            1.1413         4.53
Exhaustive-OPQ-m32 (self)                             12_792.17    11_242.63    24_034.81       0.2574          1.2173            1.2150         4.53
Exhaustive-OPQ-m64 (query)                            18_659.36     4_585.92    23_245.28       0.4084          1.0974            1.0946         6.05
Exhaustive-OPQ-m64 (self)                             18_659.36    18_371.59    37_030.95       0.3479          1.1498            1.1473         6.05
Exhaustive-OPQ-m128 (query)                           28_995.89     8_944.96    37_940.85       0.5283          1.0567            1.0548         9.11
Exhaustive-OPQ-m128 (self)                            28_995.89    33_127.60    62_123.49       0.4662          1.0907            1.0887         9.11
IVF-OPQ-nl158-m16-np7 (query)                         10_009.52     1_181.45    11_190.97       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16-np12 (query)                        10_009.52     1_361.09    11_370.61       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16-np17 (query)                        10_009.52     1_546.63    11_556.15       0.5293          1.0553            1.0540         4.98
IVF-OPQ-nl158-m16 (self)                              10_009.52     8_465.78    18_475.30       0.4254          1.1062            1.1043         4.98
IVF-OPQ-nl158-m32-np7 (query)                         12_783.94     1_377.58    14_161.52       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32-np12 (query)                        12_783.94     1_654.06    14_438.00       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32-np17 (query)                        12_783.94     1_973.74    14_757.68       0.6745          1.0243            1.0231         5.74
IVF-OPQ-nl158-m32 (self)                              12_783.94     9_805.08    22_589.02       0.6000          1.0460            1.0436         5.74
IVF-OPQ-nl158-m64-np7 (query)                         18_516.57     1_665.14    20_181.71       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64-np12 (query)                        18_516.57     2_145.05    20_661.62       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64-np17 (query)                        18_516.57     2_626.27    21_142.84       0.7713          1.0115            1.0104         7.27
IVF-OPQ-nl158-m64 (self)                              18_516.57    12_036.69    30_553.25       0.7240          1.0209            1.0186         7.27
IVF-OPQ-nl158-m128-np7 (query)                        28_486.51     2_481.19    30_967.70       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np12 (query)                       28_486.51     3_416.46    31_902.97       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128-np17 (query)                       28_486.51     4_397.72    32_884.24       0.8306          1.0063            1.0052        10.32
IVF-OPQ-nl158-m128 (self)                             28_486.51    17_955.35    46_441.86       0.8040          1.0106            1.0085        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_379.04     1_341.31    11_720.35       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_379.04     1_456.43    11_835.47       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_379.04     1_707.03    12_086.07       0.5395          1.0522            1.0510         5.17
IVF-OPQ-nl223-m16 (self)                              10_379.04     8_959.86    19_338.89       0.4406          1.0996            1.0977         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_249.30     1_601.96    14_851.26       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_249.30     1_777.93    15_027.23       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_249.30     2_204.56    15_453.86       0.6832          1.0229            1.0217         5.93
IVF-OPQ-nl223-m32 (self)                              13_249.30    10_656.42    23_905.72       0.6099          1.0434            1.0413         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_984.10     2_051.73    20_035.84       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_984.10     2_368.48    20_352.58       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_984.10     3_025.88    21_009.98       0.7782          1.0108            1.0097         7.46
IVF-OPQ-nl223-m64 (self)                              17_984.10    13_373.90    31_358.00       0.7314          1.0196            1.0177         7.46
IVF-OPQ-nl223-m128-np11 (query)                       29_003.39     3_227.82    32_231.20       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np14 (query)                       29_003.39     3_823.21    32_826.60       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np21 (query)                       29_003.39     5_202.24    34_205.62       0.8356          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128 (self)                             29_003.39    20_576.75    49_580.14       0.8077          1.0100            1.0082        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_502.74     1_480.27    11_983.02       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_502.74     1_558.04    12_060.79       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_502.74     1_845.77    12_348.51       0.5423          1.0516            1.0503         6.19
IVF-OPQ-nl316-m16 (self)                              10_502.74     9_428.26    19_931.00       0.4446          1.0980            1.0962         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_312.91     1_838.73    15_151.64       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_312.91     1_995.93    15_308.85       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_312.91     2_456.62    15_769.53       0.6876          1.0224            1.0212         6.96
IVF-OPQ-nl316-m32 (self)                              13_312.91    11_428.68    24_741.59       0.6139          1.0423            1.0405         6.96
IVF-OPQ-nl316-m64-np15 (query)                        19_519.36     2_421.50    21_940.85       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64-np17 (query)                        19_519.36     2_617.85    22_137.21       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64-np25 (query)                        19_519.36     3_399.47    22_918.83       0.7806          1.0106            1.0096         8.48
IVF-OPQ-nl316-m64 (self)                              19_519.36    14_753.06    34_272.42       0.7332          1.0192            1.0176         8.48
IVF-OPQ-nl316-m128-np15 (query)                       29_506.26     3_954.49    33_460.76       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np17 (query)                       29_506.26     4_332.35    33_838.61       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np25 (query)                       29_506.26     5_884.12    35_390.39       0.8369          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128 (self)                             29_506.26    22_949.50    52_455.76       0.8088          1.0097            1.0082        11.54
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
Exhaustive (query)                                        33.05       716.42       749.47       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.05     2_456.64     2_489.70       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_743.72       765.69     4_509.41       0.7911          1.0819            1.0684         1.26
Exhaustive-OPQ-m16 (self)                              3_743.72     2_749.57     6_493.29       0.7232          1.1502            1.1255         1.26
Exhaustive-OPQ-m32 (query)                             6_214.30     1_630.15     7_844.45       0.8303          1.0536            1.0424         2.03
Exhaustive-OPQ-m32 (self)                              6_214.30     5_608.72    11_823.02       0.7763          1.0975            1.0767         2.03
Exhaustive-OPQ-m64 (query)                             9_674.14     3_755.27    13_429.41       0.8562          1.0398            1.0292         3.55
Exhaustive-OPQ-m64 (self)                              9_674.14    12_759.68    22_433.83       0.8092          1.0723            1.0534         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_890.06       294.23     5_184.29       0.8907          1.0205            1.0162         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_890.06       437.57     5_327.63       0.8914          1.0201            1.0160         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_890.06       579.41     5_469.47       0.8914          1.0201            1.0161         1.67
IVF-OPQ-nl158-m16 (self)                               4_890.06     2_291.93     7_181.99       0.8482          1.0400            1.0322         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_362.59       480.21     6_842.81       0.9109          1.0134            1.0098         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_362.59       753.96     7_116.55       0.9118          1.0130            1.0097         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_362.59     1_037.75     7_400.34       0.9118          1.0129            1.0097         2.43
IVF-OPQ-nl158-m32 (self)                               6_362.59     3_803.02    10_165.61       0.8768          1.0257            1.0197         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_820.49       824.16    10_644.65       0.9243          1.0097            1.0066         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_820.49     1_364.27    11_184.75       0.9251          1.0093            1.0064         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_820.49     1_853.30    11_673.79       0.9251          1.0093            1.0064         3.96
IVF-OPQ-nl158-m64 (self)                               9_820.49     6_396.41    16_216.90       0.8964          1.0184            1.0132         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_278.32       391.54     4_669.86       0.8976          1.0178            1.0137         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_278.32       472.82     4_751.14       0.8978          1.0177            1.0137         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_278.32       677.49     4_955.81       0.8978          1.0177            1.0137         1.73
IVF-OPQ-nl223-m16 (self)                               4_278.32     2_583.22     6_861.54       0.8577          1.0353            1.0274         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_668.12       633.39     7_301.51       0.9156          1.0118            1.0087         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_668.12       788.20     7_456.32       0.9158          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_668.12     1_130.37     7_798.49       0.9158          1.0117            1.0086         2.50
IVF-OPQ-nl223-m32 (self)                               6_668.12     4_127.41    10_795.53       0.8834          1.0232            1.0174         2.50
IVF-OPQ-nl223-m64-np11 (query)                         9_853.13     1_086.21    10_939.33       0.9269          1.0091            1.0058         4.02
IVF-OPQ-nl223-m64-np14 (query)                         9_853.13     1_347.54    11_200.67       0.9270          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64-np21 (query)                         9_853.13     1_987.31    11_840.43       0.9271          1.0090            1.0058         4.02
IVF-OPQ-nl223-m64 (self)                               9_853.13     6_831.92    16_685.05       0.8991          1.0176            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_299.34       509.15     4_808.49       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_299.34       565.61     4_864.95       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_299.34       791.76     5_091.10       0.9016          1.0167            1.0125         2.07
IVF-OPQ-nl316-m16 (self)                               4_299.34     2_962.93     7_262.27       0.8627          1.0334            1.0249         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_687.33       820.48     7_507.81       0.9170          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_687.33       888.27     7_575.60       0.9170          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_687.33     1_263.57     7_950.90       0.9171          1.0113            1.0083         2.84
IVF-OPQ-nl316-m32 (self)                               6_687.33     4_569.27    11_256.61       0.8845          1.0229            1.0168         2.84
IVF-OPQ-nl316-m64-np15 (query)                        10_120.92     1_348.25    11_469.17       0.9287          1.0086            1.0056         4.36
IVF-OPQ-nl316-m64-np17 (query)                        10_120.92     1_523.16    11_644.08       0.9288          1.0085            1.0056         4.36
IVF-OPQ-nl316-m64-np25 (query)                        10_120.92     2_195.77    12_316.70       0.9288          1.0085            1.0056         4.36
IVF-OPQ-nl316-m64 (self)                              10_120.92     7_533.91    17_654.83       0.9011          1.0169            1.0116         4.36
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
Exhaustive (query)                                        68.18     1_327.35     1_395.54       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.18     4_350.29     4_418.47       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_123.62     1_040.87     7_164.49       0.7546          1.1136            1.0983         2.26
Exhaustive-OPQ-m16 (self)                              6_123.62     4_829.63    10_953.25       0.6788          1.2037            1.1739         2.26
Exhaustive-OPQ-m32 (query)                             8_328.64     1_916.42    10_245.05       0.8064          1.0692            1.0572         3.03
Exhaustive-OPQ-m32 (self)                              8_328.64     7_798.24    16_126.88       0.7455          1.1245            1.1019         3.03
Exhaustive-OPQ-m64 (query)                            13_081.60     4_059.63    17_141.24       0.8413          1.0455            1.0364         4.55
Exhaustive-OPQ-m64 (self)                             13_081.60    14_581.92    27_663.52       0.7916          1.0819            1.0654         4.55
Exhaustive-OPQ-m128 (query)                           19_872.93     8_338.53    28_211.46       0.9198          1.0107            1.0069         7.61
Exhaustive-OPQ-m128 (self)                            19_872.93    29_252.16    49_125.09       0.8933          1.0192            1.0139         7.61
IVF-OPQ-nl158-m16-np7 (query)                          6_596.19       634.12     7_230.30       0.8873          1.0229            1.0172         3.07
IVF-OPQ-nl158-m16-np12 (query)                         6_596.19       814.50     7_410.69       0.8876          1.0228            1.0171         3.07
IVF-OPQ-nl158-m16-np17 (query)                         6_596.19     1_003.25     7_599.44       0.8876          1.0227            1.0171         3.07
IVF-OPQ-nl158-m16 (self)                               6_596.19     4_795.05    11_391.23       0.8423          1.0463            1.0331         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_761.66       781.00     9_542.67       0.9007          1.0173            1.0129         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_761.66     1_065.96     9_827.62       0.9011          1.0171            1.0129         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_761.66     1_362.96    10_124.62       0.9011          1.0171            1.0129         3.84
IVF-OPQ-nl158-m32 (self)                               8_761.66     6_026.00    14_787.66       0.8619          1.0347            1.0247         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_516.82     1_164.56    14_681.38       0.9101          1.0145            1.0102         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_516.82     1_729.64    15_246.46       0.9106          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_516.82     2_290.50    15_807.32       0.9106          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64 (self)                              13_516.82     9_048.31    22_565.13       0.8751          1.0284            1.0196         5.36
IVF-OPQ-nl158-m128-np7 (query)                        19_731.44     1_801.30    21_532.74       0.9613          1.0029            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                       19_731.44     2_793.53    22_524.97       0.9620          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                       19_731.44     3_777.07    23_508.50       0.9621          1.0026            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                             19_731.44    14_033.69    33_765.13       0.9450          1.0056            1.0016         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_844.25       761.18     7_605.43       0.8976          1.0194            1.0144         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_844.25       830.92     7_675.17       0.8977          1.0194            1.0144         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_844.25     1_065.80     7_910.05       0.8977          1.0194            1.0143         3.20
IVF-OPQ-nl223-m16 (self)                               6_844.25     4_972.41    11_816.66       0.8550          1.0394            1.0271         3.20
IVF-OPQ-nl223-m32-np11 (query)                         9_070.10       959.34    10_029.44       0.9080          1.0153            1.0107         3.96
IVF-OPQ-nl223-m32-np14 (query)                         9_070.10     1_112.48    10_182.58       0.9081          1.0153            1.0107         3.96
IVF-OPQ-nl223-m32-np21 (query)                         9_070.10     1_479.74    10_549.84       0.9081          1.0152            1.0107         3.96
IVF-OPQ-nl223-m32 (self)                               9_070.10     6_341.05    15_411.16       0.8702          1.0311            1.0211         3.96
IVF-OPQ-nl223-m64-np11 (query)                        13_780.44     1_483.46    15_263.90       0.9168          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64-np14 (query)                        13_780.44     1_863.73    15_644.16       0.9169          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64-np21 (query)                        13_780.44     2_496.55    16_276.99       0.9169          1.0122            1.0083         5.49
IVF-OPQ-nl223-m64 (self)                              13_780.44     9_645.02    23_425.46       0.8830          1.0245            1.0169         5.49
IVF-OPQ-nl223-m128-np11 (query)                       20_461.46     2_362.57    22_824.03       0.9658          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                       20_461.46     2_855.90    23_317.36       0.9659          1.0022            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                       20_461.46     4_070.16    24_531.62       0.9660          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                             20_461.46    15_544.15    36_005.61       0.9485          1.0050            1.0009         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_161.78       859.37     8_021.16       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_161.78       918.14     8_079.92       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_161.78     1_179.75     8_341.53       0.9047          1.0163            1.0119         3.88
IVF-OPQ-nl316-m16 (self)                               7_161.78     5_367.38    12_529.16       0.8659          1.0333            1.0225         3.88
IVF-OPQ-nl316-m32-np15 (query)                        10_769.10     1_159.90    11_929.00       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32-np17 (query)                        10_769.10     1_235.76    12_004.86       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32-np25 (query)                        10_769.10     1_625.68    12_394.78       0.9146          1.0127            1.0090         4.65
IVF-OPQ-nl316-m32 (self)                              10_769.10     6_884.44    17_653.54       0.8789          1.0266            1.0180         4.65
IVF-OPQ-nl316-m64-np15 (query)                        15_912.09     1_784.11    17_696.19       0.9210          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64-np17 (query)                        15_912.09     1_983.99    17_896.08       0.9210          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64-np25 (query)                        15_912.09     2_748.00    18_660.09       0.9211          1.0108            1.0073         6.17
IVF-OPQ-nl316-m64 (self)                              15_912.09    10_554.62    26_466.71       0.8886          1.0222            1.0150         6.17
IVF-OPQ-nl316-m128-np15 (query)                       21_031.69     2_833.55    23_865.23       0.9689          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                       21_031.69     3_146.33    24_178.02       0.9690          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                       21_031.69     4_463.46    25_495.15       0.9690          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                             21_031.69    16_416.61    37_448.30       0.9519          1.0045            1.0005         9.23
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
Exhaustive (query)                                       100.51     1_917.55     2_018.06       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.51     6_292.58     6_393.09       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                            10_029.42     1_558.48    11_587.90       0.7383          1.1306            1.1121         3.76
Exhaustive-OPQ-m16 (self)                             10_029.42     8_406.86    18_436.28       0.6595          1.2295            1.1962         3.76
Exhaustive-OPQ-m32 (query)                            12_514.93     2_450.72    14_965.66       0.8493          1.0411            1.0321         4.53
Exhaustive-OPQ-m32 (self)                             12_514.93    11_695.20    24_210.13       0.8006          1.0714            1.0580         4.53
Exhaustive-OPQ-m64 (query)                            17_928.12     4_602.09    22_530.21       0.8796          1.0255            1.0186         6.05
Exhaustive-OPQ-m64 (self)                             17_928.12    18_207.21    36_135.34       0.8413          1.0441            1.0341         6.05
Exhaustive-OPQ-m128 (query)                           28_329.16     8_888.95    37_218.11       0.9051          1.0148            1.0104         9.11
Exhaustive-OPQ-m128 (self)                            28_329.16    32_979.81    61_308.97       0.8741          1.0264            1.0201         9.11
IVF-OPQ-nl158-m16-np7 (query)                         10_380.07     1_214.42    11_594.49       0.8919          1.0221            1.0161         4.98
IVF-OPQ-nl158-m16-np12 (query)                        10_380.07     1_411.27    11_791.34       0.8921          1.0220            1.0160         4.98
IVF-OPQ-nl158-m16-np17 (query)                        10_380.07     1_641.59    12_021.66       0.8921          1.0220            1.0160         4.98
IVF-OPQ-nl158-m16 (self)                              10_380.07     9_234.22    19_614.29       0.8469          1.0440            1.0305         4.98
IVF-OPQ-nl158-m32-np7 (query)                         12_830.23     1_400.61    14_230.84       0.9368          1.0085            1.0035         5.74
IVF-OPQ-nl158-m32-np12 (query)                        12_830.23     1_854.31    14_684.54       0.9370          1.0084            1.0035         5.74
IVF-OPQ-nl158-m32-np17 (query)                        12_830.23     2_097.09    14_927.32       0.9370          1.0084            1.0035         5.74
IVF-OPQ-nl158-m32 (self)                              12_830.23    10_369.98    23_200.21       0.9075          1.0183            1.0075         5.74
IVF-OPQ-nl158-m64-np7 (query)                         18_206.41     1_769.60    19_976.01       0.9510          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64-np12 (query)                        18_206.41     2_367.04    20_573.45       0.9513          1.0050            1.0012         7.27
IVF-OPQ-nl158-m64-np17 (query)                        18_206.41     2_973.48    21_179.89       0.9513          1.0050            1.0012         7.27
IVF-OPQ-nl158-m64 (self)                              18_206.41    13_107.91    31_314.32       0.9271          1.0115            1.0036         7.27
IVF-OPQ-nl158-m128-np7 (query)                        28_238.40     2_656.87    30_895.28       0.9609          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128-np12 (query)                       28_238.40     3_872.38    32_110.78       0.9611          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128-np17 (query)                       28_238.40     5_065.00    33_303.40       0.9611          1.0032            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                             28_238.40    20_183.77    48_422.17       0.9395          1.0077            1.0017        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_716.02     1_348.67    12_064.69       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_716.02     1_474.64    12_190.66       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_716.02     1_773.96    12_489.98       0.8992          1.0189            1.0137         5.17
IVF-OPQ-nl223-m16 (self)                              10_716.02     9_109.06    19_825.08       0.8590          1.0364            1.0253         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_303.25     1_623.09    14_926.34       0.9424          1.0072            1.0027         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_303.25     1_820.72    15_123.97       0.9425          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_303.25     2_312.59    15_615.83       0.9425          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32 (self)                              13_303.25    10_882.41    24_185.66       0.9162          1.0147            1.0058         5.93
IVF-OPQ-nl223-m64-np11 (query)                        18_969.08     2_147.00    21_116.08       0.9546          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                        18_969.08     2_453.28    21_422.36       0.9547          1.0045            1.0007         7.46
IVF-OPQ-nl223-m64-np21 (query)                        18_969.08     3_221.50    22_190.57       0.9547          1.0045            1.0007         7.46
IVF-OPQ-nl223-m64 (self)                              18_969.08    13_968.34    32_937.41       0.9327          1.0098            1.0027         7.46
IVF-OPQ-nl223-m128-np11 (query)                       29_465.94     3_391.66    32_857.60       0.9638          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                       29_465.94     4_032.61    33_498.55       0.9639          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                       29_465.94     5_555.24    35_021.18       0.9639          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                             29_465.94    21_822.85    51_288.79       0.9434          1.0068            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_593.37     1_505.35    13_098.72       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_593.37     1_577.86    13_171.23       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_593.37     1_897.58    13_490.95       0.9043          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16 (self)                              11_593.37     9_543.92    21_137.29       0.8645          1.0338            1.0231         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_920.90     1_860.23    15_781.13       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_920.90     1_982.55    15_903.44       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_920.90     2_491.40    16_412.29       0.9444          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32 (self)                              13_920.90    11_633.02    25_553.92       0.9193          1.0137            1.0052         6.96
IVF-OPQ-nl316-m64-np15 (query)                        19_106.36     2_473.86    21_580.22       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np17 (query)                        19_106.36     2_676.17    21_782.53       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64-np25 (query)                        19_106.36     3_522.67    22_629.02       0.9565          1.0040            1.0006         8.48
IVF-OPQ-nl316-m64 (self)                              19_106.36    15_135.16    34_241.52       0.9359          1.0090            1.0023         8.48
IVF-OPQ-nl316-m128-np15 (query)                       29_641.02     4_050.77    33_691.79       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np17 (query)                       29_641.02     4_484.20    34_125.22       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np25 (query)                       29_641.02     6_134.22    35_775.24       0.9656          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128 (self)                             29_641.02    23_792.67    53_433.69       0.9462          1.0061            1.0009        11.54
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
Exhaustive (query)                                        68.40     1_258.65     1_327.05       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_505.39       131.84     1_637.23       0.3468          1.0755            1.0751         2.24
IVFPQ-m64-nl111-np1                                    2_489.33       243.53     2_732.86       0.4500          1.0504            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_663.34       134.40     1_797.74       0.3219          1.2732            1.0774         4.72
IVFPQ-m32-nl111-np2                                    1_505.39       179.52     1_684.91       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np2                                    2_489.33       321.98     2_811.31       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_663.34       187.89     1_851.23       0.3477          1.0764            1.0749         4.72
IVFPQ-m32-nl111-np4                                    1_505.39       275.48     1_780.87       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np4                                    2_489.33       500.53     2_989.86       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_663.34       284.75     1_948.09       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np5                                    1_505.39       322.93     1_828.32       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np5                                    2_489.33       597.71     3_087.04       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_663.34       334.42     1_997.76       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np8                                    1_505.39       468.16     1_973.55       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np8                                    2_489.33       873.22     3_362.55       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_663.34       492.36     2_155.70       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl111-np10                                   1_505.39       560.55     2_065.94       0.3479          1.0748            1.0749         2.24
IVFPQ-m64-nl111-np10                                   2_489.33     1_040.40     3_529.72       0.4516          1.0495            1.0440         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_663.34       587.27     2_250.61       0.3479          1.0748            1.0749         4.72
IVFPQ-m32-nl158-np1                                    1_662.18       129.15     1_791.33       0.3490          1.0730            1.0736         2.34
IVFPQ-m64-nl158-np1                                    2_608.55       233.68     2_842.22       0.4593          1.0469            1.0428         3.86
SOARPQ-shift0.5-m32-nl158-np1                          1_869.58       138.16     2_007.74       0.3165          1.1950            1.0771         4.82
IVFPQ-m32-nl158-np2                                    1_662.18       182.74     1_844.92       0.3526          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np2                                    2_608.55       318.43     2_926.98       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np2                          1_869.58       184.24     2_053.82       0.3520          1.0726            1.0731         4.82
IVFPQ-m32-nl158-np4                                    1_662.18       272.80     1_934.98       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np4                                    2_608.55       490.37     3_098.91       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np4                          1_869.58       280.04     2_149.62       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np7                                    1_662.18       430.82     2_093.00       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np7                                    2_608.55       759.38     3_367.92       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np7                          1_869.58       417.65     2_287.23       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np8                                    1_662.18       464.11     2_126.28       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np8                                    2_608.55       836.98     3_445.53       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np8                          1_869.58       469.31     2_338.89       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl158-np12                                   1_662.18       649.46     2_311.63       0.3527          1.0715            1.0730         2.34
IVFPQ-m64-nl158-np12                                   2_608.55     1_196.96     3_805.51       0.4644          1.0449            1.0422         3.86
SOARPQ-shift0.5-m32-nl158-np12                         1_869.58       666.67     2_536.24       0.3527          1.0715            1.0730         4.82
IVFPQ-m32-nl223-np1                                    1_705.33       106.40     1_811.72       0.3524          1.0705            1.0702         2.46
IVFPQ-m64-nl223-np1                                    2_690.38       169.71     2_860.09       0.4383          1.0504            1.0448         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_932.49       116.71     2_049.20       0.3306          1.1705            1.0732         4.95
IVFPQ-m32-nl223-np2                                    1_705.33       157.26     1_862.59       0.3660          1.0664            1.0666         2.46
IVFPQ-m64-nl223-np2                                    2_690.38       265.62     2_956.00       0.4668          1.0444            1.0401         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_932.49       176.25     2_108.73       0.3633          1.0686            1.0686         4.95
IVFPQ-m32-nl223-np4                                    1_705.33       258.22     1_963.55       0.3690          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np4                                    2_690.38       468.07     3_158.46       0.4762          1.0430            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_932.49       281.11     2_213.60       0.3673          1.0667            1.0665         4.95
IVFPQ-m32-nl223-np8                                    1_705.33       459.39     2_164.71       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np8                                    2_690.38       820.63     3_511.01       0.4775          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_932.49       480.79     2_413.28       0.3692          1.0656            1.0657         4.95
IVFPQ-m32-nl223-np11                                   1_705.33       610.75     2_316.08       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np11                                   2_690.38     1_098.35     3_788.73       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_932.49       634.91     2_567.40       0.3693          1.0656            1.0657         4.95
IVFPQ-m32-nl223-np14                                   1_705.33       771.18     2_476.50       0.3693          1.0656            1.0657         2.46
IVFPQ-m64-nl223-np14                                   2_690.38     1_374.60     4_064.98       0.4776          1.0428            1.0385         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_932.49       788.20     2_720.68       0.3693          1.0656            1.0657         4.95
IVFPQ-m32-nl316-np1                                    1_757.28       105.45     1_862.73       0.3529          1.0694            1.0685         2.65
IVFPQ-m64-nl316-np1                                    2_774.93       152.51     2_927.44       0.4277          1.0517            1.0458         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_048.08       112.69     2_160.77       0.3254          1.1561            1.0727         5.13
IVFPQ-m32-nl316-np2                                    1_757.28       154.72     1_912.00       0.3727          1.0626            1.0636         2.65
IVFPQ-m64-nl316-np2                                    2_774.93       248.45     3_023.38       0.4696          1.0426            1.0391         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_048.08       170.10     2_218.18       0.3692          1.0659            1.0659         5.13
IVFPQ-m32-nl316-np4                                    1_757.28       258.18     2_015.46       0.3787          1.0612            1.0620         2.65
IVFPQ-m64-nl316-np4                                    2_774.93       448.47     3_223.40       0.4855          1.0401            1.0368         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_048.08       276.32     2_324.40       0.3751          1.0632            1.0637         5.13
IVFPQ-m32-nl316-np8                                    1_757.28       466.65     2_223.93       0.3797          1.0610            1.0619         2.65
IVFPQ-m64-nl316-np8                                    2_774.93       815.56     3_590.49       0.4890          1.0397            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_048.08       483.27     2_531.34       0.3789          1.0615            1.0622         5.13
IVFPQ-m32-nl316-np15                                   1_757.28       825.26     2_582.54       0.3797          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np15                                   2_774.93     1_457.99     4_232.92       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_048.08       841.93     2_890.01       0.3797          1.0610            1.0618         5.13
IVFPQ-m32-nl316-np17                                   1_757.28       966.88     2_724.16       0.3798          1.0610            1.0618         2.65
IVFPQ-m64-nl316-np17                                   2_774.93     1_651.80     4_426.73       0.4894          1.0396            1.0364         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_048.08       956.61     3_004.69       0.3798          1.0610            1.0618         5.13
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
Exhaustive (query)                                        68.40     1_258.65     1_327.05       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        1_897.10       136.98     2_034.08       0.3175          1.1920            1.0770         4.82
SOARPQ-near-np2                                        1_897.10       184.86     2_081.96       0.3523          1.0721            1.0731         4.82
SOARPQ-near-np4                                        1_897.10       282.86     2_179.95       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np7                                        1_897.10       437.51     2_334.61       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np8                                        1_897.10       486.96     2_384.06       0.3527          1.0715            1.0730         4.82
SOARPQ-near-np12                                       1_897.10       694.48     2_591.58       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np1                                    1_917.66       134.16     2_051.82       0.3167          1.1944            1.0771         4.82
SOARPQ-shift0.3-np2                                    1_917.66       184.57     2_102.22       0.3521          1.0725            1.0731         4.82
SOARPQ-shift0.3-np4                                    1_917.66       281.06     2_198.72       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np7                                    1_917.66       438.97     2_356.62       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np8                                    1_917.66       506.65     2_424.31       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.3-np12                                   1_917.66       693.97     2_611.63       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np1                                    1_885.74       135.49     2_021.23       0.3164          1.1957            1.0771         4.82
SOARPQ-shift0.7-np2                                    1_885.74       183.70     2_069.43       0.3519          1.0730            1.0731         4.82
SOARPQ-shift0.7-np4                                    1_885.74       281.77     2_167.50       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np7                                    1_885.74       436.87     2_322.60       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np8                                    1_885.74       507.62     2_393.36       0.3527          1.0715            1.0730         4.82
SOARPQ-shift0.7-np12                                   1_885.74       707.81     2_593.55       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np1                                       1_921.02       134.55     2_055.57       0.3179          1.1930            1.0769         4.82
SOARPQ-orth1-np2                                       1_921.02       184.17     2_105.19       0.3525          1.0718            1.0730         4.82
SOARPQ-orth1-np4                                       1_921.02       282.97     2_203.99       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np7                                       1_921.02       435.40     2_356.42       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np8                                       1_921.02       489.81     2_410.83       0.3527          1.0715            1.0730         4.82
SOARPQ-orth1-np12                                      1_921.02       707.10     2_628.12       0.3527          1.0715            1.0730         4.82
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
Exhaustive (query)                                        68.31     1_417.17     1_485.48       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_843.27       158.28     2_001.55       0.4790          1.0776            1.0739         2.24
IVFPQ-m64-nl111-np1                                    2_924.15       221.68     3_145.83       0.6165          1.0395            1.0353         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_755.87       132.14     1_888.01       0.4659          1.0980            1.0769         4.72
IVFPQ-m32-nl111-np2                                    1_843.27       172.70     2_015.97       0.4838          1.0757            1.0732         2.24
IVFPQ-m64-nl111-np2                                    2_924.15       310.78     3_234.94       0.6244          1.0367            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_755.87       182.09     1_937.96       0.4828          1.0776            1.0736         4.72
IVFPQ-m32-nl111-np4                                    1_843.27       272.95     2_116.22       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np4                                    2_924.15       496.08     3_420.23       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_755.87       282.13     2_038.00       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np5                                    1_843.27       325.05     2_168.32       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np5                                    2_924.15       622.80     3_546.95       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_755.87       337.55     2_093.43       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np8                                    1_843.27       481.14     2_324.41       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np8                                    2_924.15       891.02     3_815.17       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_755.87       514.51     2_270.38       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl111-np10                                   1_843.27       586.83     2_430.10       0.4840          1.0756            1.0732         2.24
IVFPQ-m64-nl111-np10                                   2_924.15     1_064.71     3_988.86       0.6247          1.0366            1.0346         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_755.87       635.95     2_391.82       0.4840          1.0756            1.0732         4.72
IVFPQ-m32-nl158-np1                                    1_655.74       129.76     1_785.50       0.4830          1.0756            1.0721         2.34
IVFPQ-m64-nl158-np1                                    2_644.60       226.28     2_870.88       0.6147          1.0403            1.0347         3.86
SOARPQ-shift0.5-m32-nl158-np1                          2_026.39       130.35     2_156.73       0.4858          1.0778            1.0727         4.82
IVFPQ-m32-nl158-np2                                    1_655.74       175.68     1_831.42       0.4914          1.0723            1.0710         2.34
IVFPQ-m64-nl158-np2                                    2_644.60       318.07     2_962.67       0.6281          1.0356            1.0337         3.86
SOARPQ-shift0.5-m32-nl158-np2                          2_026.39       183.91     2_210.30       0.4912          1.0727            1.0713         4.82
IVFPQ-m32-nl158-np4                                    1_655.74       274.65     1_930.39       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np4                                    2_644.60       494.73     3_139.33       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np4                          2_026.39       280.39     2_306.78       0.4919          1.0721            1.0709         4.82
IVFPQ-m32-nl158-np7                                    1_655.74       422.60     2_078.34       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np7                                    2_644.60       763.75     3_408.35       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np7                          2_026.39       433.36     2_459.75       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np8                                    1_655.74       475.81     2_131.55       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np8                                    2_644.60       852.76     3_497.36       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np8                          2_026.39       485.10     2_511.49       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl158-np12                                   1_655.74       684.40     2_340.14       0.4921          1.0720            1.0708         2.34
IVFPQ-m64-nl158-np12                                   2_644.60     1_241.46     3_886.07       0.6294          1.0352            1.0336         3.86
SOARPQ-shift0.5-m32-nl158-np12                         2_026.39       702.31     2_728.70       0.4921          1.0720            1.0708         4.82
IVFPQ-m32-nl223-np1                                    1_762.30       108.92     1_871.22       0.3968          1.1044            1.1004         2.46
IVFPQ-m64-nl223-np1                                    2_700.85       164.99     2_865.84       0.4720          1.0736            1.0668         3.99
SOARPQ-shift0.5-m32-nl223-np1                          1_904.06       117.79     2_021.85       0.4490          1.0881            1.0842         4.95
IVFPQ-m32-nl223-np2                                    1_762.30       159.26     1_921.56       0.4582          1.0822            1.0800         2.46
IVFPQ-m64-nl223-np2                                    2_700.85       275.37     2_976.22       0.5701          1.0475            1.0435         3.99
SOARPQ-shift0.5-m32-nl223-np2                          1_904.06       180.20     2_084.25       0.4798          1.0760            1.0744         4.95
IVFPQ-m32-nl223-np4                                    1_762.30       265.88     2_028.18       0.4852          1.0739            1.0723         2.46
IVFPQ-m64-nl223-np4                                    2_700.85       466.48     3_167.33       0.6182          1.0373            1.0354         3.99
SOARPQ-shift0.5-m32-nl223-np4                          1_904.06       285.58     2_189.63       0.4900          1.0725            1.0712         4.95
IVFPQ-m32-nl223-np8                                    1_762.30       475.60     2_237.90       0.4917          1.0720            1.0707         2.46
IVFPQ-m64-nl223-np8                                    2_700.85       836.58     3_537.42       0.6323          1.0346            1.0332         3.99
SOARPQ-shift0.5-m32-nl223-np8                          1_904.06       490.42     2_394.48       0.4919          1.0720            1.0707         4.95
IVFPQ-m32-nl223-np11                                   1_762.30       658.95     2_421.26       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np11                                   2_700.85     1_112.93     3_813.78       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np11                         1_904.06       649.43     2_553.49       0.4920          1.0719            1.0706         4.95
IVFPQ-m32-nl223-np14                                   1_762.30       787.92     2_550.22       0.4920          1.0719            1.0706         2.46
IVFPQ-m64-nl223-np14                                   2_700.85     1_436.09     4_136.94       0.6330          1.0344            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np14                         1_904.06       830.16     2_734.21       0.4920          1.0719            1.0706         4.95
IVFPQ-m32-nl316-np1                                    2_202.37       106.49     2_308.86       0.3503          1.1215            1.1185         2.65
IVFPQ-m64-nl316-np1                                    3_126.14       155.36     3_281.51       0.3989          1.0931            1.0885         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_308.55       117.36     2_425.90       0.4206          1.0956            1.0942         5.13
IVFPQ-m32-nl316-np2                                    2_202.37       154.66     2_357.03       0.4250          1.0922            1.0912         2.65
IVFPQ-m64-nl316-np2                                    3_126.14       251.96     3_378.11       0.5171          1.0589            1.0562         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_308.55       173.65     2_482.19       0.4612          1.0816            1.0804         5.13
IVFPQ-m32-nl316-np4                                    2_202.37       273.16     2_475.53       0.4685          1.0785            1.0771         2.65
IVFPQ-m64-nl316-np4                                    3_126.14       460.38     3_586.53       0.5954          1.0419            1.0397         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_308.55       285.82     2_594.36       0.4828          1.0746            1.0734         5.13
IVFPQ-m32-nl316-np8                                    2_202.37       473.79     2_676.17       0.4864          1.0733            1.0718         2.65
IVFPQ-m64-nl316-np8                                    3_126.14       827.58     3_953.73       0.6309          1.0349            1.0331         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_308.55       489.47     2_798.02       0.4879          1.0731            1.0717         5.13
IVFPQ-m32-nl316-np15                                   2_202.37       835.07     3_037.44       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np15                                   3_126.14     1_463.31     4_589.46       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_308.55       869.72     3_178.27       0.4881          1.0728            1.0715         5.13
IVFPQ-m32-nl316-np17                                   2_202.37       953.56     3_155.93       0.4881          1.0728            1.0715         2.65
IVFPQ-m64-nl316-np17                                   3_126.14     1_664.35     4_790.50       0.6352          1.0341            1.0323         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_308.55       956.83     3_265.37       0.4881          1.0728            1.0715         5.13
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
Exhaustive (query)                                        68.31     1_417.17     1_485.48       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_159.11       136.27     2_295.37       0.4863          1.0774            1.0725         4.82
SOARPQ-near-np2                                        2_159.11       182.61     2_341.72       0.4913          1.0726            1.0713         4.82
SOARPQ-near-np4                                        2_159.11       289.26     2_448.36       0.4920          1.0721            1.0709         4.82
SOARPQ-near-np7                                        2_159.11       437.20     2_596.30       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np8                                        2_159.11       483.73     2_642.84       0.4921          1.0720            1.0708         4.82
SOARPQ-near-np12                                       2_159.11       704.42     2_863.52       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np1                                    2_151.30       141.17     2_292.47       0.4861          1.0776            1.0726         4.82
SOARPQ-shift0.3-np2                                    2_151.30       183.45     2_334.75       0.4912          1.0726            1.0712         4.82
SOARPQ-shift0.3-np4                                    2_151.30       281.31     2_432.62       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.3-np7                                    2_151.30       436.35     2_587.65       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np8                                    2_151.30       483.77     2_635.08       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.3-np12                                   2_151.30       702.58     2_853.89       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np1                                    2_174.68       130.83     2_305.51       0.4855          1.0779            1.0727         4.82
SOARPQ-shift0.7-np2                                    2_174.68       181.69     2_356.37       0.4911          1.0727            1.0713         4.82
SOARPQ-shift0.7-np4                                    2_174.68       282.49     2_457.16       0.4919          1.0721            1.0709         4.82
SOARPQ-shift0.7-np7                                    2_174.68       431.11     2_605.78       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np8                                    2_174.68       491.42     2_666.10       0.4921          1.0720            1.0708         4.82
SOARPQ-shift0.7-np12                                   2_174.68       709.54     2_884.21       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np1                                       2_306.82       132.06     2_438.88       0.4855          1.0780            1.0728         4.82
SOARPQ-orth1-np2                                       2_306.82       182.23     2_489.05       0.4911          1.0728            1.0713         4.82
SOARPQ-orth1-np4                                       2_306.82       283.69     2_590.51       0.4919          1.0722            1.0709         4.82
SOARPQ-orth1-np7                                       2_306.82       439.19     2_746.01       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np8                                       2_306.82       491.65     2_798.47       0.4921          1.0720            1.0708         4.82
SOARPQ-orth1-np12                                      2_306.82       705.00     3_011.81       0.4921          1.0720            1.0708         4.82
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
Exhaustive (query)                                        72.92     1_467.21     1_540.12       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    2_001.75       105.16     2_106.91       0.7050          1.1847            1.0985         2.24
IVFPQ-m64-nl111-np1                                    2_941.14       169.18     3_110.32       0.7176          1.1741            1.0872         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_161.88       125.19     2_287.06       0.8135          1.0743            1.0456         4.72
IVFPQ-m32-nl111-np2                                    2_001.75       171.98     2_173.73       0.8205          1.0642            1.0404         2.24
IVFPQ-m64-nl111-np2                                    2_941.14       302.83     3_243.97       0.8435          1.0523            1.0274         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_161.88       213.70     2_375.58       0.8481          1.0449            1.0335         4.72
IVFPQ-m32-nl111-np4                                    2_001.75       298.86     2_300.60       0.8534          1.0387            1.0308         2.24
IVFPQ-m64-nl111-np4                                    2_941.14       564.84     3_505.98       0.8808          1.0260            1.0194         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_161.88       381.66     2_543.54       0.8549          1.0391            1.0309         4.72
IVFPQ-m32-nl111-np5                                    2_001.75       379.99     2_381.73       0.8551          1.0376            1.0303         2.24
IVFPQ-m64-nl111-np5                                    2_941.14       687.41     3_628.55       0.8826          1.0249            1.0190         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_161.88       469.04     2_630.92       0.8554          1.0385            1.0305         4.72
IVFPQ-m32-nl111-np8                                    2_001.75       570.30     2_572.05       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np8                                    2_941.14     1_103.84     4_044.99       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_161.88       696.16     2_858.04       0.8561          1.0374            1.0301         4.72
IVFPQ-m32-nl111-np10                                   2_001.75       721.09     2_722.83       0.8562          1.0370            1.0300         2.24
IVFPQ-m64-nl111-np10                                   2_941.14     1_363.44     4_304.58       0.8838          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_161.88       840.08     3_001.96       0.8561          1.0371            1.0300         4.72
IVFPQ-m32-nl158-np1                                    2_203.87       104.27     2_308.14       0.6971          1.1918            1.1067         2.34
IVFPQ-m64-nl158-np1                                    3_194.91       156.21     3_351.11       0.7064          1.1830            1.0984         3.86
SOARPQ-shift0.5-m32-nl158-np1                          2_343.81       115.83     2_459.64       0.8151          1.0761            1.0446         4.82
IVFPQ-m32-nl158-np2                                    2_203.87       157.33     2_361.20       0.8240          1.0636            1.0374         2.34
IVFPQ-m64-nl158-np2                                    3_194.91       273.78     3_468.68       0.8414          1.0541            1.0271         3.86
SOARPQ-shift0.5-m32-nl158-np2                          2_343.81       187.01     2_530.82       0.8573          1.0417            1.0292         4.82
IVFPQ-m32-nl158-np4                                    2_203.87       289.45     2_493.32       0.8638          1.0338            1.0259         2.34
IVFPQ-m64-nl158-np4                                    3_194.91       525.69     3_720.59       0.8850          1.0243            1.0175         3.86
SOARPQ-shift0.5-m32-nl158-np4                          2_343.81       338.81     2_682.62       0.8657          1.0347            1.0259         4.82
IVFPQ-m32-nl158-np7                                    2_203.87       469.06     2_672.93       0.8684          1.0311            1.0245         2.34
IVFPQ-m64-nl158-np7                                    3_194.91       862.37     4_057.27       0.8900          1.0215            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np7                          2_343.81       548.82     2_892.63       0.8680          1.0321            1.0248         4.82
IVFPQ-m32-nl158-np8                                    2_203.87       529.52     2_733.39       0.8685          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np8                                    3_194.91       977.80     4_172.70       0.8902          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np8                          2_343.81       611.32     2_955.13       0.8682          1.0317            1.0247         4.82
IVFPQ-m32-nl158-np12                                   2_203.87       780.45     2_984.32       0.8687          1.0310            1.0244         2.34
IVFPQ-m64-nl158-np12                                   3_194.91     1_453.11     4_648.01       0.8903          1.0214            1.0164         3.86
SOARPQ-shift0.5-m32-nl158-np12                         2_343.81       900.54     3_244.35       0.8686          1.0311            1.0245         4.82
IVFPQ-m32-nl223-np1                                    2_193.33       109.51     2_302.84       0.6865          1.1976            1.1231         2.46
IVFPQ-m64-nl223-np1                                    3_094.94       147.82     3_242.76       0.6934          1.1901            1.1152         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_447.16       109.35     2_556.51       0.8126          1.0799            1.0465         4.95
IVFPQ-m32-nl223-np2                                    2_193.33       150.04     2_343.37       0.8242          1.0647            1.0369         2.46
IVFPQ-m64-nl223-np2                                    3_094.94       258.92     3_353.86       0.8391          1.0564            1.0285         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_447.16       174.12     2_621.28       0.8641          1.0402            1.0265         4.95
IVFPQ-m32-nl223-np4                                    2_193.33       268.90     2_462.23       0.8725          1.0307            1.0222         2.46
IVFPQ-m64-nl223-np4                                    3_094.94       483.62     3_578.55       0.8923          1.0219            1.0149         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_447.16       306.02     2_753.18       0.8759          1.0314            1.0220         4.95
IVFPQ-m32-nl223-np8                                    2_193.33       506.73     2_700.06       0.8793          1.0270            1.0204         2.46
IVFPQ-m64-nl223-np8                                    3_094.94       904.52     3_999.46       0.8997          1.0180            1.0133         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_447.16       550.29     2_997.45       0.8788          1.0281            1.0206         4.95
IVFPQ-m32-nl223-np11                                   2_193.33       660.85     2_854.18       0.8796          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np11                                   3_094.94     1_194.17     4_289.10       0.9001          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_447.16       750.00     3_197.16       0.8794          1.0273            1.0203         4.95
IVFPQ-m32-nl223-np14                                   2_193.33       832.79     3_026.12       0.8797          1.0268            1.0202         2.46
IVFPQ-m64-nl223-np14                                   3_094.94     1_523.30     4_618.23       0.9002          1.0179            1.0132         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_447.16       917.82     3_364.98       0.8796          1.0270            1.0203         4.95
IVFPQ-m32-nl316-np1                                    2_324.69       108.49     2_433.18       0.6730          1.2103            1.1374         2.65
IVFPQ-m64-nl316-np1                                    3_511.88       147.04     3_658.93       0.6777          1.2048            1.1331         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_590.13       107.42     2_697.55       0.8095          1.0841            1.0485         5.13
IVFPQ-m32-nl316-np2                                    2_324.69       149.56     2_474.25       0.8234          1.0660            1.0367         2.65
IVFPQ-m64-nl316-np2                                    3_511.88       242.41     3_754.30       0.8337          1.0605            1.0304         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_590.13       166.41     2_756.54       0.8708          1.0386            1.0233         5.13
IVFPQ-m32-nl316-np4                                    2_324.69       253.79     2_578.48       0.8823          1.0266            1.0185         2.65
IVFPQ-m64-nl316-np4                                    3_511.88       432.93     3_944.81       0.8968          1.0204            1.0133         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_590.13       278.69     2_868.82       0.8865          1.0283            1.0182         5.13
IVFPQ-m32-nl316-np8                                    2_324.69       478.31     2_803.00       0.8914          1.0217            1.0162         2.65
IVFPQ-m64-nl316-np8                                    3_511.88       839.17     4_351.05       0.9066          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_590.13       516.27     3_106.40       0.8906          1.0239            1.0167         5.13
IVFPQ-m32-nl316-np15                                   2_324.69       864.94     3_189.63       0.8921          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np15                                   3_511.88     1_514.41     5_026.30       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_590.13       934.27     3_524.40       0.8919          1.0218            1.0162         5.13
IVFPQ-m32-nl316-np17                                   2_324.69       975.87     3_300.56       0.8922          1.0214            1.0161         2.65
IVFPQ-m64-nl316-np17                                   3_511.88     1_721.39     5_233.27       0.9073          1.0152            1.0111         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_590.13     1_039.19     3_629.32       0.8920          1.0216            1.0161         5.13
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
Exhaustive (query)                                        72.92     1_467.21     1_540.12       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_413.35       120.90     2_534.25       0.8150          1.0767            1.0413         4.82
SOARPQ-near-np2                                        2_413.35       196.71     2_610.07       0.8584          1.0400            1.0282         4.82
SOARPQ-near-np4                                        2_413.35       342.33     2_755.69       0.8665          1.0333            1.0255         4.82
SOARPQ-near-np7                                        2_413.35       541.36     2_954.71       0.8682          1.0316            1.0247         4.82
SOARPQ-near-np8                                        2_413.35       611.05     3_024.41       0.8684          1.0314            1.0246         4.82
SOARPQ-near-np12                                       2_413.35       887.43     3_300.78       0.8687          1.0311            1.0244         4.82
SOARPQ-shift0.3-np1                                    2_383.54       115.61     2_499.15       0.8175          1.0739            1.0429         4.82
SOARPQ-shift0.3-np2                                    2_383.54       187.32     2_570.86       0.8583          1.0407            1.0288         4.82
SOARPQ-shift0.3-np4                                    2_383.54       328.87     2_712.41       0.8660          1.0342            1.0258         4.82
SOARPQ-shift0.3-np7                                    2_383.54       551.49     2_935.03       0.8681          1.0318            1.0248         4.82
SOARPQ-shift0.3-np8                                    2_383.54       614.04     2_997.58       0.8683          1.0315            1.0246         4.82
SOARPQ-shift0.3-np12                                   2_383.54       869.27     3_252.82       0.8686          1.0311            1.0245         4.82
SOARPQ-shift0.7-np1                                    2_378.19       116.27     2_494.46       0.8118          1.0794            1.0463         4.82
SOARPQ-shift0.7-np2                                    2_378.19       186.56     2_564.75       0.8562          1.0430            1.0297         4.82
SOARPQ-shift0.7-np4                                    2_378.19       333.37     2_711.56       0.8653          1.0353            1.0261         4.82
SOARPQ-shift0.7-np7                                    2_378.19       570.81     2_949.00       0.8679          1.0324            1.0249         4.82
SOARPQ-shift0.7-np8                                    2_378.19       611.81     2_990.00       0.8682          1.0319            1.0247         4.82
SOARPQ-shift0.7-np12                                   2_378.19       869.03     3_247.22       0.8686          1.0312            1.0245         4.82
SOARPQ-orth1-np1                                       2_422.12       117.57     2_539.70       0.8149          1.0770            1.0431         4.82
SOARPQ-orth1-np2                                       2_422.12       190.88     2_613.00       0.8579          1.0413            1.0287         4.82
SOARPQ-orth1-np4                                       2_422.12       330.46     2_752.58       0.8661          1.0343            1.0258         4.82
SOARPQ-orth1-np7                                       2_422.12       543.08     2_965.20       0.8681          1.0319            1.0248         4.82
SOARPQ-orth1-np8                                       2_422.12       604.40     3_026.52       0.8683          1.0316            1.0246         4.82
SOARPQ-orth1-np12                                      2_422.12       871.53     3_293.65       0.8686          1.0311            1.0244         4.82
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
Exhaustive (query)                                        75.67     1_497.20     1_572.86       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                    2_016.69        96.28     2_112.96       0.7674          1.1423            1.0643         2.24
IVFPQ-m64-nl111-np1                                    2_899.87       154.89     3_054.76       0.7738          1.1368            1.0562         3.77
SOARPQ-orth1-m32-nl111-np1                             2_037.74       116.10     2_153.84       0.8492          1.0681            1.0350         4.72
IVFPQ-m32-nl111-np2                                    2_016.69       173.43     2_190.12       0.8656          1.0441            1.0265         2.24
IVFPQ-m64-nl111-np2                                    2_899.87       269.71     3_169.57       0.8760          1.0383            1.0207         3.77
SOARPQ-orth1-m32-nl111-np2                             2_037.74       195.47     2_233.21       0.8782          1.0396            1.0248         4.72
IVFPQ-m32-nl111-np4                                    2_016.69       273.04     2_289.73       0.8857          1.0293            1.0218         2.24
IVFPQ-m64-nl111-np4                                    2_899.87       501.52     3_401.39       0.8968          1.0237            1.0165         3.77
SOARPQ-orth1-m32-nl111-np4                             2_037.74       342.81     2_380.55       0.8854          1.0317            1.0223         4.72
IVFPQ-m32-nl111-np5                                    2_016.69       341.87     2_358.56       0.8866          1.0289            1.0215         2.24
IVFPQ-m64-nl111-np5                                    2_899.87       630.87     3_530.73       0.8978          1.0232            1.0163         3.77
SOARPQ-orth1-m32-nl111-np5                             2_037.74       404.16     2_441.90       0.8862          1.0305            1.0219         4.72
IVFPQ-m32-nl111-np8                                    2_016.69       521.97     2_538.66       0.8872          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np8                                    2_899.87     1_009.67     3_909.54       0.8984          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np8                             2_037.74       622.31     2_660.06       0.8871          1.0290            1.0215         4.72
IVFPQ-m32-nl111-np10                                   2_016.69       638.10     2_654.78       0.8873          1.0286            1.0213         2.24
IVFPQ-m64-nl111-np10                                   2_899.87     1_248.84     4_148.71       0.8985          1.0229            1.0161         3.77
SOARPQ-orth1-m32-nl111-np10                            2_037.74       758.38     2_796.12       0.8872          1.0288            1.0214         4.72
IVFPQ-m32-nl158-np1                                    2_108.88        97.56     2_206.45       0.7517          1.1575            1.0756         2.34
IVFPQ-m64-nl158-np1                                    3_194.14       150.19     3_344.33       0.7581          1.1518            1.0676         3.86
SOARPQ-orth1-m32-nl158-np1                             2_368.62       110.95     2_479.57       0.8454          1.0732            1.0355         4.82
IVFPQ-m32-nl158-np2                                    2_108.88       155.66     2_264.54       0.8638          1.0466            1.0258         2.34
IVFPQ-m64-nl158-np2                                    3_194.14       261.40     3_455.53       0.8739          1.0408            1.0199         3.86
SOARPQ-orth1-m32-nl158-np2                             2_368.62       177.93     2_546.56       0.8822          1.0394            1.0227         4.82
IVFPQ-m32-nl158-np4                                    2_108.88       266.67     2_375.56       0.8918          1.0259            1.0187         2.34
IVFPQ-m64-nl158-np4                                    3_194.14       480.35     3_674.49       0.9036          1.0200            1.0138         3.86
SOARPQ-orth1-m32-nl158-np4                             2_368.62       311.00     2_679.63       0.8918          1.0289            1.0191         4.82
IVFPQ-m32-nl158-np7                                    2_108.88       458.68     2_567.56       0.8942          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np7                                    3_194.14       809.85     4_003.99       0.9060          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np7                             2_368.62       511.33     2_879.95       0.8940          1.0254            1.0182         4.82
IVFPQ-m32-nl158-np8                                    2_108.88       499.14     2_608.03       0.8943          1.0245            1.0180         2.34
IVFPQ-m64-nl158-np8                                    3_194.14       943.40     4_137.54       0.9061          1.0187            1.0132         3.86
SOARPQ-orth1-m32-nl158-np8                             2_368.62       581.47     2_950.09       0.8941          1.0250            1.0182         4.82
IVFPQ-m32-nl158-np12                                   2_108.88       736.85     2_845.73       0.8944          1.0244            1.0180         2.34
IVFPQ-m64-nl158-np12                                   3_194.14     1_378.04     4_572.18       0.9063          1.0186            1.0132         3.86
SOARPQ-orth1-m32-nl158-np12                            2_368.62       836.19     3_204.81       0.8944          1.0245            1.0180         4.82
IVFPQ-m32-nl223-np1                                    2_343.53        96.09     2_439.62       0.7283          1.1815            1.1041         2.46
IVFPQ-m64-nl223-np1                                    3_395.89       155.16     3_551.05       0.7318          1.1772            1.1004         3.99
SOARPQ-orth1-m32-nl223-np1                             2_573.05       102.10     2_675.15       0.8393          1.0777            1.0371         4.95
IVFPQ-m32-nl223-np2                                    2_343.53       145.28     2_488.81       0.8613          1.0501            1.0263         2.46
IVFPQ-m64-nl223-np2                                    3_395.89       239.99     3_635.89       0.8679          1.0460            1.0217         3.99
SOARPQ-orth1-m32-nl223-np2                             2_573.05       163.10     2_736.16       0.8871          1.0370            1.0205         4.95
IVFPQ-m32-nl223-np4                                    2_343.53       259.90     2_603.42       0.8979          1.0231            1.0163         2.46
IVFPQ-m64-nl223-np4                                    3_395.89       430.35     3_826.24       0.9065          1.0191            1.0133         3.99
SOARPQ-orth1-m32-nl223-np4                             2_573.05       281.75     2_854.80       0.8976          1.0276            1.0170         4.95
IVFPQ-m32-nl223-np8                                    2_343.53       459.26     2_802.78       0.9010          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np8                                    3_395.89       843.28     4_239.18       0.9099          1.0173            1.0126         3.99
SOARPQ-orth1-m32-nl223-np8                             2_573.05       507.07     3_080.13       0.9006          1.0227            1.0160         4.95
IVFPQ-m32-nl223-np11                                   2_343.53       630.69     2_974.21       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np11                                   3_395.89     1_127.61     4_523.51       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np11                            2_573.05       686.02     3_259.08       0.9009          1.0218            1.0158         4.95
IVFPQ-m32-nl223-np14                                   2_343.53       784.92     3_128.45       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np14                                   3_395.89     1_445.86     4_841.75       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np14                            2_573.05       876.24     3_449.29       0.9011          1.0216            1.0157         4.95
IVFPQ-m32-nl316-np1                                    2_345.46       100.09     2_445.54       0.7037          1.2091            1.1328         2.65
IVFPQ-m64-nl316-np1                                    3_317.73       140.22     3_457.95       0.7071          1.2044            1.1272         4.17
SOARPQ-orth1-m32-nl316-np1                             2_633.96       102.76     2_736.72       0.8250          1.0882            1.0452         5.13
IVFPQ-m32-nl316-np2                                    2_345.46       144.50     2_489.96       0.8495          1.0587            1.0313         2.65
IVFPQ-m64-nl316-np2                                    3_317.73       232.13     3_549.87       0.8573          1.0541            1.0258         4.17
SOARPQ-orth1-m32-nl316-np2                             2_633.96       158.64     2_792.60       0.8855          1.0375            1.0212         5.13
IVFPQ-m32-nl316-np4                                    2_345.46       243.55     2_589.01       0.8982          1.0231            1.0160         2.65
IVFPQ-m64-nl316-np4                                    3_317.73       414.21     3_731.94       0.9091          1.0184            1.0119         4.17
SOARPQ-orth1-m32-nl316-np4                             2_633.96       264.71     2_898.67       0.8994          1.0269            1.0166         5.13
IVFPQ-m32-nl316-np8                                    2_345.46       446.63     2_792.09       0.9033          1.0203            1.0148         2.65
IVFPQ-m64-nl316-np8                                    3_317.73       782.97     4_100.70       0.9144          1.0156            1.0109         4.17
SOARPQ-orth1-m32-nl316-np8                             2_633.96       480.93     3_114.89       0.9028          1.0220            1.0152         5.13
IVFPQ-m32-nl316-np15                                   2_345.46       809.44     3_154.90       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np15                                   3_317.73     1_447.70     4_765.43       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np15                            2_633.96       879.97     3_513.93       0.9035          1.0204            1.0148         5.13
IVFPQ-m32-nl316-np17                                   2_345.46       917.13     3_262.59       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np17                                   3_317.73     1_655.64     4_973.37       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np17                            2_633.96       974.29     3_608.25       0.9035          1.0203            1.0148         5.13
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
Exhaustive (query)                                        75.67     1_497.20     1_572.86       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        2_080.65       113.23     2_193.88       0.8501          1.0646            1.0334         4.82
SOARPQ-near-np2                                        2_080.65       180.15     2_260.80       0.8844          1.0345            1.0218         4.82
SOARPQ-near-np4                                        2_080.65       307.03     2_387.68       0.8927          1.0266            1.0188         4.82
SOARPQ-near-np7                                        2_080.65       504.01     2_584.66       0.8942          1.0248            1.0181         4.82
SOARPQ-near-np8                                        2_080.65       580.06     2_660.71       0.8943          1.0246            1.0181         4.82
SOARPQ-near-np12                                       2_080.65       819.26     2_899.91       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.3-np1                                    2_156.60       108.91     2_265.51       0.8492          1.0675            1.0348         4.82
SOARPQ-shift0.3-np2                                    2_156.60       176.32     2_332.91       0.8831          1.0374            1.0226         4.82
SOARPQ-shift0.3-np4                                    2_156.60       308.27     2_464.86       0.8918          1.0282            1.0191         4.82
SOARPQ-shift0.3-np7                                    2_156.60       504.64     2_661.23       0.8940          1.0252            1.0182         4.82
SOARPQ-shift0.3-np8                                    2_156.60       569.71     2_726.31       0.8941          1.0249            1.0182         4.82
SOARPQ-shift0.3-np12                                   2_156.60       819.43     2_976.03       0.8944          1.0245            1.0180         4.82
SOARPQ-shift0.7-np1                                    2_120.28       108.52     2_228.80       0.8440          1.0748            1.0368         4.82
SOARPQ-shift0.7-np2                                    2_120.28       176.01     2_296.29       0.8805          1.0417            1.0235         4.82
SOARPQ-shift0.7-np4                                    2_120.28       312.09     2_432.37       0.8906          1.0306            1.0196         4.82
SOARPQ-shift0.7-np7                                    2_120.28       506.86     2_627.13       0.8936          1.0261            1.0184         4.82
SOARPQ-shift0.7-np8                                    2_120.28       571.49     2_691.77       0.8939          1.0256            1.0183         4.82
SOARPQ-shift0.7-np12                                   2_120.28       827.57     2_947.85       0.8943          1.0247            1.0180         4.82
SOARPQ-orth1-np1                                       2_147.06       109.69     2_256.75       0.8454          1.0732            1.0355         4.82
SOARPQ-orth1-np2                                       2_147.06       175.73     2_322.79       0.8822          1.0394            1.0227         4.82
SOARPQ-orth1-np4                                       2_147.06       307.35     2_454.41       0.8918          1.0289            1.0191         4.82
SOARPQ-orth1-np7                                       2_147.06       503.43     2_650.49       0.8940          1.0254            1.0182         4.82
SOARPQ-orth1-np8                                       2_147.06       568.03     2_715.09       0.8941          1.0250            1.0182         4.82
SOARPQ-orth1-np12                                      2_147.06       811.72     2_958.78       0.8944          1.0245            1.0180         4.82
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
Exhaustive (query)                                        68.23     1_260.29     1_328.53       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   8_076.16       466.74     8_542.90       0.3585          1.0695            1.0694         3.49
IVFOPQ-m64-nl111-np1                                  12_809.50       581.15    13_390.66       0.4592          1.0462            1.0426         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_246.99       472.73     9_719.72       0.3380          1.2121            1.0714         5.98
IVFOPQ-m32-nl111-np2                                   8_076.16       525.82     8_601.98       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np2                                  12_809.50       686.82    13_496.33       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_246.99       526.21     9_773.20       0.3596          1.0702            1.0692         5.98
IVFOPQ-m32-nl111-np4                                   8_076.16       616.28     8_692.44       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np4                                  12_809.50       895.09    13_704.59       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_246.99       645.66     9_892.65       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np5                                   8_076.16       663.58     8_739.74       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np5                                  12_809.50     1_009.27    13_818.78       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_246.99       683.07     9_930.06       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np8                                   8_076.16       816.66     8_892.82       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np8                                  12_809.50     1_315.02    14_124.53       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_246.99       841.86    10_088.85       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl111-np10                                  8_076.16       924.48     9_000.64       0.3597          1.0689            1.0692         3.49
IVFOPQ-m64-nl111-np10                                 12_809.50     1_536.84    14_346.34       0.4613          1.0452            1.0424         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_246.99       948.27    10_195.26       0.3597          1.0689            1.0692         5.98
IVFOPQ-m32-nl158-np1                                   8_494.81       481.09     8_975.89       0.3639          1.0672            1.0680         3.84
IVFOPQ-m64-nl158-np1                                  13_246.18       577.50    13_823.68       0.4702          1.0432            1.0408         5.36
SOAROPQ-shift0.5-m32-nl158-np1                         9_763.22       472.88    10_236.10       0.3358          1.1637            1.0708         6.32
IVFOPQ-m32-nl158-np2                                   8_494.81       519.68     9_014.49       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np2                                  13_246.18       682.92    13_929.10       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np2                         9_763.22       527.33    10_290.55       0.3674          1.0667            1.0675         6.32
IVFOPQ-m32-nl158-np4                                   8_494.81       616.54     9_111.35       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np4                                  13_246.18       883.63    14_129.80       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np4                         9_763.22       630.47    10_393.69       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np7                                   8_494.81       753.42     9_248.23       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np7                                  13_246.18     1_193.54    14_439.72       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np7                         9_763.22       765.10    10_528.31       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np8                                   8_494.81       801.89     9_296.69       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np8                                  13_246.18     1_299.17    14_545.35       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np8                         9_763.22       816.13    10_579.34       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl158-np12                                  8_494.81       993.94     9_488.74       0.3680          1.0655            1.0674         3.84
IVFOPQ-m64-nl158-np12                                 13_246.18     1_715.29    14_961.47       0.4758          1.0410            1.0402         5.36
SOAROPQ-shift0.5-m32-nl158-np12                        9_763.22     1_009.36    10_772.58       0.3680          1.0655            1.0674         6.32
IVFOPQ-m32-nl223-np1                                   8_730.25       446.10     9_176.35       0.3606          1.0674            1.0674         3.96
IVFOPQ-m64-nl223-np1                                  13_321.65       516.37    13_838.02       0.4461          1.0466            1.0437         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_971.11       453.46    10_424.57       0.3478          1.1117            1.0696         6.45
IVFOPQ-m32-nl223-np2                                   8_730.25       498.98     9_229.23       0.3754          1.0628            1.0639         3.96
IVFOPQ-m64-nl223-np2                                  13_321.65       631.15    13_952.80       0.4761          1.0402            1.0390         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_971.11       516.45    10_487.56       0.3741          1.0642            1.0651         6.45
IVFOPQ-m32-nl223-np4                                   8_730.25       601.34     9_331.59       0.3790          1.0620            1.0631         3.96
IVFOPQ-m64-nl223-np4                                  13_321.65       903.56    14_225.21       0.4855          1.0389            1.0374         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_971.11       619.11    10_590.22       0.3778          1.0628            1.0635         6.45
IVFOPQ-m32-nl223-np8                                   8_730.25       797.84     9_528.09       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np8                                  13_321.65     1_272.67    14_594.32       0.4869          1.0387            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_971.11       815.81    10_786.91       0.3793          1.0619            1.0630         6.45
IVFOPQ-m32-nl223-np11                                  8_730.25       954.95     9_685.20       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np11                                 13_321.65     1_587.21    14_908.86       0.4869          1.0387            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_971.11       962.20    10_933.30       0.3794          1.0619            1.0629         6.45
IVFOPQ-m32-nl223-np14                                  8_730.25     1_084.28     9_814.53       0.3794          1.0619            1.0629         3.96
IVFOPQ-m64-nl223-np14                                 13_321.65     1_895.97    15_217.62       0.4869          1.0386            1.0372         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_971.11     1_106.65    11_077.76       0.3794          1.0619            1.0629         6.45
IVFOPQ-m32-nl316-np1                                   8_682.23       441.67     9_123.90       0.3591          1.0668            1.0666         4.65
IVFOPQ-m64-nl316-np1                                  13_662.00       506.83    14_168.83       0.4345          1.0485            1.0448         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_401.57       454.11    10_855.68       0.3407          1.1111            1.0694         7.13
IVFOPQ-m32-nl316-np2                                   8_682.23       498.34     9_180.56       0.3800          1.0598            1.0615         4.65
IVFOPQ-m64-nl316-np2                                  13_662.00       614.44    14_276.44       0.4771          1.0392            1.0381         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_401.57       507.99    10_909.56       0.3779          1.0616            1.0632         7.13
IVFOPQ-m32-nl316-np4                                   8_682.23       595.89     9_278.12       0.3868          1.0581            1.0600         4.65
IVFOPQ-m64-nl316-np4                                  13_662.00       848.47    14_510.47       0.4945          1.0366            1.0357         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_401.57       611.95    11_013.52       0.3836          1.0596            1.0613         7.13
IVFOPQ-m32-nl316-np8                                   8_682.23       794.29     9_476.51       0.3878          1.0579            1.0598         4.65
IVFOPQ-m64-nl316-np8                                  13_662.00     1_251.47    14_913.46       0.4982          1.0361            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_401.57       809.94    11_211.51       0.3870          1.0583            1.0600         7.13
IVFOPQ-m32-nl316-np15                                  8_682.23     1_149.52     9_831.74       0.3878          1.0579            1.0597         4.65
IVFOPQ-m64-nl316-np15                                 13_662.00     1_988.87    15_650.87       0.4984          1.0360            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_401.57     1_161.95    11_563.51       0.3878          1.0579            1.0597         7.13
IVFOPQ-m32-nl316-np17                                  8_682.23     1_251.16     9_933.38       0.3878          1.0579            1.0597         4.65
IVFOPQ-m64-nl316-np17                                 13_662.00     2_211.14    15_873.14       0.4984          1.0360            1.0352         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_401.57     1_267.75    11_669.32       0.3878          1.0579            1.0597         7.13
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
Exhaustive (query)                                        68.23     1_260.29     1_328.53       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_048.55       477.00    10_525.55       0.3367          1.1614            1.0707         6.32
SOAROPQ-near-np2                                      10_048.55       526.56    10_575.11       0.3676          1.0662            1.0675         6.32
SOAROPQ-near-np4                                      10_048.55       625.52    10_674.06       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np7                                      10_048.55       770.36    10_818.91       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np8                                      10_048.55       816.85    10_865.40       0.3680          1.0655            1.0674         6.32
SOAROPQ-near-np12                                     10_048.55     1_012.61    11_061.15       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np1                                  10_217.95       478.93    10_696.88       0.3360          1.1627            1.0708         6.32
SOAROPQ-shift0.3-np2                                  10_217.95       524.29    10_742.24       0.3674          1.0665            1.0675         6.32
SOAROPQ-shift0.3-np4                                  10_217.95       621.36    10_839.31       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np7                                  10_217.95       765.29    10_983.24       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np8                                  10_217.95       819.72    11_037.67       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.3-np12                                 10_217.95     1_016.48    11_234.43       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np1                                  10_162.36       476.83    10_639.19       0.3358          1.1639            1.0708         6.32
SOAROPQ-shift0.7-np2                                  10_162.36       531.97    10_694.33       0.3673          1.0671            1.0675         6.32
SOAROPQ-shift0.7-np4                                  10_162.36       622.03    10_784.39       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np7                                  10_162.36       765.06    10_927.42       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np8                                  10_162.36       822.59    10_984.95       0.3680          1.0655            1.0674         6.32
SOAROPQ-shift0.7-np12                                 10_162.36     1_013.59    11_175.95       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np1                                     10_149.60       497.11    10_646.71       0.3371          1.1616            1.0706         6.32
SOAROPQ-orth1-np2                                     10_149.60       524.27    10_673.87       0.3678          1.0658            1.0674         6.32
SOAROPQ-orth1-np4                                     10_149.60       623.19    10_772.79       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np7                                     10_149.60       769.98    10_919.58       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np8                                     10_149.60       818.47    10_968.07       0.3680          1.0655            1.0674         6.32
SOAROPQ-orth1-np12                                    10_149.60     1_025.35    11_174.95       0.3680          1.0655            1.0674         6.32
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
Exhaustive (query)                                        69.69     1_372.12     1_441.81       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   8_325.11       462.17     8_787.28       0.6648          1.0298            1.0250         3.49
IVFOPQ-m64-nl111-np1                                  13_258.51       571.61    13_830.11       0.7593          1.0165            1.0112         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_567.96       471.90    10_039.86       0.6656          1.0330            1.0252         5.98
IVFOPQ-m32-nl111-np2                                   8_325.11       513.31     8_838.42       0.6764          1.0260            1.0244         3.49
IVFOPQ-m64-nl111-np2                                  13_258.51       680.77    13_939.28       0.7734          1.0125            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_567.96       524.67    10_092.63       0.6764          1.0261            1.0245         5.98
IVFOPQ-m32-nl111-np4                                   8_325.11       609.55     8_934.66       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np4                                  13_258.51       885.77    14_144.27       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_567.96       619.05    10_187.01       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np5                                   8_325.11       673.93     8_999.03       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np5                                  13_258.51       991.61    14_250.11       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_567.96       672.49    10_240.45       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np8                                   8_325.11       802.70     9_127.81       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np8                                  13_258.51     1_305.14    14_563.64       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_567.96       848.01    10_415.97       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl111-np10                                  8_325.11       901.09     9_226.20       0.6770          1.0258            1.0244         3.49
IVFOPQ-m64-nl111-np10                                 13_258.51     1_519.66    14_778.17       0.7743          1.0123            1.0109         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_567.96       951.09    10_519.04       0.6770          1.0258            1.0244         5.98
IVFOPQ-m32-nl158-np1                                   8_831.17       471.69     9_302.86       0.6634          1.0312            1.0245         3.84
IVFOPQ-m64-nl158-np1                                  13_694.98       572.61    14_267.59       0.7543          1.0184            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_671.78       469.93    11_141.71       0.6770          1.0274            1.0242         6.32
IVFOPQ-m32-nl158-np2                                   8_831.17       515.20     9_346.37       0.6818          1.0253            1.0235         3.84
IVFOPQ-m64-nl158-np2                                  13_694.98       673.88    14_368.86       0.7762          1.0122            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_671.78       523.48    11_195.27       0.6834          1.0249            1.0235         6.32
IVFOPQ-m32-nl158-np4                                   8_831.17       618.52     9_449.68       0.6838          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np4                                  13_694.98       876.24    14_571.22       0.7788          1.0116            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_671.78       619.18    11_290.96       0.6839          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np7                                   8_831.17       751.68     9_582.85       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np7                                  13_694.98     1_191.02    14_886.00       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_671.78       755.64    11_427.43       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np8                                   8_831.17       797.04     9_628.20       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np8                                  13_694.98     1_283.30    14_978.28       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_671.78       808.85    11_480.63       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl158-np12                                  8_831.17       984.87     9_816.03       0.6840          1.0247            1.0234         3.84
IVFOPQ-m64-nl158-np12                                 13_694.98     1_683.93    15_378.91       0.7789          1.0115            1.0105         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_671.78     1_066.89    11_738.68       0.6840          1.0247            1.0234         6.32
IVFOPQ-m32-nl223-np1                                   8_998.85       446.43     9_445.28       0.4945          1.0652            1.0581         3.96
IVFOPQ-m64-nl223-np1                                  13_922.03       517.05    14_439.08       0.5346          1.0541            1.0467         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_411.34       457.94    10_869.28       0.5990          1.0416            1.0356         6.45
IVFOPQ-m32-nl223-np2                                   8_998.85       520.02     9_518.87       0.6125          1.0373            1.0324         3.96
IVFOPQ-m64-nl223-np2                                  13_922.03       639.86    14_561.89       0.6813          1.0254            1.0187         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_411.34       525.54    10_936.88       0.6610          1.0285            1.0263         6.45
IVFOPQ-m32-nl223-np4                                   8_998.85       608.29     9_607.14       0.6718          1.0266            1.0247         3.96
IVFOPQ-m64-nl223-np4                                  13_922.03       856.06    14_778.09       0.7590          1.0143            1.0119         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_411.34       626.54    11_037.88       0.6852          1.0243            1.0230         6.45
IVFOPQ-m32-nl223-np8                                   8_998.85       795.86     9_794.71       0.6888          1.0237            1.0225         3.96
IVFOPQ-m64-nl223-np8                                  13_922.03     1_266.79    15_188.82       0.7817          1.0112            1.0101         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_411.34       815.99    11_227.33       0.6898          1.0236            1.0224         6.45
IVFOPQ-m32-nl223-np11                                  8_998.85       940.69     9_939.54       0.6898          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np11                                 13_922.03     1_568.46    15_490.49       0.7833          1.0110            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_411.34       968.68    11_380.02       0.6898          1.0235            1.0223         6.45
IVFOPQ-m32-nl223-np14                                  8_998.85     1_090.85    10_089.70       0.6898          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np14                                 13_922.03     1_891.56    15_813.59       0.7833          1.0110            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_411.34     1_101.95    11_513.29       0.6898          1.0235            1.0223         6.45
IVFOPQ-m32-nl316-np1                                   9_135.86       447.28     9_583.15       0.4099          1.0855            1.0797         4.65
IVFOPQ-m64-nl316-np1                                  13_929.69       508.09    14_437.79       0.4293          1.0753            1.0691         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_564.24       452.59    11_016.83       0.5369          1.0531            1.0492         7.13
IVFOPQ-m32-nl316-np2                                   9_135.86       493.50     9_629.36       0.5493          1.0491            1.0454         4.65
IVFOPQ-m64-nl316-np2                                  13_929.69       616.51    14_546.20       0.5978          1.0379            1.0336         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_564.24       509.73    11_073.97       0.6301          1.0342            1.0315         7.13
IVFOPQ-m32-nl316-np4                                   9_135.86       600.00     9_735.86       0.6436          1.0315            1.0288         4.65
IVFOPQ-m64-nl316-np4                                  13_929.69       845.85    14_775.54       0.7203          1.0196            1.0159         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_564.24       615.15    11_179.39       0.6787          1.0256            1.0241         7.13
IVFOPQ-m32-nl316-np8                                   9_135.86       797.04     9_932.91       0.6876          1.0240            1.0227         4.65
IVFOPQ-m64-nl316-np8                                  13_929.69     1_244.38    15_174.07       0.7776          1.0118            1.0106         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_564.24       812.41    11_376.64       0.6923          1.0233            1.0221         7.13
IVFOPQ-m32-nl316-np15                                  9_135.86     1_159.38    10_295.24       0.6930          1.0231            1.0219         4.65
IVFOPQ-m64-nl316-np15                                 13_929.69     1_980.13    15_909.83       0.7851          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_564.24     1_135.66    11_699.90       0.6930          1.0231            1.0219         7.13
IVFOPQ-m32-nl316-np17                                  9_135.86     1_246.97    10_382.83       0.6930          1.0231            1.0219         4.65
IVFOPQ-m64-nl316-np17                                 13_929.69     2_167.04    16_096.74       0.7851          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_564.24     1_242.60    11_806.84       0.6930          1.0231            1.0219         7.13
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
Exhaustive (query)                                        69.69     1_372.12     1_441.81       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_002.39       466.86    10_469.25       0.6770          1.0274            1.0242         6.32
SOAROPQ-near-np2                                      10_002.39       518.81    10_521.19       0.6834          1.0250            1.0235         6.32
SOAROPQ-near-np4                                      10_002.39       614.25    10_616.64       0.6839          1.0247            1.0234         6.32
SOAROPQ-near-np7                                      10_002.39       755.95    10_758.34       0.6840          1.0247            1.0234         6.32
SOAROPQ-near-np8                                      10_002.39       807.99    10_810.37       0.6840          1.0247            1.0234         6.32
SOAROPQ-near-np12                                     10_002.39     1_008.79    11_011.18       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np1                                  10_127.68       489.53    10_617.22       0.6770          1.0274            1.0241         6.32
SOAROPQ-shift0.3-np2                                  10_127.68       530.35    10_658.03       0.6834          1.0250            1.0235         6.32
SOAROPQ-shift0.3-np4                                  10_127.68       614.99    10_742.67       0.6839          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np7                                  10_127.68       755.99    10_883.68       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np8                                  10_127.68       805.86    10_933.55       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.3-np12                                 10_127.68     1_004.01    11_131.69       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np1                                  10_175.23       477.26    10_652.49       0.6769          1.0275            1.0242         6.32
SOAROPQ-shift0.7-np2                                  10_175.23       518.42    10_693.65       0.6833          1.0250            1.0235         6.32
SOAROPQ-shift0.7-np4                                  10_175.23       613.90    10_789.13       0.6839          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np7                                  10_175.23       754.81    10_930.04       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np8                                  10_175.23       808.15    10_983.37       0.6840          1.0247            1.0234         6.32
SOAROPQ-shift0.7-np12                                 10_175.23     1_000.74    11_175.97       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np1                                     10_152.46       484.90    10_637.36       0.6766          1.0276            1.0242         6.32
SOAROPQ-orth1-np2                                     10_152.46       520.30    10_672.76       0.6833          1.0250            1.0235         6.32
SOAROPQ-orth1-np4                                     10_152.46       626.94    10_779.40       0.6838          1.0247            1.0234         6.32
SOAROPQ-orth1-np7                                     10_152.46       758.31    10_910.78       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np8                                     10_152.46       803.35    10_955.82       0.6840          1.0247            1.0234         6.32
SOAROPQ-orth1-np12                                    10_152.46     1_000.76    11_153.22       0.6840          1.0247            1.0234         6.32
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
Exhaustive (query)                                        71.18     1_321.17     1_392.36       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   8_364.37       443.98     8_808.35       0.7214          1.1714            1.0835         3.49
IVFOPQ-m64-nl111-np1                                  13_436.54       517.32    13_953.85       0.7259          1.1682            1.0806         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_467.14       466.99     9_934.13       0.8433          1.0560            1.0277         5.98
IVFOPQ-m32-nl111-np2                                   8_364.37       504.51     8_868.88       0.8509          1.0487            1.0238         3.49
IVFOPQ-m64-nl111-np2                                  13_436.54       658.83    14_095.37       0.8604          1.0452            1.0197         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_467.14       551.08    10_018.22       0.8849          1.0261            1.0174         5.98
IVFOPQ-m32-nl111-np4                                   8_364.37       636.04     9_000.41       0.8896          1.0220            1.0160         3.49
IVFOPQ-m64-nl111-np4                                  13_436.54       942.94    14_379.48       0.9012          1.0184            1.0124         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_467.14       732.95    10_200.09       0.8919          1.0213            1.0156         5.98
IVFOPQ-m32-nl111-np5                                   8_364.37       698.93     9_063.30       0.8916          1.0209            1.0156         3.49
IVFOPQ-m64-nl111-np5                                  13_436.54     1_098.02    14_534.55       0.9033          1.0173            1.0120         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_467.14       795.94    10_263.08       0.8923          1.0209            1.0155         5.98
IVFOPQ-m32-nl111-np8                                   8_364.37       891.87     9_256.23       0.8929          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np8                                  13_436.54     1_524.92    14_961.46       0.9047          1.0166            1.0117         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_467.14     1_020.36    10_487.50       0.8928          1.0205            1.0154         5.98
IVFOPQ-m32-nl111-np10                                  8_364.37     1_039.34     9_403.71       0.8929          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np10                                 13_436.54     1_807.36    15_243.90       0.9048          1.0166            1.0117         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_467.14     1_151.54    10_618.68       0.8929          1.0203            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   8_945.62       438.29     9_383.91       0.7099          1.1800            1.0954         3.84
IVFOPQ-m64-nl158-np1                                  13_529.35       506.89    14_036.25       0.7128          1.1777            1.0937         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_142.13       456.83    10_598.97       0.8400          1.0596            1.0281         6.32
IVFOPQ-m32-nl158-np2                                   8_945.62       495.38     9_440.99       0.8497          1.0504            1.0232         3.84
IVFOPQ-m64-nl158-np2                                  13_529.35       635.64    14_165.00       0.8566          1.0478            1.0202         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_142.13       531.29    10_673.42       0.8901          1.0245            1.0155         6.32
IVFOPQ-m32-nl158-np4                                   8_945.62       617.12     9_562.74       0.8953          1.0201            1.0139         3.84
IVFOPQ-m64-nl158-np4                                  13_529.35       890.31    14_419.66       0.9046          1.0174            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_142.13       672.33    10_814.47       0.8992          1.0187            1.0132         6.32
IVFOPQ-m32-nl158-np7                                   8_945.62       800.44     9_746.06       0.9007          1.0173            1.0129         3.84
IVFOPQ-m64-nl158-np7                                  13_529.35     1_274.32    14_803.67       0.9101          1.0145            1.0102         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_142.13       869.96    11_012.09       0.9007          1.0175            1.0130         6.32
IVFOPQ-m32-nl158-np8                                   8_945.62       846.63     9_792.24       0.9009          1.0171            1.0129         3.84
IVFOPQ-m64-nl158-np8                                  13_529.35     1_410.95    14_940.30       0.9104          1.0144            1.0101         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_142.13       966.40    11_108.54       0.9008          1.0174            1.0129         6.32
IVFOPQ-m32-nl158-np12                                  8_945.62     1_079.93    10_025.55       0.9011          1.0171            1.0129         3.84
IVFOPQ-m64-nl158-np12                                 13_529.35     1_914.73    15_444.09       0.9106          1.0143            1.0100         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_142.13     1_195.31    11_337.45       0.9011          1.0171            1.0129         6.32
IVFOPQ-m32-nl223-np1                                   9_208.88       437.28     9_646.16       0.6953          1.1884            1.1141         3.96
IVFOPQ-m64-nl223-np1                                  13_871.42       501.58    14_373.00       0.6973          1.1861            1.1118         5.49
SOAROPQ-shift0.5-m32-nl223-np1                        10_206.87       445.13    10_652.00       0.8322          1.0659            1.0325         6.45
IVFOPQ-m32-nl223-np2                                   9_208.88       512.35     9_721.23       0.8446          1.0542            1.0256         3.96
IVFOPQ-m64-nl223-np2                                  13_871.42       608.39    14_479.81       0.8501          1.0515            1.0221         5.49
SOAROPQ-shift0.5-m32-nl223-np2                        10_206.87       505.54    10_712.41       0.8922          1.0252            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   9_208.88       593.26     9_802.14       0.8997          1.0193            1.0124         3.96
IVFOPQ-m64-nl223-np4                                  13_871.42       843.07    14_714.49       0.9079          1.0164            1.0098         5.49
SOAROPQ-shift0.5-m32-nl223-np4                        10_206.87       631.68    10_838.55       0.9052          1.0176            1.0115         6.45
IVFOPQ-m32-nl223-np8                                   9_208.88       810.97    10_019.86       0.9076          1.0154            1.0108         3.96
IVFOPQ-m64-nl223-np8                                  13_871.42     1_287.62    15_159.03       0.9164          1.0124            1.0084         5.49
SOAROPQ-shift0.5-m32-nl223-np8                        10_206.87       874.26    11_081.13       0.9075          1.0158            1.0109         6.45
IVFOPQ-m32-nl223-np11                                  9_208.88       972.07    10_180.95       0.9080          1.0153            1.0107         3.96
IVFOPQ-m64-nl223-np11                                 13_871.42     1_657.05    15_528.46       0.9168          1.0122            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np11                       10_206.87     1_043.42    11_250.29       0.9079          1.0155            1.0108         6.45
IVFOPQ-m32-nl223-np14                                  9_208.88     1_125.81    10_334.69       0.9081          1.0153            1.0107         3.96
IVFOPQ-m64-nl223-np14                                 13_871.42     1_998.52    15_869.94       0.9169          1.0122            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np14                       10_206.87     1_244.72    11_451.59       0.9080          1.0153            1.0108         6.45
IVFOPQ-m32-nl316-np1                                   9_097.72       440.87     9_538.60       0.6789          1.2032            1.1314         4.65
IVFOPQ-m64-nl316-np1                                  14_185.57       495.62    14_681.19       0.6805          1.2011            1.1296         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_403.70       444.43    10_848.13       0.8245          1.0715            1.0363         7.13
IVFOPQ-m32-nl316-np2                                   9_097.72       491.35     9_589.07       0.8379          1.0584            1.0277         4.65
IVFOPQ-m64-nl316-np2                                  14_185.57       605.78    14_791.35       0.8415          1.0568            1.0256         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_403.70       501.75    10_905.46       0.8936          1.0251            1.0141         7.13
IVFOPQ-m32-nl316-np4                                   9_097.72       589.98     9_687.70       0.9033          1.0182            1.0111         4.65
IVFOPQ-m64-nl316-np4                                  14_185.57       863.67    15_049.24       0.9089          1.0163            1.0093         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_403.70       615.67    11_019.37       0.9102          1.0161            1.0102         7.13
IVFOPQ-m32-nl316-np8                                   9_097.72       791.28     9_889.01       0.9138          1.0131            1.0092         4.65
IVFOPQ-m64-nl316-np8                                  14_185.57     1_248.81    15_434.38       0.9202          1.0111            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_403.70       835.80    11_239.51       0.9137          1.0137            1.0093         7.13
IVFOPQ-m32-nl316-np15                                  9_097.72     1_147.61    10_245.33       0.9146          1.0127            1.0090         4.65
IVFOPQ-m64-nl316-np15                                 14_185.57     2_029.96    16_215.53       0.9210          1.0108            1.0073         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_403.70     1_217.33    11_621.03       0.9145          1.0129            1.0091         7.13
IVFOPQ-m32-nl316-np17                                  9_097.72     1_259.09    10_356.81       0.9146          1.0127            1.0090         4.65
IVFOPQ-m64-nl316-np17                                 14_185.57     2_258.63    16_444.20       0.9210          1.0108            1.0073         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_403.70     1_334.47    11_738.18       0.9145          1.0128            1.0091         7.13
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
Exhaustive (query)                                        71.18     1_321.17     1_392.36       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                       9_996.73       457.93    10_454.66       0.8390          1.0615            1.0261         6.32
SOAROPQ-near-np2                                       9_996.73       527.92    10_524.65       0.8901          1.0242            1.0152         6.32
SOAROPQ-near-np4                                       9_996.73       669.30    10_666.04       0.8995          1.0183            1.0132         6.32
SOAROPQ-near-np7                                       9_996.73       871.73    10_868.46       0.9008          1.0174            1.0129         6.32
SOAROPQ-near-np8                                       9_996.73       936.33    10_933.07       0.9009          1.0173            1.0129         6.32
SOAROPQ-near-np12                                      9_996.73     1_192.22    11_188.95       0.9011          1.0171            1.0129         6.32
SOAROPQ-shift0.3-np1                                  10_180.73       457.15    10_637.88       0.8423          1.0578            1.0271         6.32
SOAROPQ-shift0.3-np2                                  10_180.73       526.47    10_707.20       0.8908          1.0239            1.0153         6.32
SOAROPQ-shift0.3-np4                                  10_180.73       681.13    10_861.86       0.8995          1.0184            1.0132         6.32
SOAROPQ-shift0.3-np7                                  10_180.73       871.74    11_052.47       0.9008          1.0174            1.0129         6.32
SOAROPQ-shift0.3-np8                                  10_180.73     1_004.94    11_185.67       0.9009          1.0173            1.0129         6.32
SOAROPQ-shift0.3-np12                                 10_180.73     1_191.40    11_372.13       0.9011          1.0171            1.0129         6.32
SOAROPQ-shift0.7-np1                                  10_041.36       461.32    10_502.68       0.8365          1.0625            1.0295         6.32
SOAROPQ-shift0.7-np2                                  10_041.36       526.65    10_568.01       0.8890          1.0254            1.0158         6.32
SOAROPQ-shift0.7-np4                                  10_041.36       668.39    10_709.75       0.8989          1.0190            1.0134         6.32
SOAROPQ-shift0.7-np7                                  10_041.36       869.15    10_910.51       0.9006          1.0176            1.0130         6.32
SOAROPQ-shift0.7-np8                                  10_041.36       933.77    10_975.13       0.9008          1.0175            1.0130         6.32
SOAROPQ-shift0.7-np12                                 10_041.36     1_198.55    11_239.91       0.9011          1.0171            1.0129         6.32
SOAROPQ-orth1-np1                                      9_881.29       461.74    10_343.03       0.8391          1.0608            1.0274         6.32
SOAROPQ-orth1-np2                                      9_881.29       528.48    10_409.78       0.8901          1.0245            1.0154         6.32
SOAROPQ-orth1-np4                                      9_881.29       666.23    10_547.53       0.8994          1.0186            1.0132         6.32
SOAROPQ-orth1-np7                                      9_881.29       874.31    10_755.61       0.9008          1.0175            1.0129         6.32
SOAROPQ-orth1-np8                                      9_881.29       934.93    10_816.23       0.9009          1.0173            1.0129         6.32
SOAROPQ-orth1-np12                                     9_881.29     1_190.53    11_071.82       0.9011          1.0171            1.0129         6.32
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
Exhaustive (query)                                        74.30     1_359.17     1_433.47       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   8_059.08       435.14     8_494.22       0.7751          1.1356            1.0552         3.49
IVFOPQ-m64-nl111-np1                                  12_781.23       505.88    13_287.10       0.7781          1.1337            1.0517         5.02
SOAROPQ-orth1-m32-nl111-np1                            9_401.47       454.73     9_856.20       0.8634          1.0554            1.0260         5.98
IVFOPQ-m32-nl111-np2                                   8_059.08       492.71     8_551.79       0.8795          1.0366            1.0195         3.49
IVFOPQ-m64-nl111-np2                                  12_781.23       650.55    13_431.78       0.8843          1.0344            1.0173         5.02
SOAROPQ-orth1-m32-nl111-np2                            9_401.47       563.17     9_964.64       0.8948          1.0283            1.0173         5.98
IVFOPQ-m32-nl111-np4                                   8_059.08       621.65     8_680.73       0.9011          1.0217            1.0152         3.49
IVFOPQ-m64-nl111-np4                                  12_781.23       910.25    13_691.47       0.9065          1.0196            1.0130         5.02
SOAROPQ-orth1-m32-nl111-np4                            9_401.47       681.26    10_082.73       0.9014          1.0226            1.0153         5.98
IVFOPQ-m32-nl111-np5                                   8_059.08       683.50     8_742.58       0.9022          1.0212            1.0150         3.49
IVFOPQ-m64-nl111-np5                                  12_781.23     1_036.32    13_817.55       0.9075          1.0191            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np5                            9_401.47       754.08    10_155.55       0.9021          1.0220            1.0151         5.98
IVFOPQ-m32-nl111-np8                                   8_059.08       868.89     8_927.97       0.9028          1.0209            1.0148         3.49
IVFOPQ-m64-nl111-np8                                  12_781.23     1_443.34    14_224.56       0.9081          1.0188            1.0126         5.02
SOAROPQ-orth1-m32-nl111-np8                            9_401.47       969.00    10_370.47       0.9027          1.0212            1.0149         5.98
IVFOPQ-m32-nl111-np10                                  8_059.08       985.97     9_045.05       0.9029          1.0209            1.0148         3.49
IVFOPQ-m64-nl111-np10                                 12_781.23     1_696.90    14_478.13       0.9082          1.0188            1.0126         5.02
SOAROPQ-orth1-m32-nl111-np10                           9_401.47     1_158.97    10_560.44       0.9029          1.0211            1.0148         5.98
IVFOPQ-m32-nl158-np1                                   8_634.52       435.17     9_069.69       0.7588          1.1517            1.0674         3.84
IVFOPQ-m64-nl158-np1                                  13_413.34       501.03    13_914.36       0.7618          1.1491            1.0634         5.36
SOAROPQ-orth1-m32-nl158-np1                           10_125.09       449.10    10_574.19       0.8580          1.0609            1.0275         6.32
IVFOPQ-m32-nl158-np2                                   8_634.52       493.07     9_127.59       0.8754          1.0406            1.0197         3.84
IVFOPQ-m64-nl158-np2                                  13_413.34       619.59    14_032.92       0.8808          1.0379            1.0169         5.36
SOAROPQ-orth1-m32-nl158-np2                           10_125.09       515.97    10_641.05       0.8969          1.0287            1.0165         6.32
IVFOPQ-m32-nl158-np4                                   8_634.52       601.99     9_236.51       0.9056          1.0194            1.0134         3.84
IVFOPQ-m64-nl158-np4                                  13_413.34       863.35    14_276.69       0.9119          1.0168            1.0110         5.36
SOAROPQ-orth1-m32-nl158-np4                           10_125.09       649.63    10_774.72       0.9062          1.0205            1.0135         6.32
IVFOPQ-m32-nl158-np7                                   8_634.52       793.04     9_427.55       0.9081          1.0180            1.0128         3.84
IVFOPQ-m64-nl158-np7                                  13_413.34     1_240.27    14_653.60       0.9146          1.0155            1.0105         5.36
SOAROPQ-orth1-m32-nl158-np7                           10_125.09       845.88    10_970.96       0.9079          1.0185            1.0129         6.32
IVFOPQ-m32-nl158-np8                                   8_634.52       830.24     9_464.76       0.9082          1.0179            1.0128         3.84
IVFOPQ-m64-nl158-np8                                  13_413.34     1_390.62    14_803.95       0.9147          1.0154            1.0104         5.36
SOAROPQ-orth1-m32-nl158-np8                           10_125.09       911.17    11_036.26       0.9080          1.0183            1.0128         6.32
IVFOPQ-m32-nl158-np12                                  8_634.52     1_060.10     9_694.62       0.9084          1.0179            1.0127         3.84
IVFOPQ-m64-nl158-np12                                 13_413.34     1_870.76    15_284.09       0.9148          1.0154            1.0104         5.36
SOAROPQ-orth1-m32-nl158-np12                          10_125.09     1_155.73    11_280.82       0.9083          1.0180            1.0127         6.32
IVFOPQ-m32-nl223-np1                                   8_965.63       436.35     9_401.98       0.7330          1.1767            1.0999         3.96
IVFOPQ-m64-nl223-np1                                  13_762.17       492.58    14_254.75       0.7347          1.1749            1.0978         5.49
SOAROPQ-orth1-m32-nl223-np1                           10_459.90       440.47    10_900.37       0.8494          1.0687            1.0294         6.45
IVFOPQ-m32-nl223-np2                                   8_965.63       482.77     9_448.39       0.8709          1.0451            1.0207         3.96
IVFOPQ-m64-nl223-np2                                  13_762.17       602.06    14_364.23       0.8741          1.0434            1.0185         5.49
SOAROPQ-orth1-m32-nl223-np2                           10_459.90       498.97    10_958.87       0.9000          1.0280            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   8_965.63       587.45     9_553.07       0.9099          1.0179            1.0120         3.96
IVFOPQ-m64-nl223-np4                                  13_762.17       830.27    14_592.44       0.9149          1.0162            1.0103         5.49
SOAROPQ-orth1-m32-nl223-np4                           10_459.90       612.91    11_072.81       0.9108          1.0198            1.0121         6.45
IVFOPQ-m32-nl223-np8                                   8_965.63       787.16     9_752.79       0.9134          1.0162            1.0113         3.96
IVFOPQ-m64-nl223-np8                                  13_762.17     1_275.30    15_037.47       0.9185          1.0144            1.0095         5.49
SOAROPQ-orth1-m32-nl223-np8                           10_459.90       842.69    11_302.59       0.9132          1.0168            1.0114         6.45
IVFOPQ-m32-nl223-np11                                  8_965.63       970.07     9_935.70       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np11                                 13_762.17     1_620.78    15_382.94       0.9187          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np11                          10_459.90     1_016.21    11_476.12       0.9135          1.0163            1.0113         6.45
IVFOPQ-m32-nl223-np14                                  8_965.63     1_089.33    10_054.96       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np14                                 13_762.17     1_950.12    15_712.28       0.9188          1.0143            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np14                          10_459.90     1_189.32    11_649.23       0.9135          1.0162            1.0112         6.45
IVFOPQ-m32-nl316-np1                                   9_407.98       438.32     9_846.31       0.7080          1.2039            1.1272         4.65
IVFOPQ-m64-nl316-np1                                  14_096.70       491.73    14_588.43       0.7097          1.2018            1.1246         6.17
SOAROPQ-orth1-m32-nl316-np1                           10_704.26       443.17    11_147.42       0.8347          1.0799            1.0371         7.13
IVFOPQ-m32-nl316-np2                                   9_407.98       483.57     9_891.55       0.8590          1.0536            1.0252         4.65
IVFOPQ-m64-nl316-np2                                  14_096.70       600.49    14_697.18       0.8632          1.0515            1.0223         6.17
SOAROPQ-orth1-m32-nl316-np2                           10_704.26       512.18    11_216.43       0.8984          1.0293            1.0154         7.13
IVFOPQ-m32-nl316-np4                                   9_407.98       620.27    10_028.25       0.9114          1.0177            1.0112         4.65
IVFOPQ-m64-nl316-np4                                  14_096.70       820.71    14_917.41       0.9173          1.0157            1.0093         6.17
SOAROPQ-orth1-m32-nl316-np4                           10_704.26       605.87    11_310.13       0.9135          1.0191            1.0112         7.13
IVFOPQ-m32-nl316-np8                                   9_407.98       796.39    10_204.37       0.9170          1.0148            1.0101         4.65
IVFOPQ-m64-nl316-np8                                  14_096.70     1_234.82    15_331.51       0.9234          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np8                           10_704.26       818.86    11_523.12       0.9167          1.0159            1.0103         7.13
IVFOPQ-m32-nl316-np15                                  9_407.98     1_117.32    10_525.31       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np15                                 14_096.70     1_990.62    16_087.31       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np15                          10_704.26     1_194.82    11_899.08       0.9173          1.0148            1.0100         7.13
IVFOPQ-m32-nl316-np17                                  9_407.98     1_225.69    10_633.67       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np17                                 14_096.70     2_244.92    16_341.62       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np17                          10_704.26     1_317.33    12_021.59       0.9174          1.0147            1.0100         7.13
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
Exhaustive (query)                                        74.30     1_359.17     1_433.47       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                      10_169.75       450.72    10_620.47       0.8624          1.0547            1.0259         6.32
SOAROPQ-near-np2                                      10_169.75       517.01    10_686.75       0.8988          1.0257            1.0158         6.32
SOAROPQ-near-np4                                      10_169.75       651.34    10_821.09       0.9068          1.0193            1.0133         6.32
SOAROPQ-near-np7                                      10_169.75       847.34    11_017.09       0.9081          1.0181            1.0128         6.32
SOAROPQ-near-np8                                      10_169.75       911.94    11_081.69       0.9082          1.0180            1.0128         6.32
SOAROPQ-near-np12                                     10_169.75     1_166.44    11_336.19       0.9084          1.0179            1.0127         6.32
SOAROPQ-shift0.3-np1                                  10_771.16       451.11    11_222.27       0.8619          1.0562            1.0271         6.32
SOAROPQ-shift0.3-np2                                  10_771.16       517.45    11_288.62       0.8976          1.0275            1.0163         6.32
SOAROPQ-shift0.3-np4                                  10_771.16       655.91    11_427.07       0.9060          1.0202            1.0135         6.32
SOAROPQ-shift0.3-np7                                  10_771.16       851.09    11_622.26       0.9078          1.0185            1.0129         6.32
SOAROPQ-shift0.3-np8                                  10_771.16       925.14    11_696.30       0.9080          1.0183            1.0129         6.32
SOAROPQ-shift0.3-np12                                 10_771.16     1_159.16    11_930.32       0.9083          1.0180            1.0127         6.32
SOAROPQ-shift0.7-np1                                  10_986.40       469.38    11_455.77       0.8570          1.0624            1.0289         6.32
SOAROPQ-shift0.7-np2                                  10_986.40       524.37    11_510.77       0.8952          1.0306            1.0170         6.32
SOAROPQ-shift0.7-np4                                  10_986.40       657.99    11_644.38       0.9050          1.0219            1.0139         6.32
SOAROPQ-shift0.7-np7                                  10_986.40       851.10    11_837.50       0.9074          1.0191            1.0131         6.32
SOAROPQ-shift0.7-np8                                  10_986.40       918.60    11_905.00       0.9078          1.0187            1.0130         6.32
SOAROPQ-shift0.7-np12                                 10_986.40     1_165.99    12_152.39       0.9083          1.0180            1.0128         6.32
SOAROPQ-orth1-np1                                     11_099.35       450.73    11_550.08       0.8580          1.0609            1.0275         6.32
SOAROPQ-orth1-np2                                     11_099.35       522.97    11_622.33       0.8969          1.0287            1.0165         6.32
SOAROPQ-orth1-np4                                     11_099.35       655.95    11_755.30       0.9062          1.0205            1.0135         6.32
SOAROPQ-orth1-np7                                     11_099.35       850.21    11_949.56       0.9079          1.0185            1.0129         6.32
SOAROPQ-orth1-np8                                     11_099.35       925.53    12_024.88       0.9080          1.0183            1.0128         6.32
SOAROPQ-orth1-np12                                    11_099.35     1_168.24    12_267.59       0.9083          1.0180            1.0127         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
