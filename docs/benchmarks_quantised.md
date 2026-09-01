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
Exhaustive (query)                                        11.27       624.07       635.35       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.27     6_458.61     6_469.89       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.82     1_267.73     1_282.55       0.9828          1.0001            1.0000         9.16
Exhaustive-BF16 (self)                                    14.82    13_435.79    13_450.60       0.9798          1.0001            1.0000         9.16
IVF-BF16-nl273-np13 (query)                              466.77       119.84       586.61       0.9806          1.0003            1.0000         9.19
IVF-BF16-nl273-np16 (query)                              466.77       143.05       609.82       0.9825          1.0001            1.0000         9.19
IVF-BF16-nl273-np23 (query)                              466.77       220.99       687.76       0.9828          1.0001            1.0000         9.19
IVF-BF16-nl273 (self)                                    466.77     1_606.58     2_073.35       0.9798          1.0001            1.0000         9.19
IVF-BF16-nl387-np19 (query)                              687.57       126.22       813.79       0.9820          1.0001            1.0000         9.21
IVF-BF16-nl387-np27 (query)                              687.57       169.70       857.28       0.9828          1.0001            1.0000         9.21
IVF-BF16-nl387 (self)                                    687.57     1_285.23     1_972.80       0.9798          1.0001            1.0000         9.21
IVF-BF16-nl547-np23 (query)                            1_170.18       118.99     1_289.18       0.9773          1.0005            1.0000         9.23
IVF-BF16-nl547-np27 (query)                            1_170.18       132.29     1_302.47       0.9816          1.0002            1.0000         9.23
IVF-BF16-nl547-np33 (query)                            1_170.18       154.46     1_324.64       0.9828          1.0001            1.0000         9.23
IVF-BF16-nl547 (self)                                  1_170.18     1_219.69     2_389.87       0.9798          1.0001            1.0000         9.23
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
Exhaustive (query)                                        11.60       732.25       743.84       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         11.60     7_130.69     7_142.28       1.0000          1.0000            1.0000        18.88
Exhaustive-BF16 (query)                                   15.46     1_261.83     1_277.29       0.8870          1.0071            1.0019         9.44
Exhaustive-BF16 (self)                                    15.46    12_820.31    12_835.77       0.8852          1.0073            1.0020         9.44
IVF-BF16-nl273-np13 (query)                              299.49        97.87       397.37       0.8860          1.0073            1.0020         9.48
IVF-BF16-nl273-np16 (query)                              299.49       113.21       412.70       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273-np23 (query)                              299.49       151.95       451.44       0.8870          1.0071            1.0019         9.48
IVF-BF16-nl273 (self)                                    299.49     1_554.24     1_853.73       0.8852          1.0073            1.0020         9.48
IVF-BF16-nl387-np19 (query)                              543.48       101.44       644.92       0.8867          1.0072            1.0019         9.49
IVF-BF16-nl387-np27 (query)                              543.48       130.62       674.10       0.8870          1.0071            1.0019         9.49
IVF-BF16-nl387 (self)                                    543.48     1_338.70     1_882.18       0.8852          1.0073            1.0020         9.49
IVF-BF16-nl547-np23 (query)                            1_062.74        95.33     1_158.07       0.8848          1.0075            1.0021         9.51
IVF-BF16-nl547-np27 (query)                            1_062.74       106.54     1_169.29       0.8866          1.0072            1.0020         9.51
IVF-BF16-nl547-np33 (query)                            1_062.74       125.11     1_187.85       0.8870          1.0071            1.0019         9.51
IVF-BF16-nl547 (self)                                  1_062.74     1_260.32     2_323.06       0.8852          1.0073            1.0020         9.51
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
Exhaustive (query)                                        11.13       662.78       673.91       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.13     6_329.96     6_341.09       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   12.08     1_210.24     1_222.31       0.9345          1.0018            1.0011         9.16
Exhaustive-BF16 (self)                                    12.08    12_006.12    12_018.20       0.9184          1.0030            1.0021         9.16
IVF-BF16-nl273-np13 (query)                              304.82       119.52       424.34       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273-np16 (query)                              304.82       127.56       432.38       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273-np23 (query)                              304.82       172.10       476.92       0.9345          1.0018            1.0011         9.19
IVF-BF16-nl273 (self)                                    304.82     1_230.94     1_535.76       0.9184          1.0030            1.0021         9.19
IVF-BF16-nl387-np19 (query)                              566.14       123.25       689.39       0.9345          1.0018            1.0011         9.21
IVF-BF16-nl387-np27 (query)                              566.14       149.73       715.87       0.9345          1.0018            1.0011         9.21
IVF-BF16-nl387 (self)                                    566.14     1_100.08     1_666.21       0.9184          1.0030            1.0021         9.21
IVF-BF16-nl547-np23 (query)                            1_057.01       114.16     1_171.17       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547-np27 (query)                            1_057.01       124.37     1_181.38       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547-np33 (query)                            1_057.01       141.24     1_198.26       0.9345          1.0018            1.0011         9.23
IVF-BF16-nl547 (self)                                  1_057.01     1_089.43     2_146.44       0.9184          1.0030            1.0021         9.23
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
Exhaustive (query)                                        11.22       662.22       673.44       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.22     6_311.83     6_323.05       1.0000          1.0000            1.0000        18.31
Exhaustive-BF16 (query)                                   14.88     1_198.78     1_213.67       0.9541          1.0010            1.0003         9.16
Exhaustive-BF16 (self)                                    14.88    11_981.37    11_996.25       0.9429          1.0017            1.0009         9.16
IVF-BF16-nl273-np13 (query)                              337.26       110.86       448.12       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np16 (query)                              337.26       123.20       460.46       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273-np23 (query)                              337.26       171.17       508.43       0.9541          1.0010            1.0003         9.19
IVF-BF16-nl273 (self)                                    337.26     1_240.14     1_577.40       0.9429          1.0017            1.0009         9.19
IVF-BF16-nl387-np19 (query)                              573.82       116.72       690.54       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387-np27 (query)                              573.82       151.28       725.10       0.9541          1.0010            1.0003         9.21
IVF-BF16-nl387 (self)                                    573.82     1_105.64     1_679.46       0.9429          1.0017            1.0009         9.21
IVF-BF16-nl547-np23 (query)                            1_058.24       111.46     1_169.71       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np27 (query)                            1_058.24       121.25     1_179.49       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547-np33 (query)                            1_058.24       140.63     1_198.87       0.9541          1.0010            1.0003         9.23
IVF-BF16-nl547 (self)                                  1_058.24     1_051.36     2_109.60       0.9429          1.0017            1.0009         9.23
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
Exhaustive (query)                                        49.47     1_213.60     1_263.06       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.47    11_928.55    11_978.02       1.0000          1.0000            1.0000        73.24
Exhaustive-BF16 (query)                                   62.02     5_241.92     5_303.93       0.9723          1.0002            1.0000        36.62
Exhaustive-BF16 (self)                                    62.02    53_180.54    53_242.56       0.9679          1.0005            1.0000        36.62
IVF-BF16-nl273-np13 (query)                              697.02       326.84     1_023.87       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np16 (query)                              697.02       352.61     1_049.63       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273-np23 (query)                              697.02       484.39     1_181.41       0.9723          1.0002            1.0000        36.76
IVF-BF16-nl273 (self)                                    697.02     4_349.43     5_046.45       0.9679          1.0005            1.0000        36.76
IVF-BF16-nl387-np19 (query)                            1_332.96       319.81     1_652.77       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387-np27 (query)                            1_332.96       416.85     1_749.81       0.9723          1.0002            1.0000        36.81
IVF-BF16-nl387 (self)                                  1_332.96     3_955.33     5_288.29       0.9679          1.0005            1.0000        36.81
IVF-BF16-nl547-np23 (query)                            2_316.75       303.52     2_620.28       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np27 (query)                            2_316.75       335.33     2_652.09       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547-np33 (query)                            2_316.75       390.44     2_707.19       0.9723          1.0002            1.0000        36.89
IVF-BF16-nl547 (self)                                  2_316.75     3_475.11     5_791.86       0.9679          1.0005            1.0000        36.89
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
Exhaustive (query)                                        11.49       646.28       657.77       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.49     6_186.54     6_198.03       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    19.66       993.82     1_013.48       0.9256          1.0018            1.0009         5.15
Exhaustive-SQ8 (self)                                     19.66     9_945.43     9_965.09       0.9251          1.0018            1.0009         5.15
IVF-SQ8-nl273-np13 (query)                               334.80        68.49       403.29       0.9244          1.0020            1.0009         6.33
IVF-SQ8-nl273-np16 (query)                               334.80        77.93       412.73       0.9258          1.0018            1.0009         6.33
IVF-SQ8-nl273-np23 (query)                               334.80        98.78       433.58       0.9260          1.0018            1.0009         6.33
IVF-SQ8-nl273 (self)                                     334.80       972.00     1_306.80       0.9253          1.0018            1.0009         6.33
IVF-SQ8-nl387-np19 (query)                               603.43        75.01       678.44       0.9243          1.0019            1.0009         6.35
IVF-SQ8-nl387-np27 (query)                               603.43        92.68       696.11       0.9248          1.0018            1.0009         6.35
IVF-SQ8-nl387 (self)                                     603.43       880.40     1_483.84       0.9252          1.0018            1.0009         6.35
IVF-SQ8-nl547-np23 (query)                             1_123.47        67.26     1_190.72       0.9215          1.0022            1.0010         6.37
IVF-SQ8-nl547-np27 (query)                             1_123.47        75.88     1_199.35       0.9244          1.0019            1.0009         6.37
IVF-SQ8-nl547-np33 (query)                             1_123.47        85.68     1_209.15       0.9251          1.0018            1.0009         6.37
IVF-SQ8-nl547 (self)                                   1_123.47       839.19     1_962.65       0.9252          1.0018            1.0009         6.37
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
Exhaustive (query)                                        12.38       706.28       718.66       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         12.38     7_013.21     7_025.58       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    21.15       923.67       944.82       0.7397          1.0354            1.0161         5.15
Exhaustive-SQ8 (self)                                     21.15    10_413.32    10_434.48       0.7390          1.0356            1.0159         5.15
IVF-SQ8-nl273-np13 (query)                               318.24        65.04       383.28       0.7391          1.0362            1.0153         6.33
IVF-SQ8-nl273-np16 (query)                               318.24        77.60       395.84       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273-np23 (query)                               318.24        96.58       414.82       0.7395          1.0361            1.0153         6.33
IVF-SQ8-nl273 (self)                                     318.24       999.30     1_317.53       0.7378          1.0362            1.0152         6.33
IVF-SQ8-nl387-np19 (query)                               583.40        67.93       651.33       0.7378          1.0360            1.0155         6.35
IVF-SQ8-nl387-np27 (query)                               583.40        84.28       667.68       0.7379          1.0360            1.0155         6.35
IVF-SQ8-nl387 (self)                                     583.40       880.94     1_464.35       0.7375          1.0362            1.0157         6.35
IVF-SQ8-nl547-np23 (query)                             1_101.88        68.66     1_170.53       0.7365          1.0365            1.0165         6.37
IVF-SQ8-nl547-np27 (query)                             1_101.88        72.04     1_173.91       0.7370          1.0364            1.0163         6.37
IVF-SQ8-nl547-np33 (query)                             1_101.88        82.57     1_184.44       0.7369          1.0364            1.0162         6.37
IVF-SQ8-nl547 (self)                                   1_101.88       844.83     1_946.70       0.7360          1.0365            1.0159         6.37
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
Exhaustive (query)                                        11.36       645.64       657.00       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.36     6_392.14     6_403.50       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    21.40       990.31     1_011.71       0.8146          1.0165            1.0148         5.15
Exhaustive-SQ8 (self)                                     21.40    10_045.79    10_067.19       0.8120          1.0175            1.0155         5.15
IVF-SQ8-nl273-np13 (query)                               351.00        65.00       416.01       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np16 (query)                               351.00        70.25       421.25       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273-np23 (query)                               351.00        89.17       440.18       0.8155          1.0163            1.0145         6.33
IVF-SQ8-nl273 (self)                                     351.00       847.23     1_198.23       0.8121          1.0174            1.0155         6.33
IVF-SQ8-nl387-np19 (query)                               582.65        68.00       650.66       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387-np27 (query)                               582.65        79.77       662.43       0.8142          1.0165            1.0145         6.35
IVF-SQ8-nl387 (self)                                     582.65       779.08     1_361.73       0.8119          1.0175            1.0155         6.35
IVF-SQ8-nl547-np23 (query)                             1_129.69        64.19     1_193.87       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np27 (query)                             1_129.69        70.45     1_200.14       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547-np33 (query)                             1_129.69        81.47     1_211.16       0.8145          1.0165            1.0146         6.37
IVF-SQ8-nl547 (self)                                   1_129.69       766.37     1_896.06       0.8118          1.0175            1.0155         6.37
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
Exhaustive (query)                                        11.45       645.13       656.58       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.45     6_268.59     6_280.03       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    21.38     1_010.77     1_032.15       0.7893          1.0266            1.0244         5.15
Exhaustive-SQ8 (self)                                     21.38    10_011.69    10_033.07       0.7897          1.0281            1.0258         5.15
IVF-SQ8-nl273-np13 (query)                               354.45        62.93       417.38       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273-np16 (query)                               354.45        68.25       422.70       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273-np23 (query)                               354.45        91.91       446.36       0.7900          1.0265            1.0241         6.33
IVF-SQ8-nl273 (self)                                     354.45       862.67     1_217.12       0.7899          1.0281            1.0257         6.33
IVF-SQ8-nl387-np19 (query)                               590.47        63.89       654.37       0.7899          1.0265            1.0244         6.35
IVF-SQ8-nl387-np27 (query)                               590.47        80.39       670.86       0.7899          1.0265            1.0244         6.35
IVF-SQ8-nl387 (self)                                     590.47       784.50     1_374.97       0.7903          1.0280            1.0256         6.35
IVF-SQ8-nl547-np23 (query)                             1_098.08        64.26     1_162.35       0.7898          1.0265            1.0243         6.37
IVF-SQ8-nl547-np27 (query)                             1_098.08        71.53     1_169.61       0.7898          1.0265            1.0243         6.37
IVF-SQ8-nl547-np33 (query)                             1_098.08        78.67     1_176.75       0.7898          1.0265            1.0243         6.37
IVF-SQ8-nl547 (self)                                   1_098.08       773.77     1_871.86       0.7899          1.0280            1.0256         6.37
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
Exhaustive (query)                                        52.43     1_218.57     1_271.00       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         52.43    11_886.70    11_939.13       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    90.86     1_201.98     1_292.84       0.8797          1.0062            1.0051        18.88
Exhaustive-SQ8 (self)                                     90.86    12_088.87    12_179.73       0.8868          1.0073            1.0059        18.88
IVF-SQ8-nl273-np13 (query)                               693.78        82.40       776.17       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np16 (query)                               693.78        91.41       785.18       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273-np23 (query)                               693.78       116.59       810.37       0.8800          1.0061            1.0050        20.16
IVF-SQ8-nl273 (self)                                     693.78       944.72     1_638.49       0.8865          1.0073            1.0059        20.16
IVF-SQ8-nl387-np19 (query)                             1_246.90        94.42     1_341.32       0.8799          1.0061            1.0051        20.22
IVF-SQ8-nl387-np27 (query)                             1_246.90       106.95     1_353.85       0.8799          1.0061            1.0051        20.22
IVF-SQ8-nl387 (self)                                   1_246.90       887.81     2_134.71       0.8867          1.0073            1.0059        20.22
IVF-SQ8-nl547-np23 (query)                             2_467.24        91.15     2_558.39       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np27 (query)                             2_467.24        97.19     2_564.43       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547-np33 (query)                             2_467.24       113.23     2_580.47       0.8799          1.0061            1.0051        20.30
IVF-SQ8-nl547 (self)                                   2_467.24       883.82     3_351.06       0.8865          1.0073            1.0059        20.30
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
Exhaustive (query)                                        11.88       669.52       681.40       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.88     6_462.48     6_474.36       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    20.77     1_005.27     1_026.04       0.9256          1.0018            1.0009         5.15
HNSW-M16-ef100-s50 (query)                               818.90        51.05       869.95       0.9292          1.0179            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              818.90        86.41       905.31       0.9633          1.0075            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              818.90       171.97       990.87       0.9822          1.0042            1.0000        38.52
HNSW-M16-ef100 (self)                                    818.90       823.52     1_642.42       0.9637          1.0103            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_604.83        82.07     1_686.90       0.9593          1.0079            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_604.83        98.19     1_703.02       0.9828          1.0037            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_604.83       172.46     1_777.29       0.9923          1.0015            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_604.83       884.27     2_489.10       0.9832          1.0039            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_684.72        56.41     1_741.13       0.9697          1.0118            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_684.72       102.02     1_786.74       0.9883          1.0064            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_684.72       183.40     1_868.12       0.9955          1.0015            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_684.72       973.94     2_658.66       0.9885          1.0049            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_777.42        62.70     1_840.12       0.9734          1.0101            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_777.42       111.77     1_889.19       0.9897          1.0062            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_777.42       194.03     1_971.45       0.9960          1.0032            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_777.42     1_051.43     2_828.85       0.9901          1.0055            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          745.12        35.72       780.84       0.8776          1.0208            1.0032        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         745.12        70.26       815.38       0.9026          1.0106            1.0020        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         745.12       122.60       867.72       0.9151          1.0063            1.0014        26.89
HNSW-SQ8U-M16-ef100 (self)                               745.12       658.08     1_403.20       0.9020          1.0103            1.0020        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_395.48        37.21     1_432.69       0.8981          1.0163            1.0021        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_395.48        69.71     1_465.19       0.9142          1.0101            1.0014        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_395.48       132.65     1_528.13       0.9207          1.0050            1.0011        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_395.48       670.51     2_065.99       0.9142          1.0080            1.0014        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_526.05        43.65     1_569.70       0.9067          1.0100            1.0017        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_526.05        76.55     1_602.60       0.9184          1.0057            1.0012        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_526.05       146.21     1_672.26       0.9227          1.0034            1.0010        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_526.05       724.68     2_250.73       0.9179          1.0064            1.0012        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_594.54        45.07     1_639.60       0.9085          1.0098            1.0016        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_594.54        80.99     1_675.53       0.9192          1.0056            1.0012        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_594.54       145.48     1_740.02       0.9230          1.0036            1.0010        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_594.54       800.65     2_395.18       0.9187          1.0066            1.0012        45.20
HNSW-SQ8U-drop0 (query)                                1_383.56        69.05     1_452.61       0.8956          1.0062            1.0024        26.89
HNSW-SQ8U-drop0.001 (query)                            1_381.69        68.71     1_450.41       0.9149          1.0075            1.0014        26.89
HNSW-SQ8U-drop0.01 (query)                             1_433.40        79.91     1_513.31       0.8982          1.0151            1.0018        26.89
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
Exhaustive (query)                                        14.15       708.72       722.87       1.0000          1.0000            1.0000        18.88
Exhaustive (self)                                         14.15     6_968.74     6_982.89       1.0000          1.0000            1.0000        18.88
Exhaustive-SQ8 (query)                                    23.04       922.60       945.64       0.7397          1.0354            1.0161         5.15
HNSW-M16-ef100-s50 (query)                               842.19        50.03       892.21       0.9346          1.0177            1.0000        39.09
HNSW-M16-ef100-s100 (query)                              842.19       108.41       950.60       0.9693          1.0087            1.0000        39.09
HNSW-M16-ef100-s200 (query)                              842.19       164.73     1_006.92       0.9880          1.0036            1.0000        39.09
HNSW-M16-ef100 (self)                                    842.19       850.87     1_693.06       0.9699          1.0081            1.0000        39.09
HNSW-M16-ef200-s50 (query)                             1_715.63        63.94     1_779.57       0.9638          1.0129            1.0000        39.09
HNSW-M16-ef200-s100 (query)                            1_715.63       102.46     1_818.09       0.9871          1.0040            1.0000        39.09
HNSW-M16-ef200-s200 (query)                            1_715.63       180.77     1_896.40       0.9950          1.0019            1.0000        39.09
HNSW-M16-ef200 (self)                                  1_715.63       928.18     2_643.80       0.9870          1.0056            1.0000        39.09
HNSW-M24-ef200-s50 (query)                             1_874.87        59.92     1_934.79       0.9737          1.0084            1.0000        48.23
HNSW-M24-ef200-s100 (query)                            1_874.87       104.64     1_979.51       0.9912          1.0029            1.0000        48.23
HNSW-M24-ef200-s200 (query)                            1_874.87       188.92     2_063.79       0.9968          1.0025            1.0000        48.23
HNSW-M24-ef200 (self)                                  1_874.87     1_017.54     2_892.41       0.9911          1.0023            1.0000        48.23
HNSW-M32-ef200-s50 (query)                             1_781.11        60.92     1_842.03       0.9749          1.0668            1.0000        57.37
HNSW-M32-ef200-s100 (query)                            1_781.11       108.86     1_889.97       0.9916          1.0065            1.0000        57.37
HNSW-M32-ef200-s200 (query)                            1_781.11       200.65     1_981.76       0.9973          1.0002            1.0000        57.37
HNSW-M32-ef200 (self)                                  1_781.11     1_056.14     2_837.25       0.9915          1.0171            1.0000        57.37
HNSW-SQ8U-M16-ef100-s50 (query)                          747.09        36.48       783.57       0.6837          1.0630            1.0292        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         747.09        74.91       822.00       0.7082          1.0483            1.0242        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         747.09       128.87       875.96       0.7222          1.0440            1.0208        26.89
HNSW-SQ8U-M16-ef100 (self)                               747.09       636.37     1_383.46       0.7077          1.0510            1.0240        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_451.33        37.88     1_489.21       0.7072          1.0545            1.0234        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_451.33        72.30     1_523.63       0.7239          1.0470            1.0201        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_451.33       131.71     1_583.04       0.7316          1.0402            1.0186        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_451.33       683.05     2_134.38       0.7235          1.0456            1.0199        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_556.81        44.11     1_600.92       0.7190          1.0400            1.0205        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_556.81        80.34     1_637.15       0.7309          1.0380            1.0183        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_556.81       139.70     1_696.51       0.7356          1.0371            1.0173        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_556.81       760.06     2_316.87       0.7298          1.0386            1.0181        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_673.76        49.80     1_723.56       0.7216          1.0981            1.0195        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_673.76        83.82     1_757.58       0.7330          1.0374            1.0175        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_673.76       148.92     1_822.68       0.7369          1.0364            1.0168        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_673.76       788.79     2_462.55       0.7318          1.0495            1.0174        45.20
HNSW-SQ8U-drop0 (query)                                1_452.15        74.71     1_526.86       0.6661          1.0615            1.0302        26.89
HNSW-SQ8U-drop0.001 (query)                            1_474.53        71.44     1_545.98       0.7234          1.0499            1.0201        26.89
HNSW-SQ8U-drop0.01 (query)                             1_406.76        69.16     1_475.93       0.6905          1.0512            1.0259        26.89
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
Exhaustive (query)                                        11.44       696.89       708.33       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.44     6_840.86     6_852.30       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.08     1_018.90     1_036.98       0.8146          1.0165            1.0148         5.15
HNSW-M16-ef100-s50 (query)                               823.39        51.70       875.08       0.9964          1.6008            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              823.39        92.04       915.42       0.9985          1.0002            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              823.39       160.15       983.54       0.9989          1.0001            1.0000        38.52
HNSW-M16-ef100 (self)                                    823.39       853.88     1_677.27       0.9984          1.0156            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_432.15        50.75     1_482.90       0.9911         13.6551            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_432.15        86.78     1_518.94       0.9990          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_432.15       151.15     1_583.31       0.9991          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_432.15       798.40     2_230.56       0.9989          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_510.94        52.90     1_563.84       0.9983          1.0001            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_510.94       107.34     1_618.28       0.9990          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_510.94       163.01     1_673.95       0.9991          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_510.94       943.56     2_454.50       0.9990          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_559.51        60.81     1_620.32       0.9985          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_559.51        90.99     1_650.50       0.9990          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_559.51       165.34     1_724.85       0.9991          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_559.51       871.46     2_430.97       0.9990          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          741.23        47.23       788.46       0.8110          1.0722            1.0150        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         741.23        66.20       807.44       0.8120          1.0177            1.0149        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         741.23       120.78       862.02       0.8121          1.0177            1.0149        26.89
HNSW-SQ8U-M16-ef100 (self)                               741.23       607.19     1_348.42       0.8094          1.0190            1.0157        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_326.30        35.82     1_362.12       0.8138          1.2576            1.0149        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_326.30        66.21     1_392.51       0.8145          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_326.30       115.97     1_442.27       0.8146          1.0165            1.0148        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_326.30       608.81     1_935.11       0.8119          1.0216            1.0155        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_393.82        43.06     1_436.88       0.8143          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_393.82        70.89     1_464.71       0.8145          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_393.82       127.04     1_520.85       0.8146          1.0165            1.0148        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_393.82       669.16     2_062.98       0.8119          1.0175            1.0155        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_453.03        41.30     1_494.34       0.8143          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_453.03        72.40     1_525.43       0.8145          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_453.03       129.40     1_582.43       0.8146          1.0165            1.0148        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_453.03       705.45     2_158.48       0.8119          1.0175            1.0155        45.20
HNSW-SQ8U-drop0 (query)                                1_328.32        65.40     1_393.72       0.8082          1.0176            1.0157        26.89
HNSW-SQ8U-drop0.001 (query)                            1_295.66        66.95     1_362.61       0.8140          1.2462            1.0148        26.89
HNSW-SQ8U-drop0.01 (query)                             1_313.06        70.43     1_383.49       0.8049          1.0195            1.0163        26.89
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
Exhaustive (query)                                        11.25       695.84       707.09       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.25     6_988.30     6_999.55       1.0000          1.0000            1.0000        18.31
Exhaustive-SQ8 (query)                                    18.64     1_015.92     1_034.56       0.7893          1.0266            1.0244         5.15
HNSW-M16-ef100-s50 (query)                               925.37        60.91       986.27       0.9976          1.0001            1.0000        38.52
HNSW-M16-ef100-s100 (query)                              925.37        98.48     1_023.85       0.9993          1.0000            1.0000        38.52
HNSW-M16-ef100-s200 (query)                              925.37       184.25     1_109.62       0.9995          1.0000            1.0000        38.52
HNSW-M16-ef100 (self)                                    925.37       918.63     1_844.00       0.9993          1.0000            1.0000        38.52
HNSW-M16-ef200-s50 (query)                             1_615.93        55.23     1_671.16       0.9981          1.0001            1.0000        38.52
HNSW-M16-ef200-s100 (query)                            1_615.93        97.02     1_712.95       0.9994          1.0000            1.0000        38.52
HNSW-M16-ef200-s200 (query)                            1_615.93       175.27     1_791.20       0.9995          1.0000            1.0000        38.52
HNSW-M16-ef200 (self)                                  1_615.93       961.52     2_577.46       0.9994          1.0000            1.0000        38.52
HNSW-M24-ef200-s50 (query)                             1_748.64        62.92     1_811.55       0.9989          1.0000            1.0000        47.66
HNSW-M24-ef200-s100 (query)                            1_748.64       111.59     1_860.22       0.9995          1.0000            1.0000        47.66
HNSW-M24-ef200-s200 (query)                            1_748.64       197.86     1_946.50       0.9995          1.0000            1.0000        47.66
HNSW-M24-ef200 (self)                                  1_748.64     1_018.13     2_766.77       0.9994          1.0000            1.0000        47.66
HNSW-M32-ef200-s50 (query)                             1_729.54        61.40     1_790.93       0.9990          1.0000            1.0000        56.80
HNSW-M32-ef200-s100 (query)                            1_729.54       107.86     1_837.40       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200-s200 (query)                            1_729.54       187.87     1_917.40       0.9995          1.0000            1.0000        56.80
HNSW-M32-ef200 (self)                                  1_729.54     1_052.86     2_782.40       0.9994          1.0000            1.0000        56.80
HNSW-SQ8U-M16-ef100-s50 (query)                          808.28        38.79       847.06       0.7888          1.0268            1.0245        26.89
HNSW-SQ8U-M16-ef100-s100 (query)                         808.28        77.32       885.60       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100-s200 (query)                         808.28       134.55       942.83       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef100 (self)                               808.28       692.61     1_500.89       0.7896          1.0282            1.0258        26.89
HNSW-SQ8U-M16-ef200-s50 (query)                        1_451.23        38.74     1_489.96       0.7890          1.0267            1.0245        26.89
HNSW-SQ8U-M16-ef200-s100 (query)                       1_451.23        74.58     1_525.81       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200-s200 (query)                       1_451.23       137.31     1_588.54       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-M16-ef200 (self)                             1_451.23       711.41     2_162.64       0.7897          1.0281            1.0258        26.89
HNSW-SQ8U-M24-ef200-s50 (query)                        1_518.65        53.03     1_571.69       0.7892          1.0267            1.0244        35.80
HNSW-SQ8U-M24-ef200-s100 (query)                       1_518.65        79.24     1_597.90       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200-s200 (query)                       1_518.65       142.74     1_661.39       0.7893          1.0266            1.0244        35.80
HNSW-SQ8U-M24-ef200 (self)                             1_518.65       763.31     2_281.96       0.7897          1.0281            1.0258        35.80
HNSW-SQ8U-M32-ef200-s50 (query)                        1_637.08        48.12     1_685.20       0.7892          1.0267            1.0244        45.20
HNSW-SQ8U-M32-ef200-s100 (query)                       1_637.08        90.92     1_728.00       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200-s200 (query)                       1_637.08       149.68     1_786.76       0.7893          1.0266            1.0244        45.20
HNSW-SQ8U-M32-ef200 (self)                             1_637.08       814.67     2_451.75       0.7897          1.0281            1.0258        45.20
HNSW-SQ8U-drop0 (query)                                1_442.00        73.18     1_515.18       0.7861          1.0279            1.0254        26.89
HNSW-SQ8U-drop0.001 (query)                            1_489.76        74.75     1_564.51       0.7893          1.0266            1.0244        26.89
HNSW-SQ8U-drop0.01 (query)                             1_444.99        72.35     1_517.34       0.7830          1.0294            1.0262        26.89
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
Exhaustive (query)                                        56.34     1_288.75     1_345.09       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         56.34    12_972.00    13_028.34       1.0000          1.0000            1.0000        73.24
Exhaustive-SQ8 (query)                                    86.39     1_236.96     1_323.35       0.9341          1.0074            1.0036        18.88
HNSW-M16-ef100-s50 (query)                             1_384.47        82.39     1_466.86       0.9927          1.0366            1.0000        93.45
HNSW-M16-ef100-s100 (query)                            1_384.47       151.87     1_536.34       0.9953          1.0222            1.0000        93.45
HNSW-M16-ef100-s200 (query)                            1_384.47       247.89     1_632.37       0.9971          1.0103            1.0000        93.45
HNSW-M16-ef100 (self)                                  1_384.47     1_354.71     2_739.18       0.9952          1.0222            1.0000        93.45
HNSW-M16-ef200-s50 (query)                             2_489.59        83.89     2_573.48       0.9966          1.0214            1.0000        93.45
HNSW-M16-ef200-s100 (query)                            2_489.59       146.83     2_636.42       0.9985          1.0070            1.0000        93.45
HNSW-M16-ef200-s200 (query)                            2_489.59       269.31     2_758.90       0.9992          1.0036            1.0000        93.45
HNSW-M16-ef200 (self)                                  2_489.59     1_418.56     3_908.15       0.9982          1.0101            1.0000        93.45
HNSW-M24-ef200-s50 (query)                             2_663.04        90.30     2_753.34       0.9981          1.0080            1.0000       102.59
HNSW-M24-ef200-s100 (query)                            2_663.04       156.27     2_819.31       0.9991          1.0035            1.0000       102.59
HNSW-M24-ef200-s200 (query)                            2_663.04       269.17     2_932.22       0.9996          1.0008            1.0000       102.59
HNSW-M24-ef200 (self)                                  2_663.04     1_491.95     4_155.00       0.9992          1.0034            1.0000       102.59
HNSW-M32-ef200-s50 (query)                             2_698.80        93.04     2_791.84       0.9985          1.0067            1.0000       111.73
HNSW-M32-ef200-s100 (query)                            2_698.80       154.43     2_853.22       0.9992          1.0030            1.0000       111.73
HNSW-M32-ef200-s200 (query)                            2_698.80       269.02     2_967.81       0.9996          1.0012            1.0000       111.73
HNSW-M32-ef200 (self)                                  2_698.80     1_499.45     4_198.25       0.9993          1.0032            1.0000       111.73
HNSW-SQ8U-M16-ef100-s50 (query)                          856.57        41.44       898.01       0.9288          1.0370            1.0038        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         856.57        73.67       930.24       0.9308          1.0223            1.0038        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         856.57       128.21       984.78       0.9320          1.0146            1.0038        40.63
HNSW-SQ8U-M16-ef100 (self)                               856.57       679.95     1_536.52       0.9306          1.0234            1.0038        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_565.68        51.44     1_617.12       0.9303          1.0333            1.0037        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_565.68        83.29     1_648.97       0.9324          1.0190            1.0036        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_565.68       136.66     1_702.34       0.9334          1.0119            1.0036        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_565.68       723.23     2_288.91       0.9320          1.0211            1.0037        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_675.29        54.87     1_730.16       0.9324          1.0172            1.0036        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_675.29        73.35     1_748.64       0.9331          1.0128            1.0036        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_675.29       136.30     1_811.59       0.9338          1.0092            1.0036        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_675.29       729.35     2_404.64       0.9328          1.0133            1.0036        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_746.63        44.98     1_791.61       0.9330          1.0136            1.0036        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_746.63        74.79     1_821.42       0.9338          1.0088            1.0036        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_746.63       136.20     1_882.83       0.9339          1.0077            1.0036        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_746.63       742.14     2_488.77       0.9333          1.0107            1.0036        58.94
HNSW-SQ8U-drop0 (query)                                1_579.82        69.11     1_648.93       0.8645          1.0423            1.0221        40.63
HNSW-SQ8U-drop0.001 (query)                            1_522.11        82.40     1_604.50       0.9324          1.0183            1.0036        40.63
HNSW-SQ8U-drop0.01 (query)                             1_540.92        74.23     1_615.15       0.9320          1.0392            1.0020        40.63
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
Exhaustive (query)                                        56.72     1_338.93     1_395.65       1.0000          1.0000            1.0000        73.81
Exhaustive (self)                                         56.72    13_819.22    13_875.94       1.0000          1.0000            1.0000        73.81
Exhaustive-SQ8 (query)                                    92.21     1_251.90     1_344.11       0.6675          1.3471            1.1612        18.88
HNSW-M16-ef100-s50 (query)                             1_259.66        69.13     1_328.79       0.9929          1.1335            1.0000        94.02
HNSW-M16-ef100-s100 (query)                            1_259.66       119.15     1_378.82       0.9964          1.0381            1.0000        94.02
HNSW-M16-ef100-s200 (query)                            1_259.66       211.56     1_471.22       0.9979          1.0176            1.0000        94.02
HNSW-M16-ef100 (self)                                  1_259.66     1_162.98     2_422.64       0.9957          1.0579            1.0000        94.02
HNSW-M16-ef200-s50 (query)                             2_341.65        73.45     2_415.10       0.9926          1.1623            1.0000        94.02
HNSW-M16-ef200-s100 (query)                            2_341.65       144.40     2_486.05       0.9956          1.0849            1.0000        94.02
HNSW-M16-ef200-s200 (query)                            2_341.65       253.97     2_595.62       0.9980          1.0274            1.0000        94.02
HNSW-M16-ef200 (self)                                  2_341.65     1_281.78     3_623.43       0.9957          1.0844            1.0000        94.02
HNSW-M24-ef200-s50 (query)                             2_426.96        76.04     2_503.00       0.9986          1.0157            1.0000       103.16
HNSW-M24-ef200-s100 (query)                            2_426.96       127.26     2_554.22       0.9992          1.0098            1.0000       103.16
HNSW-M24-ef200-s200 (query)                            2_426.96       228.41     2_655.38       0.9998          1.0018            1.0000       103.16
HNSW-M24-ef200 (self)                                  2_426.96     1_251.72     3_678.68       0.9992          1.0092            1.0000       103.16
HNSW-M32-ef200-s50 (query)                             2_530.95        77.77     2_608.72       0.9991          1.0125            1.0000       112.31
HNSW-M32-ef200-s100 (query)                            2_530.95       134.26     2_665.21       0.9995          1.0039            1.0000       112.31
HNSW-M32-ef200-s200 (query)                            2_530.95       233.97     2_764.93       0.9997          1.0012            1.0000       112.31
HNSW-M32-ef200 (self)                                  2_530.95     1_270.01     3_800.96       0.9993          1.0096            1.0000       112.31
HNSW-SQ8U-M16-ef100-s50 (query)                          840.90        42.73       883.62       0.6629          1.4294            1.1647        40.63
HNSW-SQ8U-M16-ef100-s100 (query)                         840.90        69.95       910.85       0.6649          1.3926            1.1632        40.63
HNSW-SQ8U-M16-ef100-s200 (query)                         840.90       124.82       965.72       0.6662          1.3622            1.1624        40.63
HNSW-SQ8U-M16-ef100 (self)                               840.90       653.53     1_494.43       0.6647          1.3881            1.1635        40.63
HNSW-SQ8U-M16-ef200-s50 (query)                        1_498.70        47.28     1_545.98       0.6636          1.4405            1.1639        40.63
HNSW-SQ8U-M16-ef200-s100 (query)                       1_498.70        70.35     1_569.05       0.6653          1.3999            1.1627        40.63
HNSW-SQ8U-M16-ef200-s200 (query)                       1_498.70       127.07     1_625.78       0.6663          1.3752            1.1622        40.63
HNSW-SQ8U-M16-ef200 (self)                             1_498.70       667.01     2_165.71       0.6652          1.3938            1.1630        40.63
HNSW-SQ8U-M24-ef200-s50 (query)                        1_633.77        46.11     1_679.88       0.6666          1.3621            1.1618        49.53
HNSW-SQ8U-M24-ef200-s100 (query)                       1_633.77        74.26     1_708.03       0.6671          1.3551            1.1614        49.53
HNSW-SQ8U-M24-ef200-s200 (query)                       1_633.77       132.45     1_766.22       0.6674          1.3484            1.1613        49.53
HNSW-SQ8U-M24-ef200 (self)                             1_633.77       703.88     2_337.65       0.6665          1.3613            1.1619        49.53
HNSW-SQ8U-M32-ef200-s50 (query)                        1_691.57        65.60     1_757.17       0.6659          1.4225            1.1620        58.94
HNSW-SQ8U-M32-ef200-s100 (query)                       1_691.57        79.50     1_771.07       0.6666          1.3889            1.1617        58.94
HNSW-SQ8U-M32-ef200-s200 (query)                       1_691.57       135.95     1_827.53       0.6675          1.3494            1.1613        58.94
HNSW-SQ8U-M32-ef200 (self)                             1_691.57       737.63     2_429.20       0.6665          1.3977            1.1618        58.94
HNSW-SQ8U-drop0 (query)                                1_505.91        73.61     1_579.52       0.6187          1.5373            1.2372        40.63
HNSW-SQ8U-drop0.001 (query)                            1_478.52        74.87     1_553.39       0.6648          1.4116            1.1630        40.63
HNSW-SQ8U-drop0.01 (query)                             1_530.60        73.60     1_604.20       0.6799          1.3333            1.1494        40.63
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
Exhaustive (query)                                        32.82       693.43       726.25       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.82     2_314.94     2_347.76       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                644.91       670.89     1_315.81       0.2580          1.1826            1.1592         1.01
Exhaustive-PQ-m16 (self)                                 644.91     2_250.01     2_894.92       0.2365          1.1998            1.1748         1.01
Exhaustive-PQ-m32 (query)                              1_158.15     1_521.62     2_679.77       0.2961          1.1446            1.1423         1.78
Exhaustive-PQ-m32 (self)                               1_158.15     5_055.09     6_213.23       0.2627          1.1633            1.1601         1.78
Exhaustive-PQ-m64 (query)                              1_829.42     3_613.50     5_442.92       0.3610          1.1111            1.1080         3.30
Exhaustive-PQ-m64 (self)                               1_829.42    11_953.38    13_782.80       0.3106          1.1303            1.1270         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_429.34       202.10     1_631.44       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_429.34       313.45     1_742.79       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_429.34       422.59     1_851.93       0.3713          1.0979            1.1001         1.17
IVF-PQ-nl158-m16 (self)                                1_429.34     1_454.56     2_883.90       0.3041          1.1282            1.1332         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_877.29       382.15     2_259.44       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_877.29       561.16     2_438.45       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_877.29       771.51     2_648.80       0.4812          1.0610            1.0583         1.93
IVF-PQ-nl158-m32 (self)                                1_877.29     2_545.00     4_422.30       0.4068          1.0804            1.0800         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_645.01       634.81     3_279.82       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_645.01     1_004.23     3_649.24       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_645.01     1_360.37     4_005.39       0.6903          1.0199            1.0166         3.46
IVF-PQ-nl158-m64 (self)                                2_645.01     4_553.04     7_198.05       0.6338          1.0271            1.0243         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_173.54       299.68     1_473.22       0.3870          1.0887            1.0895         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_173.54       361.59     1_535.13       0.3869          1.0887            1.0895         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_173.54       530.06     1_703.60       0.3869          1.0887            1.0895         1.23
IVF-PQ-nl223-m16 (self)                                1_173.54     1_776.67     2_950.21       0.3098          1.1230            1.1272         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_657.71       519.77     2_177.47       0.4975          1.0564            1.0516         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_657.71       627.62     2_285.32       0.4975          1.0564            1.0517         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_657.71       907.41     2_565.12       0.4975          1.0564            1.0517         2.00
IVF-PQ-nl223-m32 (self)                                1_657.71     3_021.72     4_679.42       0.4146          1.0780            1.0755         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_251.77       903.92     3_155.69       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_251.77     1_122.14     3_373.91       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_251.77     1_621.90     3_873.68       0.6979          1.0199            1.0155         3.52
IVF-PQ-nl223-m64 (self)                                2_251.77     5_421.30     7_673.07       0.6386          1.0273            1.0235         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_393.43       378.26     1_771.68       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_393.43       418.18     1_811.61       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_393.43       596.67     1_990.09       0.3983          1.0835            1.0847         1.32
IVF-PQ-nl316-m16 (self)                                1_393.43     1_989.29     3_382.72       0.3156          1.1188            1.1227         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_942.72       666.15     2_608.87       0.5114          1.0520            1.0487         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_942.72       732.30     2_675.01       0.5114          1.0519            1.0487         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_942.72     1_049.19     2_991.91       0.5114          1.0519            1.0487         2.09
IVF-PQ-nl316-m32 (self)                                1_942.72     3_481.80     5_424.52       0.4236          1.0742            1.0728         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_512.96     1_155.86     3_668.81       0.7073          1.0175            1.0146         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_512.96     1_299.09     3_812.05       0.7073          1.0174            1.0146         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_512.96     1_871.75     4_384.70       0.7073          1.0174            1.0146         3.61
IVF-PQ-nl316-m64 (self)                                2_512.96     6_229.97     8_742.93       0.6490          1.0248            1.0221         3.61
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
Exhaustive (query)                                        68.17     1_255.74     1_323.91       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.17     4_172.69     4_240.87       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                905.70       685.44     1_591.13       0.2444          1.1297            1.1195         1.26
Exhaustive-PQ-m16 (self)                                 905.70     2_234.01     3_139.70       0.2278          1.1396            1.1265         1.26
Exhaustive-PQ-m32 (query)                              1_268.68     1_534.89     2_803.56       0.2648          1.1130            1.1155         2.03
Exhaustive-PQ-m32 (self)                               1_268.68     5_074.87     6_343.54       0.2433          1.1221            1.1232         2.03
Exhaustive-PQ-m64 (query)                              2_160.70     3_888.82     6_049.52       0.2955          1.0990            1.1029         3.55
Exhaustive-PQ-m64 (self)                               2_160.70    12_016.08    14_176.78       0.2627          1.1103            1.1142         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_543.48       280.49     2_823.97       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_543.48       432.91     2_976.39       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_543.48       575.72     3_119.19       0.3076          1.0878            1.0922         1.57
IVF-PQ-nl158-m16 (self)                                2_543.48     1_933.23     4_476.71       0.2624          1.1080            1.1146         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_841.62       404.22     3_245.84       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_841.62       659.72     3_501.34       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_841.62       863.91     3_705.52       0.3545          1.0712            1.0721         2.34
IVF-PQ-nl158-m32 (self)                                2_841.62     2_836.75     5_678.37       0.2913          1.0913            1.0953         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_797.22       770.28     4_567.50       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_797.22     1_157.72     4_954.94       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_797.22     1_581.96     5_379.18       0.4625          1.0458            1.0423         3.86
IVF-PQ-nl158-m64 (self)                                3_797.22     5_224.32     9_021.55       0.3902          1.0583            1.0572         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_829.80       418.91     2_248.71       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_829.80       525.91     2_355.70       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_829.80       727.10     2_556.90       0.3167          1.0827            1.0852         1.70
IVF-PQ-nl223-m16 (self)                                1_829.80     2_414.31     4_244.11       0.2659          1.1044            1.1096         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_170.83       585.09     2_755.92       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_170.83       712.72     2_883.55       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_170.83     1_038.14     3_208.97       0.3686          1.0655            1.0657         2.46
IVF-PQ-nl223-m32 (self)                                2_170.83     3_427.50     5_598.33       0.2945          1.0892            1.0917         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_062.17     1_057.59     4_119.76       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_062.17     1_310.82     4_372.99       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_062.17     1_885.04     4_947.21       0.4766          1.0429            1.0387         3.99
IVF-PQ-nl223-m64 (self)                                3_062.17     6_246.17     9_308.34       0.3958          1.0576            1.0552         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_132.73       538.15     2_670.88       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_132.73       598.03     2_730.75       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_132.73       836.20     2_968.93       0.3273          1.0764            1.0805         1.88
IVF-PQ-nl316-m16 (self)                                2_132.73     2_770.26     4_902.99       0.2710          1.0999            1.1061         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_580.92       738.17     3_319.09       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_580.92       832.41     3_413.33       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_580.92     1_178.75     3_759.67       0.3790          1.0612            1.0619         2.65
IVF-PQ-nl316-m32 (self)                                2_580.92     3_879.50     6_460.42       0.2993          1.0859            1.0890         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_479.84     1_369.82     4_849.66       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_479.84     1_541.34     5_021.18       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_479.84     2_174.90     5_654.74       0.4880          1.0396            1.0362         4.17
IVF-PQ-nl316-m64 (self)                                3_479.84     7_195.72    10_675.56       0.4046          1.0543            1.0533         4.17
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
Exhaustive (query)                                       100.10     1_821.96     1_922.05       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.10     5_966.26     6_066.36       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_156.99       723.52     1_880.51       0.2346          1.1094            1.0999         1.51
Exhaustive-PQ-m16 (self)                               1_156.99     2_250.43     3_407.42       0.2205          1.1179            1.1047         1.51
Exhaustive-PQ-m32 (query)                              1_811.01     1_570.67     3_381.68       0.2566          1.0942            1.0974         2.28
Exhaustive-PQ-m32 (self)                               1_811.01     5_151.76     6_962.77       0.2390          1.1012            1.1020         2.28
Exhaustive-PQ-m64 (query)                              2_658.15     3_629.47     6_287.62       0.2774          1.0854            1.0909         3.80
Exhaustive-PQ-m64 (self)                               2_658.15    12_161.28    14_819.44       0.2516          1.0934            1.0980         3.80
Exhaustive-PQ-m128 (query)                             4_532.02     7_955.04    12_487.06       0.3160          1.0712            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_532.02    26_497.51    31_029.53       0.2754          1.0816            1.0859         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_351.43       368.48     3_719.91       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_351.43       591.16     3_942.59       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_351.43       770.63     4_122.05       0.2856          1.0780            1.0843         1.98
IVF-PQ-nl158-m16 (self)                                3_351.43     2_577.86     5_929.29       0.2511          1.0937            1.1010         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_844.87       548.25     4_393.12       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_844.87       850.22     4_695.10       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_844.87     1_167.64     5_012.51       0.3150          1.0674            1.0715         2.74
IVF-PQ-nl158-m32 (self)                                3_844.87     3_869.76     7_714.63       0.2628          1.0854            1.0912         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_807.44       832.36     5_639.80       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_807.44     1_312.41     6_119.84       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_807.44     1_790.38     6_597.82       0.3781          1.0518            1.0512         4.27
IVF-PQ-nl158-m64 (self)                                4_807.44     5_957.67    10_765.11       0.3104          1.0661            1.0674         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_700.56     1_610.45     8_311.01       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_700.56     2_519.79     9_220.35       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_700.56     3_461.52    10_162.07       0.5351          1.0270            1.0230         7.32
IVF-PQ-nl158-m128 (self)                               6_700.56    11_506.32    18_206.88       0.4636          1.0342            1.0319         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_487.42       522.34     3_009.76       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_487.42       636.11     3_123.53       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_487.42       938.70     3_426.12       0.2960          1.0725            1.0764         2.17
IVF-PQ-nl223-m16 (self)                                2_487.42     3_067.68     5_555.10       0.2553          1.0892            1.0954         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_932.91       744.82     3_677.73       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_932.91       914.90     3_847.81       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_932.91     1_336.69     4_269.60       0.3305          1.0612            1.0632         2.93
IVF-PQ-nl223-m32 (self)                                2_932.91     4_386.17     7_319.08       0.2675          1.0815            1.0861         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_902.78     1_173.99     5_076.77       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_902.78     1_447.07     5_349.85       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_902.78     2_099.81     6_002.59       0.3928          1.0479            1.0458         4.46
IVF-PQ-nl223-m64 (self)                                3_902.78     6_918.21    10_820.99       0.3132          1.0650            1.0653         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_829.15     2_310.97     8_140.12       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_829.15     2_863.63     8_692.79       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_829.15     4_183.46    10_012.61       0.5460          1.0258            1.0213         7.51
IVF-PQ-nl223-m128 (self)                               5_829.15    13_818.64    19_647.80       0.4701          1.0335            1.0307         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_084.27       671.14     3_755.41       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_084.27       740.70     3_824.97       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_084.27     1_070.56     4_154.84       0.3048          1.0680            1.0728         2.44
IVF-PQ-nl316-m16 (self)                                3_084.27     3_518.80     6_603.07       0.2597          1.0853            1.0916         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_344.06       997.86     4_341.92       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_344.06     1_104.23     4_448.29       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_344.06     1_585.31     4_929.37       0.3357          1.0583            1.0611         3.21
IVF-PQ-nl316-m32 (self)                                3_344.06     5_305.16     8_649.22       0.2693          1.0793            1.0842         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_373.00     1_654.90     6_027.89       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_373.00     1_737.54     6_110.53       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_373.00     2_562.85     6_935.84       0.4011          1.0454            1.0437         4.73
IVF-PQ-nl316-m64 (self)                                4_373.00     8_272.07    12_645.06       0.3194          1.0624            1.0635         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_476.01     3_005.18     9_481.20       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_476.01     3_362.52     9_838.54       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_476.01     4_833.64    11_309.65       0.5557          1.0234            1.0203         7.78
IVF-PQ-nl316-m128 (self)                               6_476.01    16_089.05    22_565.06       0.4776          1.0316            1.0299         7.78
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
Exhaustive (query)                                        32.94       696.38       729.32       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.94     2_323.79     2_356.73       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                656.29       665.55     1_321.84       0.2931          1.2577            1.2510         1.01
Exhaustive-PQ-m16 (self)                                 656.29     2_196.03     2_852.32       0.2301          1.3863            1.3798         1.01
Exhaustive-PQ-m32 (query)                              1_324.48     1_523.76     2_848.23       0.4007          1.1658            1.1600         1.78
Exhaustive-PQ-m32 (self)                               1_324.48     5_049.48     6_373.95       0.3180          1.2686            1.2616         1.78
Exhaustive-PQ-m64 (query)                              1_856.48     3_601.42     5_457.91       0.5384          1.0881            1.0842         3.30
Exhaustive-PQ-m64 (self)                               1_856.48    11_940.63    13_797.11       0.4587          1.1480            1.1426         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_482.26       194.64     1_676.91       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_482.26       301.40     1_783.66       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_482.26       417.68     1_899.95       0.5336          1.0886            1.0855         1.17
IVF-PQ-nl158-m16 (self)                                1_482.26     1_391.31     2_873.57       0.4286          1.1644            1.1604         1.17
IVF-PQ-nl158-m32-np7 (query)                           1_983.62       352.54     2_336.16       0.6748          1.0398            1.0375         1.93
IVF-PQ-nl158-m32-np12 (query)                          1_983.62       554.94     2_538.55       0.6748          1.0398            1.0375         1.93
IVF-PQ-nl158-m32-np17 (query)                          1_983.62       769.23     2_752.84       0.6748          1.0398            1.0375         1.93
IVF-PQ-nl158-m32 (self)                                1_983.62     2_497.70     4_481.32       0.6053          1.0692            1.0642         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_599.72       623.76     3_223.48       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_599.72       993.70     3_593.42       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_599.72     1_374.03     3_973.75       0.8332          1.0095            1.0082         3.46
IVF-PQ-nl158-m64 (self)                                2_599.72     4_577.10     7_176.82       0.7986          1.0164            1.0142         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_229.66       311.84     1_541.50       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_229.66       363.37     1_593.03       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_229.66       543.46     1_773.11       0.5372          1.0874            1.0848         1.23
IVF-PQ-nl223-m16 (self)                                1_229.66     1_741.73     2_971.39       0.4250          1.1675            1.1633         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_762.05       509.81     2_271.86       0.6754          1.0394            1.0372         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_762.05       633.67     2_395.72       0.6755          1.0394            1.0372         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_762.05       918.00     2_680.05       0.6755          1.0394            1.0372         2.00
IVF-PQ-nl223-m32 (self)                                1_762.05     3_043.33     4_805.38       0.6039          1.0699            1.0652         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_277.46       890.37     3_167.83       0.8360          1.0092            1.0079         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_277.46     1_117.90     3_395.37       0.8361          1.0091            1.0079         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_277.46     1_655.90     3_933.36       0.8361          1.0091            1.0079         3.52
IVF-PQ-nl223-m64 (self)                                2_277.46     5_484.17     7_761.63       0.8003          1.0160            1.0139         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_465.75       380.78     1_846.53       0.5373          1.0875            1.0847         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_465.75       421.52     1_887.27       0.5374          1.0875            1.0847         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_465.75       601.67     2_067.42       0.5374          1.0875            1.0847         1.32
IVF-PQ-nl316-m16 (self)                                1_465.75     2_025.27     3_491.02       0.4157          1.1742            1.1702         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_895.98       669.14     2_565.12       0.6784          1.0387            1.0364         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_895.98       726.57     2_622.55       0.6785          1.0387            1.0364         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_895.98     1_043.74     2_939.72       0.6785          1.0387            1.0364         2.09
IVF-PQ-nl316-m32 (self)                                1_895.98     3_488.38     5_384.35       0.6009          1.0711            1.0665         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_535.18     1_153.24     3_688.42       0.8382          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_535.18     1_293.59     3_828.78       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_535.18     1_898.07     4_433.26       0.8383          1.0089            1.0077         3.61
IVF-PQ-nl316-m64 (self)                                2_535.18     6_343.82     8_879.01       0.8029          1.0155            1.0136         3.61
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
Exhaustive (query)                                        68.49     1_266.42     1_334.91       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.49     4_164.63     4_233.13       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                888.12       683.40     1_571.52       0.2128          1.2291            1.2259         1.26
Exhaustive-PQ-m16 (self)                                 888.12     2_241.61     3_129.73       0.1773          1.3099            1.3107         1.26
Exhaustive-PQ-m32 (query)                              1_274.63     1_540.51     2_815.15       0.2800          1.1736            1.1699         2.03
Exhaustive-PQ-m32 (self)                               1_274.63     5_063.20     6_337.83       0.2228          1.2514            1.2496         2.03
Exhaustive-PQ-m64 (query)                              2_170.75     3_622.60     5_793.35       0.3753          1.1186            1.1154         3.55
Exhaustive-PQ-m64 (self)                               2_170.75    12_012.03    14_182.78       0.2986          1.1838            1.1817         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_412.60       269.24     2_681.84       0.3772          1.1193            1.1182         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_412.60       423.33     2_835.93       0.3772          1.1193            1.1182         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_412.60       573.18     2_985.78       0.3772          1.1193            1.1182         1.57
IVF-PQ-nl158-m16 (self)                                2_412.60     1_913.52     4_326.11       0.2721          1.2081            1.2110         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_816.34       403.51     3_219.86       0.4903          1.0728            1.0712         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_816.34       640.50     3_456.84       0.4903          1.0728            1.0712         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_816.34       847.59     3_663.94       0.4903          1.0728            1.0712         2.34
IVF-PQ-nl158-m32 (self)                                2_816.34     2_814.25     5_630.59       0.3931          1.1247            1.1233         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_730.65       720.96     4_451.61       0.6306          1.0351            1.0335         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_730.65     1_132.84     4_863.49       0.6306          1.0351            1.0335         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_730.65     1_612.34     5_342.99       0.6306          1.0351            1.0335         3.86
IVF-PQ-nl158-m64 (self)                                3_730.65     5_168.99     8_899.64       0.5743          1.0543            1.0502         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_911.62       413.79     2_325.41       0.3790          1.1177            1.1178         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_911.62       499.41     2_411.03       0.3790          1.1177            1.1178         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_911.62       765.64     2_677.26       0.3790          1.1177            1.1178         1.70
IVF-PQ-nl223-m16 (self)                                1_911.62     2_420.68     4_332.30       0.2686          1.2119            1.2160         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_305.45       591.81     2_897.26       0.4904          1.0726            1.0713         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_305.45       722.67     3_028.12       0.4904          1.0726            1.0713         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_305.45     1_054.71     3_360.17       0.4904          1.0726            1.0713         2.46
IVF-PQ-nl223-m32 (self)                                2_305.45     3_487.54     5_792.99       0.3846          1.1297            1.1284         2.46
IVF-PQ-nl223-m64-np11 (query)                          3_197.13     1_047.78     4_244.91       0.6327          1.0345            1.0330         3.99
IVF-PQ-nl223-m64-np14 (query)                          3_197.13     1_293.67     4_490.80       0.6327          1.0345            1.0330         3.99
IVF-PQ-nl223-m64-np21 (query)                          3_197.13     1_919.72     5_116.85       0.6327          1.0345            1.0330         3.99
IVF-PQ-nl223-m64 (self)                                3_197.13     6_267.33     9_464.46       0.5726          1.0548            1.0506         3.99
IVF-PQ-nl316-m16-np15 (query)                          2_415.63       547.75     2_963.38       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16-np17 (query)                          2_415.63       608.50     3_024.13       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16-np25 (query)                          2_415.63       841.82     3_257.45       0.3783          1.1178            1.1181         1.88
IVF-PQ-nl316-m16 (self)                                2_415.63     2_852.74     5_268.37       0.2624          1.2171            1.2220         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_796.91       756.14     3_553.06       0.4882          1.0729            1.0718         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_796.91       837.78     3_634.70       0.4882          1.0729            1.0718         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_796.91     1_198.48     3_995.39       0.4882          1.0729            1.0718         2.65
IVF-PQ-nl316-m32 (self)                                2_796.91     3_959.84     6_756.75       0.3734          1.1353            1.1345         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_578.48     1_349.47     4_927.95       0.6355          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_578.48     1_514.36     5_092.84       0.6355          1.0339            1.0324         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_578.48     2_174.46     5_752.94       0.6355          1.0339            1.0324         4.17
IVF-PQ-nl316-m64 (self)                                3_578.48     7_182.41    10_760.89       0.5697          1.0554            1.0517         4.17
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
Exhaustive (query)                                       100.45     1_806.43     1_906.87       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.45     5_928.12     6_028.56       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_172.53       727.89     1_900.42       0.2070          1.2190            1.2147         1.51
Exhaustive-PQ-m16 (self)                               1_172.53     2_276.32     3_448.85       0.1758          1.3086            1.3089         1.51
Exhaustive-PQ-m32 (query)                              1_627.56     1_568.56     3_196.12       0.2712          1.1686            1.1635         2.28
Exhaustive-PQ-m32 (self)                               1_627.56     5_098.86     6_726.41       0.2191          1.2527            1.2505         2.28
Exhaustive-PQ-m64 (query)                              2_670.08     3_661.03     6_331.11       0.3548          1.1211            1.1168         3.80
Exhaustive-PQ-m64 (self)                               2_670.08    12_047.47    14_717.55       0.2870          1.1905            1.1878         3.80
Exhaustive-PQ-m128 (query)                             4_883.54     8_046.49    12_930.04       0.4599          1.0780            1.0752         6.86
Exhaustive-PQ-m128 (self)                              4_883.54    26_528.11    31_411.66       0.3909          1.1257            1.1234         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_309.36       357.44     3_666.79       0.3631          1.1174            1.1165         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_309.36       543.14     3_852.50       0.3631          1.1174            1.1165         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_309.36       753.24     4_062.60       0.3631          1.1174            1.1165         1.98
IVF-PQ-nl158-m16 (self)                                3_309.36     2_537.18     5_846.53       0.2607          1.2171            1.2210         1.98
IVF-PQ-nl158-m32-np7 (query)                           3_873.09       526.45     4_399.53       0.4653          1.0758            1.0742         2.74
IVF-PQ-nl158-m32-np12 (query)                          3_873.09       883.38     4_756.46       0.4653          1.0758            1.0742         2.74
IVF-PQ-nl158-m32-np17 (query)                          3_873.09     1_136.84     5_009.93       0.4653          1.0758            1.0742         2.74
IVF-PQ-nl158-m32 (self)                                3_873.09     3_781.00     7_654.09       0.3690          1.1372            1.1366         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_639.08       815.72     5_454.81       0.5783          1.0438            1.0420         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_639.08     1_284.51     5_923.59       0.5783          1.0438            1.0420         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_639.08     1_780.96     6_420.04       0.5783          1.0438            1.0420         4.27
IVF-PQ-nl158-m64 (self)                                4_639.08     5_836.02    10_475.10       0.5192          1.0707            1.0668         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_374.06     1_554.03     7_928.09       0.7382          1.0160            1.0141         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_374.06     2_470.73     8_844.78       0.7382          1.0160            1.0141         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_374.06     3_383.57     9_757.63       0.7382          1.0160            1.0141         7.32
IVF-PQ-nl158-m128 (self)                               6_374.06    11_509.41    17_883.46       0.7140          1.0239            1.0193         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_823.92       509.98     3_333.90       0.3622          1.1176            1.1180         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_823.92       606.80     3_430.72       0.3622          1.1176            1.1180         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_823.92       876.68     3_700.59       0.3622          1.1176            1.1180         2.17
IVF-PQ-nl223-m16 (self)                                2_823.92     2_921.56     5_745.47       0.2537          1.2248            1.2300         2.17
IVF-PQ-nl223-m32-np11 (query)                          3_007.22       731.00     3_738.21       0.4606          1.0772            1.0760         2.93
IVF-PQ-nl223-m32-np14 (query)                          3_007.22       904.93     3_912.15       0.4606          1.0772            1.0760         2.93
IVF-PQ-nl223-m32-np21 (query)                          3_007.22     1_315.93     4_323.14       0.4606          1.0772            1.0760         2.93
IVF-PQ-nl223-m32 (self)                                3_007.22     4_329.37     7_336.58       0.3486          1.1494            1.1490         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_841.99     1_164.77     5_006.76       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_841.99     1_448.71     5_290.70       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_841.99     2_108.81     5_950.80       0.5749          1.0444            1.0429         4.46
IVF-PQ-nl223-m64 (self)                                3_841.99     6_973.52    10_815.52       0.5020          1.0767            1.0727         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_588.01     2_268.41     7_856.42       0.7403          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_588.01     2_834.50     8_422.51       0.7403          1.0155            1.0139         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_588.01     4_147.17     9_735.17       0.7403          1.0155            1.0139         7.51
IVF-PQ-nl223-m128 (self)                               5_588.01    13_787.53    19_375.54       0.7127          1.0239            1.0198         7.51
IVF-PQ-nl316-m16-np15 (query)                          3_143.08       663.56     3_806.64       0.3563          1.1200            1.1206         2.44
IVF-PQ-nl316-m16-np17 (query)                          3_143.08       730.40     3_873.48       0.3563          1.1200            1.1206         2.44
IVF-PQ-nl316-m16-np25 (query)                          3_143.08     1_045.34     4_188.42       0.3563          1.1200            1.1206         2.44
IVF-PQ-nl316-m16 (self)                                3_143.08     3_480.54     6_623.62       0.2439          1.2328            1.2393         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_594.33       986.66     4_580.99       0.4530          1.0793            1.0784         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_594.33     1_092.11     4_686.44       0.4530          1.0793            1.0784         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_594.33     1_566.42     5_160.76       0.4530          1.0793            1.0784         3.21
IVF-PQ-nl316-m32 (self)                                3_594.33     5_192.63     8_786.96       0.3299          1.1595            1.1607         3.21
IVF-PQ-nl316-m64-np15 (query)                          4_481.14     1_531.16     6_012.30       0.5729          1.0449            1.0431         4.73
IVF-PQ-nl316-m64-np17 (query)                          4_481.14     1_711.24     6_192.38       0.5729          1.0449            1.0431         4.73
IVF-PQ-nl316-m64-np25 (query)                          4_481.14     2_461.50     6_942.64       0.5729          1.0449            1.0431         4.73
IVF-PQ-nl316-m64 (self)                                4_481.14     8_757.94    13_239.08       0.4859          1.0822            1.0787         4.73
IVF-PQ-nl316-m128-np15 (query)                         6_280.12     2_968.84     9_248.95       0.7429          1.0152            1.0136         7.78
IVF-PQ-nl316-m128-np17 (query)                         6_280.12     3_332.20     9_612.32       0.7429          1.0152            1.0136         7.78
IVF-PQ-nl316-m128-np25 (query)                         6_280.12     4_790.98    11_071.09       0.7429          1.0152            1.0136         7.78
IVF-PQ-nl316-m128 (self)                               6_280.12    15_986.18    22_266.30       0.7103          1.0241            1.0204         7.78
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
Exhaustive (query)                                        32.75       703.40       736.15       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.75     2_307.55     2_340.30       1.0000          1.0000            1.0000        48.83
Exhaustive-PQ-m16 (query)                                662.88       669.50     1_332.38       0.7119          1.1576            1.1395         1.01
Exhaustive-PQ-m16 (self)                                 662.88     2_208.60     2_871.48       0.6210          1.2885            1.2506         1.01
Exhaustive-PQ-m32 (query)                              1_174.93     1_538.01     2_712.94       0.7717          1.0965            1.0836         1.78
Exhaustive-PQ-m32 (self)                               1_174.93     5_062.43     6_237.36       0.6993          1.1778            1.1516         1.78
Exhaustive-PQ-m64 (query)                              2_062.74     3_616.95     5_679.69       0.8251          1.0574            1.0468         3.30
Exhaustive-PQ-m64 (self)                               2_062.74    11_973.66    14_036.40       0.7675          1.1055            1.0855         3.30
IVF-PQ-nl158-m16-np7 (query)                           1_578.28       215.81     1_794.09       0.8272          1.0522            1.0448         1.17
IVF-PQ-nl158-m16-np12 (query)                          1_578.28       357.26     1_935.55       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16-np17 (query)                          1_578.28       484.58     2_062.87       0.8277          1.0518            1.0444         1.17
IVF-PQ-nl158-m16 (self)                                1_578.28     1_601.03     3_179.31       0.7669          1.0989            1.0836         1.17
IVF-PQ-nl158-m32-np7 (query)                           2_023.98       389.39     2_413.36       0.8746          1.0266            1.0219         1.93
IVF-PQ-nl158-m32-np12 (query)                          2_023.98       649.89     2_673.87       0.8751          1.0262            1.0217         1.93
IVF-PQ-nl158-m32-np17 (query)                          2_023.98       915.07     2_939.05       0.8751          1.0262            1.0217         1.93
IVF-PQ-nl158-m32 (self)                                2_023.98     3_000.72     5_024.69       0.8288          1.0511            1.0423         1.93
IVF-PQ-nl158-m64-np7 (query)                           2_685.88       718.08     3_403.96       0.9048          1.0151            1.0118         3.46
IVF-PQ-nl158-m64-np12 (query)                          2_685.88     1_206.40     3_892.28       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64-np17 (query)                          2_685.88     1_700.01     4_385.88       0.9056          1.0147            1.0116         3.46
IVF-PQ-nl158-m64 (self)                                2_685.88     6_188.57     8_874.45       0.8704          1.0288            1.0227         3.46
IVF-PQ-nl223-m16-np11 (query)                          1_468.66       338.12     1_806.79       0.8428          1.0430            1.0365         1.23
IVF-PQ-nl223-m16-np14 (query)                          1_468.66       412.08     1_880.74       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16-np21 (query)                          1_468.66       551.79     2_020.45       0.8429          1.0429            1.0365         1.23
IVF-PQ-nl223-m16 (self)                                1_468.66     1_861.03     3_329.69       0.7841          1.0842            1.0704         1.23
IVF-PQ-nl223-m32-np11 (query)                          1_586.38       542.29     2_128.67       0.8837          1.0224            1.0183         2.00
IVF-PQ-nl223-m32-np14 (query)                          1_586.38       677.54     2_263.92       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32-np21 (query)                          1_586.38     1_002.44     2_588.82       0.8838          1.0223            1.0183         2.00
IVF-PQ-nl223-m32 (self)                                1_586.38     3_332.54     4_918.92       0.8403          1.0440            1.0356         2.00
IVF-PQ-nl223-m64-np11 (query)                          2_222.49       956.34     3_178.83       0.9100          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np14 (query)                          2_222.49     1_222.61     3_445.10       0.9102          1.0134            1.0102         3.52
IVF-PQ-nl223-m64-np21 (query)                          2_222.49     1_808.68     4_031.17       0.9102          1.0133            1.0102         3.52
IVF-PQ-nl223-m64 (self)                                2_222.49     6_027.12     8_249.61       0.8765          1.0259            1.0200         3.52
IVF-PQ-nl316-m16-np15 (query)                          1_366.86       384.88     1_751.75       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np17 (query)                          1_366.86       433.99     1_800.86       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16-np25 (query)                          1_366.86       619.43     1_986.29       0.8502          1.0391            1.0334         1.32
IVF-PQ-nl316-m16 (self)                                1_366.86     2_068.18     3_435.04       0.7922          1.0785            1.0637         1.32
IVF-PQ-nl316-m32-np15 (query)                          1_808.95       684.51     2_493.46       0.8867          1.0214            1.0175         2.09
IVF-PQ-nl316-m32-np17 (query)                          1_808.95       770.47     2_579.41       0.8867          1.0214            1.0174         2.09
IVF-PQ-nl316-m32-np25 (query)                          1_808.95     1_123.47     2_932.42       0.8868          1.0214            1.0174         2.09
IVF-PQ-nl316-m32 (self)                                1_808.95     3_701.78     5_510.73       0.8425          1.0433            1.0344         2.09
IVF-PQ-nl316-m64-np15 (query)                          2_415.18     1_231.60     3_646.78       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np17 (query)                          2_415.18     1_398.07     3_813.25       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64-np25 (query)                          2_415.18     2_027.58     4_442.76       0.9127          1.0125            1.0095         3.61
IVF-PQ-nl316-m64 (self)                                2_415.18     6_694.77     9_109.95       0.8791          1.0247            1.0188         3.61
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
Exhaustive (query)                                        70.22     1_274.78     1_345.01       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.22     4_202.66     4_272.88       1.0000          1.0000            1.0000        97.66
Exhaustive-PQ-m16 (query)                                885.40       687.63     1_573.02       0.6791          1.1977            1.1746         1.26
Exhaustive-PQ-m16 (self)                                 885.40     2_233.24     3_118.64       0.5853          1.3494            1.3061         1.26
Exhaustive-PQ-m32 (query)                              1_279.01     1_546.04     2_825.05       0.7374          1.1283            1.1129         2.03
Exhaustive-PQ-m32 (self)                               1_279.01     5_148.09     6_427.10       0.6552          1.2348            1.2026         2.03
Exhaustive-PQ-m64 (query)                              2_176.50     3_654.18     5_830.68       0.7805          1.0879            1.0755         3.55
Exhaustive-PQ-m64 (self)                               2_176.50    12_089.14    14_265.65       0.7136          1.1583            1.1336         3.55
IVF-PQ-nl158-m16-np7 (query)                           2_599.93       281.91     2_881.84       0.8455          1.0448            1.0357         1.57
IVF-PQ-nl158-m16-np12 (query)                          2_599.93       451.75     3_051.68       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16-np17 (query)                          2_599.93       619.78     3_219.71       0.8458          1.0447            1.0356         1.57
IVF-PQ-nl158-m16 (self)                                2_599.93     2_054.80     4_654.73       0.7844          1.0913            1.0648         1.57
IVF-PQ-nl158-m32-np7 (query)                           2_993.97       429.19     3_423.15       0.8726          1.0297            1.0231         2.34
IVF-PQ-nl158-m32-np12 (query)                          2_993.97       712.29     3_706.26       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32-np17 (query)                          2_993.97       977.74     3_971.71       0.8731          1.0294            1.0230         2.34
IVF-PQ-nl158-m32 (self)                                2_993.97     3_221.64     6_215.61       0.8208          1.0615            1.0426         2.34
IVF-PQ-nl158-m64-np7 (query)                           3_900.38       807.01     4_707.38       0.8936          1.0202            1.0150         3.86
IVF-PQ-nl158-m64-np12 (query)                          3_900.38     1_334.23     5_234.60       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64-np17 (query)                          3_900.38     1_869.06     5_769.43       0.8941          1.0200            1.0149         3.86
IVF-PQ-nl158-m64 (self)                                3_900.38     6_215.11    10_115.49       0.8494          1.0420            1.0290         3.86
IVF-PQ-nl223-m16-np11 (query)                          1_697.83       422.49     2_120.31       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np14 (query)                          1_697.83       519.45     2_217.27       0.8543          1.0397            1.0313         1.70
IVF-PQ-nl223-m16-np21 (query)                          1_697.83       761.41     2_459.23       0.8544          1.0397            1.0313         1.70
IVF-PQ-nl223-m16 (self)                                1_697.83     2_532.00     4_229.83       0.7965          1.0807            1.0566         1.70
IVF-PQ-nl223-m32-np11 (query)                          2_154.99       612.36     2_767.34       0.8795          1.0266            1.0202         2.46
IVF-PQ-nl223-m32-np14 (query)                          2_154.99       755.52     2_910.51       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32-np21 (query)                          2_154.99     1_123.57     3_278.56       0.8795          1.0265            1.0202         2.46
IVF-PQ-nl223-m32 (self)                                2_154.99     3_674.82     5_829.80       0.8306          1.0550            1.0377         2.46
IVF-PQ-nl223-m64-np11 (query)                          2_995.82     1_107.05     4_102.87       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np14 (query)                          2_995.82     1_400.97     4_396.79       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64-np21 (query)                          2_995.82     2_059.87     5_055.68       0.9003          1.0178            1.0129         3.99
IVF-PQ-nl223-m64 (self)                                2_995.82     6_822.51     9_818.32       0.8568          1.0378            1.0256         3.99
IVF-PQ-nl316-m16-np15 (query)                          1_959.07       534.86     2_493.93       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np17 (query)                          1_959.07       592.86     2_551.93       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16-np25 (query)                          1_959.07       846.38     2_805.45       0.8694          1.0319            1.0253         1.88
IVF-PQ-nl316-m16 (self)                                1_959.07     2_836.43     4_795.50       0.8149          1.0655            1.0461         1.88
IVF-PQ-nl316-m32-np15 (query)                          2_354.36       781.30     3_135.66       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np17 (query)                          2_354.36       863.40     3_217.76       0.8917          1.0215            1.0159         2.65
IVF-PQ-nl316-m32-np25 (query)                          2_354.36     1_251.18     3_605.55       0.8917          1.0214            1.0159         2.65
IVF-PQ-nl316-m32 (self)                                2_354.36     4_098.06     6_452.42       0.8455          1.0452            1.0302         2.65
IVF-PQ-nl316-m64-np15 (query)                          3_258.84     1_402.04     4_660.88       0.9064          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np17 (query)                          3_258.84     1_580.47     4_839.31       0.9064          1.0155            1.0112         4.17
IVF-PQ-nl316-m64-np25 (query)                          3_258.84     2_291.56     5_550.40       0.9065          1.0155            1.0112         4.17
IVF-PQ-nl316-m64 (self)                                3_258.84     7_623.49    10_882.33       0.8655          1.0331            1.0218         4.17
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
Exhaustive (query)                                        99.30     1_775.43     1_874.73       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.30     5_865.19     5_964.48       1.0000          1.0000            1.0000       146.48
Exhaustive-PQ-m16 (query)                              1_151.86       705.77     1_857.64       0.6502          1.2419            1.2113         1.51
Exhaustive-PQ-m16 (self)                               1_151.86     2_239.64     3_391.50       0.5522          1.4109            1.3575         1.51
Exhaustive-PQ-m32 (query)                              1_584.80     1_558.95     3_143.75       0.7657          1.0989            1.0852         2.28
Exhaustive-PQ-m32 (self)                               1_584.80     5_013.44     6_598.24       0.6925          1.1782            1.1510         2.28
Exhaustive-PQ-m64 (query)                              2_471.84     3_558.96     6_030.80       0.8202          1.0558            1.0466         3.80
Exhaustive-PQ-m64 (self)                               2_471.84    11_809.29    14_281.13       0.7633          1.1010            1.0854         3.80
Exhaustive-PQ-m128 (query)                             4_248.09     7_801.28    12_049.37       0.8668          1.0289            1.0236         6.86
Exhaustive-PQ-m128 (self)                              4_248.09    25_987.76    30_235.85       0.8261          1.0515            1.0424         6.86
IVF-PQ-nl158-m16-np7 (query)                           3_712.27       384.30     4_096.56       0.8522          1.0420            1.0322         1.98
IVF-PQ-nl158-m16-np12 (query)                          3_712.27       601.63     4_313.90       0.8524          1.0420            1.0321         1.98
IVF-PQ-nl158-m16-np17 (query)                          3_712.27       809.53     4_521.80       0.8524          1.0420            1.0321         1.98
IVF-PQ-nl158-m16 (self)                                3_712.27     2_717.53     6_429.79       0.7910          1.0835            1.0598         1.98
IVF-PQ-nl158-m32-np7 (query)                           4_076.00       560.21     4_636.21       0.9000          1.0207            1.0126         2.74
IVF-PQ-nl158-m32-np12 (query)                          4_076.00       908.95     4_984.95       0.9001          1.0206            1.0126         2.74
IVF-PQ-nl158-m32-np17 (query)                          4_076.00     1_253.24     5_329.24       0.9001          1.0206            1.0126         2.74
IVF-PQ-nl158-m32 (self)                                4_076.00     4_178.04     8_254.04       0.8546          1.0436            1.0236         2.74
IVF-PQ-nl158-m64-np7 (query)                           4_987.50       908.55     5_896.05       0.9202          1.0131            1.0071         4.27
IVF-PQ-nl158-m64-np12 (query)                          4_987.50     1_490.09     6_477.59       0.9204          1.0130            1.0070         4.27
IVF-PQ-nl158-m64-np17 (query)                          4_987.50     2_074.86     7_062.37       0.9204          1.0130            1.0070         4.27
IVF-PQ-nl158-m64 (self)                                4_987.50     6_855.61    11_843.11       0.8831          1.0284            1.0135         4.27
IVF-PQ-nl158-m128-np7 (query)                          6_685.93     1_873.26     8_559.19       0.9393          1.0072            1.0031         7.32
IVF-PQ-nl158-m128-np12 (query)                         6_685.93     2_959.00     9_644.93       0.9395          1.0071            1.0031         7.32
IVF-PQ-nl158-m128-np17 (query)                         6_685.93     4_129.27    10_815.20       0.9395          1.0071            1.0031         7.32
IVF-PQ-nl158-m128 (self)                               6_685.93    13_705.11    20_391.04       0.9071          1.0171            1.0071         7.32
IVF-PQ-nl223-m16-np11 (query)                          2_250.32       517.05     2_767.36       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16-np14 (query)                          2_250.32       628.83     2_879.14       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16-np21 (query)                          2_250.32       919.24     3_169.55       0.8627          1.0359            1.0281         2.17
IVF-PQ-nl223-m16 (self)                                2_250.32     3_058.08     5_308.40       0.8061          1.0703            1.0513         2.17
IVF-PQ-nl223-m32-np11 (query)                          2_705.68       769.47     3_475.15       0.9087          1.0172            1.0103         2.93
IVF-PQ-nl223-m32-np14 (query)                          2_705.68       963.57     3_669.25       0.9088          1.0172            1.0103         2.93
IVF-PQ-nl223-m32-np21 (query)                          2_705.68     1_426.79     4_132.47       0.9088          1.0172            1.0103         2.93
IVF-PQ-nl223-m32 (self)                                2_705.68     4_704.05     7_409.73       0.8678          1.0351            1.0191         2.93
IVF-PQ-nl223-m64-np11 (query)                          3_627.99     1_238.38     4_866.37       0.9272          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np14 (query)                          3_627.99     1_554.99     5_182.97       0.9272          1.0111            1.0057         4.46
IVF-PQ-nl223-m64-np21 (query)                          3_627.99     2_315.43     5_943.42       0.9273          1.0111            1.0057         4.46
IVF-PQ-nl223-m64 (self)                                3_627.99     7_712.70    11_340.69       0.8925          1.0236            1.0111         4.46
IVF-PQ-nl223-m128-np11 (query)                         5_393.88     2_429.27     7_823.16       0.9439          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np14 (query)                         5_393.88     3_076.09     8_469.98       0.9440          1.0059            1.0023         7.51
IVF-PQ-nl223-m128-np21 (query)                         5_393.88     4_563.58     9_957.46       0.9440          1.0059            1.0023         7.51
IVF-PQ-nl223-m128 (self)                               5_393.88    15_209.51    20_603.39       0.9133          1.0148            1.0058         7.51
IVF-PQ-nl316-m16-np15 (query)                          2_599.32       671.40     3_270.72       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16-np17 (query)                          2_599.32       747.28     3_346.61       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16-np25 (query)                          2_599.32     1_060.71     3_660.04       0.8689          1.0325            1.0254         2.44
IVF-PQ-nl316-m16 (self)                                2_599.32     3_525.70     6_125.02       0.8141          1.0644            1.0466         2.44
IVF-PQ-nl316-m32-np15 (query)                          3_062.47       992.22     4_054.69       0.9129          1.0154            1.0094         3.21
IVF-PQ-nl316-m32-np17 (query)                          3_062.47     1_115.87     4_178.33       0.9129          1.0154            1.0094         3.21
IVF-PQ-nl316-m32-np25 (query)                          3_062.47     1_593.51     4_655.98       0.9129          1.0154            1.0093         3.21
IVF-PQ-nl316-m32 (self)                                3_062.47     5_302.93     8_365.40       0.8726          1.0331            1.0177         3.21
IVF-PQ-nl316-m64-np15 (query)                          3_996.21     1_587.71     5_583.93       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64-np17 (query)                          3_996.21     1_769.56     5_765.77       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64-np25 (query)                          3_996.21     2_581.35     6_577.56       0.9302          1.0099            1.0050         4.73
IVF-PQ-nl316-m64 (self)                                3_996.21     8_554.24    12_550.45       0.8964          1.0221            1.0099         4.73
IVF-PQ-nl316-m128-np15 (query)                         5_717.43     3_092.54     8_809.97       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128-np17 (query)                         5_717.43     3_482.16     9_199.59       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128-np25 (query)                         5_717.43     5_076.75    10_794.18       0.9458          1.0055            1.0020         7.78
IVF-PQ-nl316-m128 (self)                               5_717.43    16_870.94    22_588.37       0.9164          1.0139            1.0052         7.78
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
Exhaustive (query)                                        32.34       670.66       702.99       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.34     2_174.30     2_206.64       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_364.42       711.69     4_076.11       0.2866          1.1528            1.1332         1.26
Exhaustive-OPQ-m16 (self)                              3_364.42     2_693.86     6_058.28       0.2585          1.1711            1.1497         1.26
Exhaustive-OPQ-m32 (query)                             5_439.09     1_542.91     6_982.00       0.3260          1.1208            1.1170         2.03
Exhaustive-OPQ-m32 (self)                              5_439.09     5_435.12    10_874.21       0.2831          1.1440            1.1382         2.03
Exhaustive-OPQ-m64 (query)                             8_297.90     3_625.23    11_923.13       0.3796          1.0982            1.0951         3.55
Exhaustive-OPQ-m64 (self)                              8_297.90    12_396.11    20_694.01       0.3219          1.1205            1.1171         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_099.97       262.97     4_362.94       0.3869          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_099.97       371.93     4_471.90       0.3869          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_099.97       475.69     4_575.65       0.3869          1.0889            1.0912         1.67
IVF-OPQ-nl158-m16 (self)                               4_099.97     1_894.69     5_994.66       0.3184          1.1180            1.1227         1.67
IVF-OPQ-nl158-m32-np7 (query)                          5_967.81       423.22     6_391.03       0.4936          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32-np12 (query)                         5_967.81       621.09     6_588.90       0.4936          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32-np17 (query)                         5_967.81       819.71     6_787.52       0.4936          1.0558            1.0546         2.43
IVF-OPQ-nl158-m32 (self)                               5_967.81     3_009.98     8_977.78       0.4170          1.0758            1.0763         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_729.73       691.40     9_421.13       0.6944          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_729.73     1_041.53     9_771.26       0.6944          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_729.73     1_391.59    10_121.32       0.6944          1.0187            1.0163         3.96
IVF-OPQ-nl158-m64 (self)                               8_729.73     4_948.49    13_678.21       0.6381          1.0260            1.0239         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_865.10       350.50     4_215.60       0.3977          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_865.10       417.56     4_282.66       0.3977          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_865.10       583.17     4_448.27       0.3977          1.0838            1.0848         1.73
IVF-OPQ-nl223-m16 (self)                               3_865.10     2_271.27     6_136.38       0.3209          1.1161            1.1204         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_305.02       571.35     6_876.36       0.5054          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_305.02       681.80     6_986.82       0.5054          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_305.02       976.22     7_281.24       0.5054          1.0530            1.0504         2.50
IVF-OPQ-nl223-m32 (self)                               6_305.02     3_546.27     9_851.29       0.4236          1.0742            1.0736         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_672.09       975.61     9_647.70       0.7016          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_672.09     1_177.63     9_849.72       0.7016          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_672.09     1_682.85    10_354.94       0.7016          1.0183            1.0153         4.02
IVF-OPQ-nl223-m64 (self)                               8_672.09     5_886.22    14_558.31       0.6435          1.0256            1.0228         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_106.62       449.64     4_556.26       0.4070          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_106.62       489.74     4_596.36       0.4070          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_106.62       717.99     4_824.61       0.4070          1.0803            1.0817         2.07
IVF-OPQ-nl316-m16 (self)                               4_106.62     2_619.34     6_725.96       0.3255          1.1134            1.1174         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_145.90       717.57     6_863.47       0.5177          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_145.90       801.15     6_947.05       0.5177          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_145.90     1_134.56     7_280.47       0.5177          1.0486            1.0472         2.84
IVF-OPQ-nl316-m32 (self)                               6_145.90     4_081.80    10_227.70       0.4332          1.0702            1.0706         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_908.50     1_223.01    10_131.51       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_908.50     1_363.12    10_271.62       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_908.50     1_927.52    10_836.03       0.7119          1.0163            1.0142         4.36
IVF-OPQ-nl316-m64 (self)                               8_908.50     6_804.50    15_713.01       0.6529          1.0237            1.0215         4.36
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
Exhaustive (query)                                        67.68     1_270.75     1_338.43       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.68     4_279.80     4_347.48       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_725.17     1_016.48     6_741.65       0.2658          1.1129            1.1007         2.26
Exhaustive-OPQ-m16 (self)                              5_725.17     4_691.11    10_416.28       0.2457          1.1245            1.1094         2.26
Exhaustive-OPQ-m32 (query)                             7_447.84     1_835.11     9_282.95       0.2865          1.0971            1.0962         3.03
Exhaustive-OPQ-m32 (self)                              7_447.84     7_424.44    14_872.28       0.2607          1.1082            1.1059         3.03
Exhaustive-OPQ-m64 (query)                            11_679.81     3_922.36    15_602.18       0.3195          1.0822            1.0844         4.55
Exhaustive-OPQ-m64 (self)                             11_679.81    14_336.00    26_015.82       0.2789          1.0967            1.0989         4.55
Exhaustive-OPQ-m128 (query)                           17_261.43     8_111.62    25_373.05       0.3687          1.0680            1.0687         7.61
Exhaustive-OPQ-m128 (self)                            17_261.43    28_425.15    45_686.58       0.3153          1.0825            1.0831         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_100.90       598.77     7_699.67       0.3240          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_100.90       740.01     7_840.90       0.3240          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_100.90       894.41     7_995.31       0.3240          1.0784            1.0827         3.07
IVF-OPQ-nl158-m16 (self)                               7_100.90     4_415.29    11_516.18       0.2767          1.0976            1.1034         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_859.55       729.96     9_589.51       0.3688          1.0656            1.0670         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_859.55       947.81     9_807.37       0.3688          1.0656            1.0670         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_859.55     1_171.47    10_031.02       0.3688          1.0656            1.0670         3.84
IVF-OPQ-nl158-m32 (self)                               8_859.55     5_379.16    14_238.71       0.3027          1.0856            1.0894         3.84
IVF-OPQ-nl158-m64-np7 (query)                         12_967.73     1_051.21    14_018.95       0.4762          1.0414            1.0398         5.36
IVF-OPQ-nl158-m64-np12 (query)                        12_967.73     1_453.41    14_421.14       0.4762          1.0414            1.0398         5.36
IVF-OPQ-nl158-m64-np17 (query)                        12_967.73     1_858.65    14_826.39       0.4762          1.0414            1.0398         5.36
IVF-OPQ-nl158-m64 (self)                              12_967.73     7_580.94    20_548.68       0.4022          1.0545            1.0548         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_370.26     1_573.87    19_944.12       0.6814          1.0145            1.0117         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_370.26     2_259.53    20_629.79       0.6814          1.0145            1.0117         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_370.26     2_958.63    21_328.89       0.6814          1.0145            1.0117         8.42
IVF-OPQ-nl158-m128 (self)                             18_370.26    11_238.30    29_608.56       0.6257          1.0193            1.0170         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_657.89       729.70     7_387.59       0.3297          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_657.89       806.78     7_464.67       0.3298          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_657.89     1_009.40     7_667.29       0.3298          1.0761            1.0787         3.20
IVF-OPQ-nl223-m16 (self)                               6_657.89     4_771.82    11_429.71       0.2774          1.0971            1.1019         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_510.17       907.23     9_417.40       0.3782          1.0617            1.0630         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_510.17     1_027.87     9_538.04       0.3782          1.0617            1.0630         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_510.17     1_339.11     9_849.27       0.3782          1.0617            1.0630         3.96
IVF-OPQ-nl223-m32 (self)                               8_510.17     5_876.27    14_386.44       0.3048          1.0841            1.0875         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_944.44     1_383.43    14_327.87       0.4880          1.0384            1.0371         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_944.44     1_605.86    14_550.30       0.4880          1.0384            1.0371         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_944.44     2_185.90    15_130.34       0.4880          1.0384            1.0371         5.49
IVF-OPQ-nl223-m64 (self)                              12_944.44     8_642.71    21_587.15       0.4072          1.0532            1.0530         5.49
IVF-OPQ-nl223-m128-np11 (query)                       19_390.05     2_144.48    21_534.52       0.6882          1.0133            1.0113         8.54
IVF-OPQ-nl223-m128-np14 (query)                       19_390.05     2_535.44    21_925.48       0.6882          1.0133            1.0113         8.54
IVF-OPQ-nl223-m128-np21 (query)                       19_390.05     3_523.71    22_913.75       0.6882          1.0133            1.0113         8.54
IVF-OPQ-nl223-m128 (self)                             19_390.05    13_071.72    32_461.77       0.6307          1.0187            1.0164         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_357.67       850.18     8_207.85       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_357.67       885.21     8_242.88       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_357.67     1_124.43     8_482.10       0.3377          1.0724            1.0763         3.88
IVF-OPQ-nl316-m16 (self)                               7_357.67     5_125.56    12_483.23       0.2805          1.0944            1.1002         3.88
IVF-OPQ-nl316-m32-np15 (query)                         9_643.90     1_089.26    10_733.17       0.3859          1.0585            1.0600         4.65
IVF-OPQ-nl316-m32-np17 (query)                         9_643.90     1_164.96    10_808.87       0.3859          1.0585            1.0600         4.65
IVF-OPQ-nl316-m32-np25 (query)                         9_643.90     1_514.01    11_157.91       0.3859          1.0585            1.0600         4.65
IVF-OPQ-nl316-m32 (self)                               9_643.90     6_427.30    16_071.20       0.3067          1.0823            1.0860         4.65
IVF-OPQ-nl316-m64-np15 (query)                        14_010.15     1_674.96    15_685.11       0.4958          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64-np17 (query)                        14_010.15     1_838.89    15_849.04       0.4958          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64-np25 (query)                        14_010.15     2_508.62    16_518.77       0.4958          1.0365            1.0355         6.17
IVF-OPQ-nl316-m64 (self)                              14_010.15    10_415.56    24_425.70       0.4128          1.0513            1.0520         6.17
IVF-OPQ-nl316-m128-np15 (query)                       20_644.51     2_669.14    23_313.65       0.6969          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np17 (query)                       20_644.51     2_896.61    23_541.11       0.6969          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128-np25 (query)                       20_644.51     3_992.02    24_636.52       0.6969          1.0125            1.0106         9.23
IVF-OPQ-nl316-m128 (self)                             20_644.51    14_760.14    35_404.65       0.6381          1.0176            1.0158         9.23
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
Exhaustive (query)                                       100.78     2_083.38     2_184.17       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.78     6_210.30     6_311.08       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_421.25     1_520.47    10_941.73       0.2602          1.0925            1.0831         3.76
Exhaustive-OPQ-m16 (self)                              9_421.25     8_169.75    17_591.00       0.2417          1.1021            1.0891         3.76
Exhaustive-OPQ-m32 (query)                            11_433.40     2_333.35    13_766.76       0.2792          1.0786            1.0796         4.53
Exhaustive-OPQ-m32 (self)                             11_433.40    10_938.56    22_371.97       0.2567          1.0877            1.0869         4.53
Exhaustive-OPQ-m64 (query)                            15_846.85     4_414.51    20_261.36       0.2961          1.0724            1.0758         6.05
Exhaustive-OPQ-m64 (self)                             15_846.85    17_763.86    33_610.71       0.2655          1.0825            1.0855         6.05
Exhaustive-OPQ-m128 (query)                           24_361.44     8_633.62    32_995.06       0.3310          1.0632            1.0657         9.11
Exhaustive-OPQ-m128 (self)                            24_361.44    31_995.99    56_357.43       0.2833          1.0754            1.0784         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_062.00     1_219.38    12_281.38       0.3038          1.0684            1.0735         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_062.00     1_327.71    12_389.71       0.3038          1.0684            1.0735         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_062.00     1_524.76    12_586.76       0.3038          1.0684            1.0735         4.98
IVF-OPQ-nl158-m16 (self)                              11_062.00     8_189.23    19_251.23       0.2669          1.0820            1.0877         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_225.32     1_316.37    14_541.70       0.3313          1.0606            1.0637         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_225.32     1_612.44    14_837.77       0.3313          1.0606            1.0637         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_225.32     1_901.39    15_126.71       0.3313          1.0606            1.0637         5.74
IVF-OPQ-nl158-m32 (self)                              13_225.32     9_485.35    22_710.67       0.2761          1.0776            1.0826         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_488.85     1_623.17    19_112.02       0.3896          1.0477            1.0481         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_488.85     2_083.16    19_572.01       0.3896          1.0477            1.0481         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_488.85     2_565.39    20_054.24       0.3896          1.0477            1.0481         7.27
IVF-OPQ-nl158-m64 (self)                              17_488.85    11_695.26    29_184.11       0.3202          1.0625            1.0646         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_836.11     2_453.76    28_289.87       0.5418          1.0242            1.0223        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_836.11     3_425.40    29_261.51       0.5418          1.0242            1.0223        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_836.11     4_393.96    30_230.07       0.5418          1.0242            1.0223        10.32
IVF-OPQ-nl158-m128 (self)                             25_836.11    17_772.66    43_608.77       0.4723          1.0318            1.0309        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_597.89     1_293.14    11_891.02       0.3071          1.0668            1.0709         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_597.89     1_400.25    11_998.13       0.3071          1.0668            1.0709         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_597.89     1_660.73    12_258.62       0.3071          1.0668            1.0709         5.17
IVF-OPQ-nl223-m16 (self)                              10_597.89     8_677.92    19_275.80       0.2669          1.0819            1.0876         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_784.00     1_557.75    14_341.75       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_784.00     1_731.78    14_515.77       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_784.00     2_141.39    14_925.39       0.3392          1.0579            1.0604         5.93
IVF-OPQ-nl223-m32 (self)                              12_784.00    10_335.03    23_119.02       0.2769          1.0772            1.0816         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_286.34     1_998.18    19_284.53       0.4027          1.0445            1.0445         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_286.34     2_282.34    19_568.69       0.4027          1.0445            1.0445         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_286.34     2_946.38    20_232.72       0.4027          1.0445            1.0445         7.46
IVF-OPQ-nl223-m64 (self)                              17_286.34    12_892.88    30_179.23       0.3228          1.0614            1.0633         7.46
IVF-OPQ-nl223-m128-np11 (query)                       26_735.74     3_213.51    29_949.25       0.5532          1.0227            1.0207        10.51
IVF-OPQ-nl223-m128-np14 (query)                       26_735.74     3_805.56    30_541.31       0.5532          1.0227            1.0207        10.51
IVF-OPQ-nl223-m128-np21 (query)                       26_735.74     5_155.48    31_891.22       0.5532          1.0227            1.0207        10.51
IVF-OPQ-nl223-m128 (self)                             26_735.74    20_392.70    47_128.45       0.4787          1.0309            1.0300        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_185.55     1_443.70    12_629.25       0.3135          1.0641            1.0690         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_185.55     1_514.65    12_700.20       0.3135          1.0641            1.0690         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_185.55     1_811.95    12_997.50       0.3135          1.0641            1.0690         6.19
IVF-OPQ-nl316-m16 (self)                              11_185.55     9_190.75    20_376.30       0.2693          1.0801            1.0861         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_329.99     1_806.11    15_136.10       0.3444          1.0558            1.0587         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_329.99     1_899.82    15_229.81       0.3444          1.0558            1.0587         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_329.99     2_365.74    15_695.73       0.3444          1.0558            1.0587         6.96
IVF-OPQ-nl316-m32 (self)                              13_329.99    11_051.84    24_381.83       0.2782          1.0755            1.0804         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_546.70     2_352.21    19_898.92       0.4109          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_546.70     2_539.59    20_086.30       0.4109          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_546.70     3_268.69    20_815.39       0.4109          1.0423            1.0425         8.48
IVF-OPQ-nl316-m64 (self)                              17_546.70    14_102.55    31_649.25       0.3273          1.0596            1.0618         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_134.91     3_966.10    30_101.01       0.5612          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_134.91     4_390.16    30_525.07       0.5612          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_134.91     5_868.86    32_003.78       0.5612          1.0214            1.0200        11.54
IVF-OPQ-nl316-m128 (self)                             26_134.91    22_666.10    48_801.01       0.4851          1.0296            1.0291        11.54
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
Exhaustive (query)                                        32.48       697.09       729.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.48     2_347.23     2_379.72       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_378.55       720.50     4_099.06       0.3008          1.2503            1.2433         1.26
Exhaustive-OPQ-m16 (self)                              3_378.55     2_672.84     6_051.40       0.2368          1.3778            1.3714         1.26
Exhaustive-OPQ-m32 (query)                             5_753.99     1_566.88     7_320.87       0.4204          1.1526            1.1474         2.03
Exhaustive-OPQ-m32 (self)                              5_753.99     5_434.34    11_188.33       0.3378          1.2478            1.2416         2.03
Exhaustive-OPQ-m64 (query)                             9_278.82     3_689.93    12_968.75       0.5661          1.0765            1.0733         3.55
Exhaustive-OPQ-m64 (self)                              9_278.82    12_468.70    21_747.53       0.4876          1.1287            1.1241         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_307.35       265.64     4_572.99       0.7005          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_307.35       375.51     4_682.86       0.7005          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_307.35       500.52     4_807.88       0.7005          1.0326            1.0310         1.67
IVF-OPQ-nl158-m16 (self)                               4_307.35     2_002.05     6_309.41       0.6205          1.0625            1.0598         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_396.97       420.29     6_817.26       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_396.97       626.31     7_023.28       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_396.97       846.97     7_243.94       0.7984          1.0139            1.0127         2.43
IVF-OPQ-nl158-m32 (self)                               6_396.97     3_165.92     9_562.89       0.7478          1.0257            1.0237         2.43
IVF-OPQ-nl158-m64-np7 (query)                          9_305.32       685.96     9_991.28       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np12 (query)                         9_305.32     1_040.73    10_346.05       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64-np17 (query)                         9_305.32     1_428.40    10_733.72       0.8603          1.0065            1.0055         3.96
IVF-OPQ-nl158-m64 (self)                               9_305.32     5_147.87    14_453.19       0.8308          1.0112            1.0097         3.96
IVF-OPQ-nl223-m16-np11 (query)                         4_083.32       357.73     4_441.05       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16-np14 (query)                         4_083.32       428.72     4_512.04       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16-np21 (query)                         4_083.32       600.08     4_683.40       0.7066          1.0310            1.0294         1.73
IVF-OPQ-nl223-m16 (self)                               4_083.32     2_341.09     6_424.41       0.6283          1.0597            1.0570         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_237.19       578.08     6_815.27       0.8037          1.0132            1.0120         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_237.19       715.15     6_952.34       0.8037          1.0132            1.0119         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_237.19     1_007.08     7_244.27       0.8037          1.0132            1.0119         2.50
IVF-OPQ-nl223-m32 (self)                               6_237.19     3_689.00     9_926.20       0.7548          1.0243            1.0224         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_719.00       952.21     9_671.21       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_719.00     1_169.58     9_888.58       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_719.00     1_706.74    10_425.74       0.8637          1.0061            1.0052         4.02
IVF-OPQ-nl223-m64 (self)                               8_719.00     5_951.82    14_670.82       0.8350          1.0106            1.0092         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_112.04       436.15     4_548.19       0.7130          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_112.04       481.18     4_593.21       0.7131          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_112.04       664.37     4_776.40       0.7131          1.0296            1.0280         2.07
IVF-OPQ-nl316-m16 (self)                               4_112.04     2_564.32     6_676.36       0.6356          1.0570            1.0544         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_237.17       736.95     6_974.12       0.8078          1.0127            1.0114         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_237.17       817.01     7_054.17       0.8079          1.0126            1.0114         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_237.17     1_151.61     7_388.77       0.8079          1.0126            1.0114         2.84
IVF-OPQ-nl316-m32 (self)                               6_237.17     4_160.97    10_398.14       0.7589          1.0235            1.0217         2.84
IVF-OPQ-nl316-m64-np15 (query)                         8_967.38     1_211.28    10_178.66       0.8656          1.0060            1.0050         4.36
IVF-OPQ-nl316-m64-np17 (query)                         8_967.38     1_348.25    10_315.63       0.8657          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64-np25 (query)                         8_967.38     1_931.27    10_898.66       0.8657          1.0059            1.0050         4.36
IVF-OPQ-nl316-m64 (self)                               8_967.38     6_741.27    15_708.65       0.8369          1.0103            1.0090         4.36
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
Exhaustive (query)                                        70.98     1_268.51     1_339.49       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.98     4_206.63     4_277.62       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             6_125.49     1_040.84     7_166.33       0.2317          1.2142            1.2107         2.26
Exhaustive-OPQ-m16 (self)                              6_125.49     4_734.34    10_859.83       0.1879          1.2983            1.2985         2.26
Exhaustive-OPQ-m32 (query)                             8_209.35     1_840.22    10_049.57       0.3189          1.1505            1.1472         3.03
Exhaustive-OPQ-m32 (self)                              8_209.35     7_438.83    15_648.18       0.2588          1.2171            1.2145         3.03
Exhaustive-OPQ-m64 (query)                            12_745.33     3_969.44    16_714.78       0.4332          1.0939            1.0912         4.55
Exhaustive-OPQ-m64 (self)                             12_745.33    14_609.86    27_355.19       0.3621          1.1417            1.1393         4.55
Exhaustive-OPQ-m128 (query)                           17_454.34     8_124.08    25_578.42       0.5699          1.0489            1.0475         7.61
Exhaustive-OPQ-m128 (self)                            17_454.34    28_422.89    45_877.24       0.4999          1.0773            1.0755         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_000.50       596.72     7_597.21       0.5391          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_000.50       743.15     7_743.64       0.5391          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_000.50       900.42     7_900.92       0.5391          1.0567            1.0554         3.07
IVF-OPQ-nl158-m16 (self)                               7_000.50     4_410.82    11_411.32       0.4372          1.1019            1.0999         3.07
IVF-OPQ-nl158-m32-np7 (query)                          8_784.25       720.03     9_504.28       0.6838          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32-np12 (query)                         8_784.25       936.09     9_720.34       0.6838          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32-np17 (query)                         8_784.25     1_161.80     9_946.05       0.6838          1.0246            1.0234         3.84
IVF-OPQ-nl158-m32 (self)                               8_784.25     5_344.00    14_128.25       0.6092          1.0440            1.0417         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_073.37     1_037.10    14_110.48       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_073.37     1_459.76    14_533.13       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_073.37     1_859.77    14_933.14       0.7778          1.0117            1.0106         5.36
IVF-OPQ-nl158-m64 (self)                              13_073.37     7_543.95    20_617.32       0.7294          1.0202            1.0180         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_204.80     1_549.60    19_754.40       0.8371          1.0062            1.0052         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_204.80     2_253.39    20_458.19       0.8371          1.0062            1.0052         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_204.80     2_925.08    21_129.88       0.8371          1.0062            1.0052         8.42
IVF-OPQ-nl158-m128 (self)                             18_204.80    11_148.39    29_353.19       0.8086          1.0101            1.0081         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_620.51       707.89     7_328.40       0.5464          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_620.51       795.76     7_416.27       0.5464          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_620.51       997.18     7_617.69       0.5464          1.0541            1.0529         3.20
IVF-OPQ-nl223-m16 (self)                               6_620.51     4_703.04    11_323.55       0.4480          1.0973            1.0954         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_592.98       902.59     9_495.57       0.6912          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_592.98     1_027.33     9_620.30       0.6912          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_592.98     1_347.62     9_940.60       0.6912          1.0233            1.0222         3.96
IVF-OPQ-nl223-m32 (self)                               8_592.98     5_841.84    14_434.82       0.6190          1.0414            1.0395         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_640.87     1_372.67    14_013.54       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_640.87     1_615.22    14_256.09       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_640.87     2_198.25    14_839.12       0.7814          1.0111            1.0102         5.49
IVF-OPQ-nl223-m64 (self)                              12_640.87     8_607.61    21_248.48       0.7360          1.0190            1.0173         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_180.89     2_086.92    20_267.81       0.8409          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_180.89     2_501.38    20_682.27       0.8409          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_180.89     3_520.75    21_701.64       0.8409          1.0059            1.0049         8.54
IVF-OPQ-nl223-m128 (self)                             18_180.89    12_989.30    31_170.19       0.8124          1.0095            1.0079         8.54
IVF-OPQ-nl316-m16-np15 (query)                         7_068.23       830.94     7_899.17       0.5528          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16-np17 (query)                         7_068.23       881.01     7_949.23       0.5528          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16-np25 (query)                         7_068.23     1_130.11     8_198.34       0.5528          1.0528            1.0516         3.88
IVF-OPQ-nl316-m16 (self)                               7_068.23     5_099.51    12_167.74       0.4526          1.0952            1.0936         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_957.99     1_073.77    10_031.76       0.6934          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_957.99     1_170.81    10_128.80       0.6934          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_957.99     1_511.52    10_469.51       0.6934          1.0229            1.0216         4.65
IVF-OPQ-nl316-m32 (self)                               8_957.99     6_392.55    15_350.54       0.6229          1.0404            1.0387         4.65
IVF-OPQ-nl316-m64-np15 (query)                        13_082.02     1_677.66    14_759.68       0.7856          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np17 (query)                        13_082.02     1_841.57    14_923.59       0.7856          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64-np25 (query)                        13_082.02     2_500.57    15_582.60       0.7856          1.0108            1.0098         6.17
IVF-OPQ-nl316-m64 (self)                              13_082.02     9_887.80    22_969.82       0.7389          1.0184            1.0168         6.17
IVF-OPQ-nl316-m128-np15 (query)                       19_286.53     2_587.03    21_873.56       0.8422          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np17 (query)                       19_286.53     2_857.42    22_143.95       0.8422          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128-np25 (query)                       19_286.53     3_961.76    23_248.29       0.8422          1.0058            1.0049         9.23
IVF-OPQ-nl316-m128 (self)                             19_286.53    14_568.46    33_854.99       0.8144          1.0092            1.0078         9.23
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
Exhaustive (query)                                        99.33     1_780.64     1_879.97       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.33     6_167.86     6_267.20       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_303.75     1_511.23    10_814.98       0.2295          1.2023            1.1985         3.76
Exhaustive-OPQ-m16 (self)                              9_303.75     8_162.51    17_466.27       0.1867          1.2974            1.2971         3.76
Exhaustive-OPQ-m32 (query)                            11_504.93     2_348.55    13_853.48       0.3122          1.1452            1.1413         4.53
Exhaustive-OPQ-m32 (self)                             11_504.93    10_958.05    22_462.98       0.2574          1.2173            1.2150         4.53
Exhaustive-OPQ-m64 (query)                            15_825.46     4_423.60    20_249.06       0.4084          1.0973            1.0946         6.05
Exhaustive-OPQ-m64 (self)                             15_825.46    17_804.14    33_629.60       0.3478          1.1497            1.1473         6.05
Exhaustive-OPQ-m128 (query)                           24_379.76     8_727.86    33_107.61       0.5284          1.0567            1.0548         9.11
Exhaustive-OPQ-m128 (self)                            24_379.76    32_070.54    56_450.30       0.4662          1.0907            1.0887         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_028.63     1_145.41    12_174.04       0.5319          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_028.63     1_401.25    12_429.88       0.5319          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_028.63     1_503.48    12_532.12       0.5319          1.0544            1.0533         4.98
IVF-OPQ-nl158-m16 (self)                              11_028.63     8_229.53    19_258.16       0.4300          1.1043            1.1024         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_208.31     1_318.00    14_526.31       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_208.31     1_605.73    14_814.05       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_208.31     1_920.07    15_128.38       0.6791          1.0237            1.0224         5.74
IVF-OPQ-nl158-m32 (self)                              13_208.31     9_505.55    22_713.86       0.6037          1.0450            1.0428         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_398.82     1_609.06    19_007.88       0.7740          1.0113            1.0101         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_398.82     2_100.84    19_499.66       0.7740          1.0113            1.0101         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_398.82     2_547.97    19_946.79       0.7740          1.0113            1.0101         7.27
IVF-OPQ-nl158-m64 (self)                              17_398.82    11_641.95    29_040.77       0.7252          1.0207            1.0184         7.27
IVF-OPQ-nl158-m128-np7 (query)                        25_992.37     2_439.67    28_432.04       0.8324          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128-np12 (query)                       25_992.37     3_417.72    29_410.09       0.8324          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128-np17 (query)                       25_992.37     4_367.78    30_360.15       0.8324          1.0062            1.0051        10.32
IVF-OPQ-nl158-m128 (self)                             25_992.37    17_721.26    43_713.62       0.8041          1.0105            1.0084        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_850.44     1_297.06    12_147.50       0.5397          1.0523            1.0511         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_850.44     1_397.74    12_248.18       0.5397          1.0523            1.0511         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_850.44     1_656.79    12_507.23       0.5397          1.0523            1.0511         5.17
IVF-OPQ-nl223-m16 (self)                              10_850.44     8_687.64    19_538.08       0.4397          1.1000            1.0983         5.17
IVF-OPQ-nl223-m32-np11 (query)                        13_057.31     1_551.07    14_608.37       0.6832          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32-np14 (query)                        13_057.31     1_730.10    14_787.41       0.6832          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32-np21 (query)                        13_057.31     2_165.54    15_222.85       0.6832          1.0229            1.0218         5.93
IVF-OPQ-nl223-m32 (self)                              13_057.31    10_346.60    23_403.91       0.6095          1.0435            1.0415         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_268.08     1_976.93    19_245.01       0.7788          1.0107            1.0097         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_268.08     2_253.54    19_521.62       0.7788          1.0107            1.0097         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_268.08     2_916.84    20_184.92       0.7788          1.0107            1.0097         7.46
IVF-OPQ-nl223-m64 (self)                              17_268.08    13_981.25    31_249.33       0.7309          1.0196            1.0178         7.46
IVF-OPQ-nl223-m128-np11 (query)                       25_917.12     3_178.22    29_095.34       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np14 (query)                       25_917.12     3_762.67    29_679.78       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128-np21 (query)                       25_917.12     5_151.06    31_068.17       0.8361          1.0059            1.0050        10.51
IVF-OPQ-nl223-m128 (self)                             25_917.12    20_323.31    46_240.43       0.8076          1.0100            1.0082        10.51
IVF-OPQ-nl316-m16-np15 (query)                        11_385.78     1_478.51    12_864.29       0.5457          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16-np17 (query)                        11_385.78     1_526.61    12_912.39       0.5457          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16-np25 (query)                        11_385.78     1_805.73    13_191.51       0.5457          1.0506            1.0494         6.19
IVF-OPQ-nl316-m16 (self)                              11_385.78     9_291.72    20_677.50       0.4461          1.0971            1.0955         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_640.49     1_776.79    15_417.28       0.6864          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_640.49     1_892.96    15_533.45       0.6864          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_640.49     2_354.09    15_994.58       0.6864          1.0225            1.0214         6.96
IVF-OPQ-nl316-m32 (self)                              13_640.49    10_975.08    24_615.57       0.6125          1.0426            1.0408         6.96
IVF-OPQ-nl316-m64-np15 (query)                        18_365.46     2_327.77    20_693.24       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np17 (query)                        18_365.46     2_512.16    20_877.62       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64-np25 (query)                        18_365.46     3_240.93    21_606.39       0.7795          1.0107            1.0097         8.48
IVF-OPQ-nl316-m64 (self)                              18_365.46    14_027.90    32_393.36       0.7332          1.0192            1.0176         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_551.39     3_910.85    30_462.24       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_551.39     4_279.18    30_830.57       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_551.39     5_824.62    32_376.00       0.8367          1.0058            1.0049        11.54
IVF-OPQ-nl316-m128 (self)                             26_551.39    22_567.38    49_118.77       0.8088          1.0097            1.0082        11.54
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
Exhaustive (query)                                        32.45       707.78       740.23       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.45     2_337.07     2_369.52       1.0000          1.0000            1.0000        48.83
Exhaustive-OPQ-m16 (query)                             3_346.15       714.41     4_060.55       0.7911          1.0819            1.0684         1.26
Exhaustive-OPQ-m16 (self)                              3_346.15     2_676.55     6_022.70       0.7231          1.1502            1.1255         1.26
Exhaustive-OPQ-m32 (query)                             5_420.76     1_546.89     6_967.65       0.8303          1.0536            1.0424         2.03
Exhaustive-OPQ-m32 (self)                              5_420.76     5_460.29    10_881.04       0.7763          1.0975            1.0767         2.03
Exhaustive-OPQ-m64 (query)                             8_159.13     3_670.53    11_829.67       0.8562          1.0398            1.0292         3.55
Exhaustive-OPQ-m64 (self)                              8_159.13    12_413.26    20_572.39       0.8092          1.0723            1.0534         3.55
IVF-OPQ-nl158-m16-np7 (query)                          4_184.44       279.10     4_463.54       0.8895          1.0208            1.0163         1.67
IVF-OPQ-nl158-m16-np12 (query)                         4_184.44       412.08     4_596.52       0.8903          1.0203            1.0161         1.67
IVF-OPQ-nl158-m16-np17 (query)                         4_184.44       549.94     4_734.39       0.8904          1.0203            1.0161         1.67
IVF-OPQ-nl158-m16 (self)                               4_184.44     2_173.21     6_357.65       0.8475          1.0405            1.0325         1.67
IVF-OPQ-nl158-m32-np7 (query)                          6_156.00       463.59     6_619.59       0.9110          1.0133            1.0098         2.43
IVF-OPQ-nl158-m32-np12 (query)                         6_156.00       718.70     6_874.70       0.9118          1.0129            1.0096         2.43
IVF-OPQ-nl158-m32-np17 (query)                         6_156.00       980.05     7_136.05       0.9118          1.0128            1.0096         2.43
IVF-OPQ-nl158-m32 (self)                               6_156.00     3_613.29     9_769.29       0.8774          1.0255            1.0196         2.43
IVF-OPQ-nl158-m64-np7 (query)                          8_821.22       778.57     9_599.80       0.9240          1.0098            1.0066         3.96
IVF-OPQ-nl158-m64-np12 (query)                         8_821.22     1_270.60    10_091.82       0.9248          1.0094            1.0064         3.96
IVF-OPQ-nl158-m64-np17 (query)                         8_821.22     1_767.71    10_588.93       0.9248          1.0094            1.0064         3.96
IVF-OPQ-nl158-m64 (self)                               8_821.22     6_060.16    14_881.38       0.8961          1.0185            1.0133         3.96
IVF-OPQ-nl223-m16-np11 (query)                         3_795.49       372.49     4_167.98       0.8979          1.0178            1.0138         1.73
IVF-OPQ-nl223-m16-np14 (query)                         3_795.49       443.09     4_238.58       0.8980          1.0177            1.0138         1.73
IVF-OPQ-nl223-m16-np21 (query)                         3_795.49       684.85     4_480.35       0.8980          1.0177            1.0138         1.73
IVF-OPQ-nl223-m16 (self)                               3_795.49     2_772.94     6_568.44       0.8581          1.0352            1.0271         1.73
IVF-OPQ-nl223-m32-np11 (query)                         6_428.35       609.00     7_037.35       0.9152          1.0118            1.0088         2.50
IVF-OPQ-nl223-m32-np14 (query)                         6_428.35       745.13     7_173.48       0.9153          1.0117            1.0087         2.50
IVF-OPQ-nl223-m32-np21 (query)                         6_428.35     1_074.18     7_502.53       0.9153          1.0117            1.0087         2.50
IVF-OPQ-nl223-m32 (self)                               6_428.35     3_928.10    10_356.45       0.8827          1.0234            1.0176         2.50
IVF-OPQ-nl223-m64-np11 (query)                         8_640.38     1_030.18     9_670.56       0.9269          1.0092            1.0060         4.02
IVF-OPQ-nl223-m64-np14 (query)                         8_640.38     1_281.02     9_921.41       0.9271          1.0091            1.0060         4.02
IVF-OPQ-nl223-m64-np21 (query)                         8_640.38     1_883.29    10_523.67       0.9271          1.0091            1.0060         4.02
IVF-OPQ-nl223-m64 (self)                               8_640.38     6_610.77    15_251.15       0.8987          1.0178            1.0123         4.02
IVF-OPQ-nl316-m16-np15 (query)                         4_196.57       477.00     4_673.58       0.9021          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16-np17 (query)                         4_196.57       522.32     4_718.89       0.9022          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16-np25 (query)                         4_196.57       757.26     4_953.83       0.9022          1.0164            1.0125         2.07
IVF-OPQ-nl316-m16 (self)                               4_196.57     2_771.58     6_968.16       0.8629          1.0334            1.0247         2.07
IVF-OPQ-nl316-m32-np15 (query)                         6_115.09       764.38     6_879.47       0.9172          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32-np17 (query)                         6_115.09       847.64     6_962.73       0.9173          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32-np25 (query)                         6_115.09     1_204.89     7_319.97       0.9174          1.0112            1.0081         2.84
IVF-OPQ-nl316-m32 (self)                               6_115.09     4_365.58    10_480.66       0.8839          1.0230            1.0170         2.84
IVF-OPQ-nl316-m64-np15 (query)                         9_074.90     1_278.62    10_353.52       0.9281          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64-np17 (query)                         9_074.90     1_436.82    10_511.72       0.9281          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64-np25 (query)                         9_074.90     2_075.91    11_150.80       0.9282          1.0087            1.0057         4.36
IVF-OPQ-nl316-m64 (self)                               9_074.90     7_112.35    16_187.25       0.9005          1.0171            1.0118         4.36
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
Exhaustive (query)                                        68.54     1_285.39     1_353.92       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.54     4_209.07     4_277.61       1.0000          1.0000            1.0000        97.66
Exhaustive-OPQ-m16 (query)                             5_617.77     1_015.42     6_633.19       0.7546          1.1136            1.0983         2.26
Exhaustive-OPQ-m16 (self)                              5_617.77     4_722.27    10_340.04       0.6788          1.2037            1.1739         2.26
Exhaustive-OPQ-m32 (query)                             7_472.94     1_843.01     9_315.95       0.8064          1.0692            1.0572         3.03
Exhaustive-OPQ-m32 (self)                              7_472.94     7_399.29    14_872.23       0.7455          1.1245            1.1019         3.03
Exhaustive-OPQ-m64 (query)                            11_643.01     3_904.11    15_547.13       0.8413          1.0455            1.0364         4.55
Exhaustive-OPQ-m64 (self)                             11_643.01    14_284.33    25_927.34       0.7916          1.0819            1.0654         4.55
Exhaustive-OPQ-m128 (query)                           17_254.07     8_099.60    25_353.68       0.9198          1.0107            1.0069         7.61
Exhaustive-OPQ-m128 (self)                            17_254.07    28_350.03    45_604.10       0.8933          1.0192            1.0139         7.61
IVF-OPQ-nl158-m16-np7 (query)                          7_177.57       621.33     7_798.89       0.8909          1.0219            1.0164         3.07
IVF-OPQ-nl158-m16-np12 (query)                         7_177.57       769.45     7_947.02       0.8913          1.0217            1.0163         3.07
IVF-OPQ-nl158-m16-np17 (query)                         7_177.57       936.50     8_114.06       0.8913          1.0217            1.0163         3.07
IVF-OPQ-nl158-m16 (self)                               7_177.57     4_557.49    11_735.05       0.8464          1.0446            1.0310         3.07
IVF-OPQ-nl158-m32-np7 (query)                          9_177.81       757.04     9_934.84       0.9020          1.0171            1.0125         3.84
IVF-OPQ-nl158-m32-np12 (query)                         9_177.81     1_024.15    10_201.96       0.9026          1.0169            1.0124         3.84
IVF-OPQ-nl158-m32-np17 (query)                         9_177.81     1_303.11    10_480.92       0.9026          1.0169            1.0124         3.84
IVF-OPQ-nl158-m32 (self)                               9_177.81     5_710.00    14_887.80       0.8629          1.0344            1.0239         3.84
IVF-OPQ-nl158-m64-np7 (query)                         13_071.60     1_122.72    14_194.32       0.9096          1.0145            1.0102         5.36
IVF-OPQ-nl158-m64-np12 (query)                        13_071.60     1_648.79    14_720.39       0.9102          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64-np17 (query)                        13_071.60     2_177.87    15_249.46       0.9102          1.0143            1.0100         5.36
IVF-OPQ-nl158-m64 (self)                              13_071.60     8_528.40    21_600.00       0.8749          1.0286            1.0197         5.36
IVF-OPQ-nl158-m128-np7 (query)                        18_574.10     1_716.07    20_290.18       0.9631          1.0027            1.0000         8.42
IVF-OPQ-nl158-m128-np12 (query)                       18_574.10     2_651.57    21_225.67       0.9638          1.0025            1.0000         8.42
IVF-OPQ-nl158-m128-np17 (query)                       18_574.10     3_590.76    22_164.86       0.9638          1.0025            1.0000         8.42
IVF-OPQ-nl158-m128 (self)                             18_574.10    13_416.06    31_990.17       0.9454          1.0056            1.0013         8.42
IVF-OPQ-nl223-m16-np11 (query)                         6_579.55       742.15     7_321.71       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np14 (query)                         6_579.55       822.17     7_401.73       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16-np21 (query)                         6_579.55     1_059.70     7_639.25       0.8980          1.0193            1.0141         3.20
IVF-OPQ-nl223-m16 (self)                               6_579.55     4_877.07    11_456.62       0.8555          1.0393            1.0269         3.20
IVF-OPQ-nl223-m32-np11 (query)                         8_542.25       935.49     9_477.75       0.9075          1.0154            1.0109         3.96
IVF-OPQ-nl223-m32-np14 (query)                         8_542.25     1_080.91     9_623.16       0.9075          1.0154            1.0109         3.96
IVF-OPQ-nl223-m32-np21 (query)                         8_542.25     1_419.97     9_962.22       0.9076          1.0154            1.0108         3.96
IVF-OPQ-nl223-m32 (self)                               8_542.25     6_101.09    14_643.34       0.8701          1.0310            1.0212         3.96
IVF-OPQ-nl223-m64-np11 (query)                        12_498.65     1_425.97    13_924.61       0.9168          1.0124            1.0083         5.49
IVF-OPQ-nl223-m64-np14 (query)                        12_498.65     1_710.46    14_209.11       0.9169          1.0123            1.0083         5.49
IVF-OPQ-nl223-m64-np21 (query)                        12_498.65     2_384.65    14_883.29       0.9169          1.0123            1.0083         5.49
IVF-OPQ-nl223-m64 (self)                              12_498.65     9_230.43    21_729.07       0.8824          1.0249            1.0169         5.49
IVF-OPQ-nl223-m128-np11 (query)                       18_043.66     2_220.04    20_263.69       0.9660          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np14 (query)                       18_043.66     2_712.80    20_756.45       0.9661          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128-np21 (query)                       18_043.66     3_857.01    21_900.67       0.9661          1.0021            1.0000         8.54
IVF-OPQ-nl223-m128 (self)                             18_043.66    14_265.66    32_309.31       0.9491          1.0049            1.0008         8.54
IVF-OPQ-nl316-m16-np15 (query)                         6_732.44       841.79     7_574.23       0.9054          1.0162            1.0116         3.88
IVF-OPQ-nl316-m16-np17 (query)                         6_732.44       893.96     7_626.40       0.9054          1.0162            1.0116         3.88
IVF-OPQ-nl316-m16-np25 (query)                         6_732.44     1_140.17     7_872.62       0.9054          1.0161            1.0116         3.88
IVF-OPQ-nl316-m16 (self)                               6_732.44     5_191.89    11_924.33       0.8661          1.0330            1.0225         3.88
IVF-OPQ-nl316-m32-np15 (query)                         8_709.47     1_106.14     9_815.62       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32-np17 (query)                         8_709.47     1_188.82     9_898.30       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32-np25 (query)                         8_709.47     1_554.33    10_263.80       0.9145          1.0130            1.0092         4.65
IVF-OPQ-nl316-m32 (self)                               8_709.47     6_657.36    15_366.84       0.8786          1.0267            1.0180         4.65
IVF-OPQ-nl316-m64-np15 (query)                        12_692.96     1_720.27    14_413.23       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64-np17 (query)                        12_692.96     1_890.63    14_583.59       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64-np25 (query)                        12_692.96     2_598.32    15_291.28       0.9208          1.0110            1.0075         6.17
IVF-OPQ-nl316-m64 (self)                              12_692.96     9_998.51    22_691.47       0.8884          1.0222            1.0150         6.17
IVF-OPQ-nl316-m128-np15 (query)                       18_264.87     2_685.16    20_950.03       0.9691          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np17 (query)                       18_264.87     2_993.78    21_258.64       0.9691          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128-np25 (query)                       18_264.87     4_213.17    22_478.04       0.9691          1.0017            1.0000         9.23
IVF-OPQ-nl316-m128 (self)                             18_264.87    15_494.38    33_759.25       0.9520          1.0044            1.0004         9.23
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
Exhaustive (query)                                        99.23     1_790.56     1_889.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.23     5_950.93     6_050.17       1.0000          1.0000            1.0000       146.48
Exhaustive-OPQ-m16 (query)                             9_313.44     1_500.87    10_814.31       0.7383          1.1306            1.1121         3.76
Exhaustive-OPQ-m16 (self)                              9_313.44     8_142.24    17_455.68       0.6595          1.2295            1.1962         3.76
Exhaustive-OPQ-m32 (query)                            11_498.27     2_336.90    13_835.17       0.8493          1.0411            1.0321         4.53
Exhaustive-OPQ-m32 (self)                             11_498.27    10_902.18    22_400.45       0.8006          1.0714            1.0580         4.53
Exhaustive-OPQ-m64 (query)                            16_404.69     4_593.67    20_998.36       0.8796          1.0255            1.0186         6.05
Exhaustive-OPQ-m64 (self)                             16_404.69    18_249.87    34_654.56       0.8413          1.0441            1.0341         6.05
Exhaustive-OPQ-m128 (query)                           24_444.77     8_631.87    33_076.63       0.9051          1.0148            1.0104         9.11
Exhaustive-OPQ-m128 (self)                            24_444.77    31_975.54    56_420.30       0.8741          1.0264            1.0201         9.11
IVF-OPQ-nl158-m16-np7 (query)                         11_490.11     1_164.88    12_654.98       0.8920          1.0223            1.0161         4.98
IVF-OPQ-nl158-m16-np12 (query)                        11_490.11     1_372.86    12_862.97       0.8922          1.0222            1.0161         4.98
IVF-OPQ-nl158-m16-np17 (query)                        11_490.11     1_589.74    13_079.85       0.8922          1.0222            1.0161         4.98
IVF-OPQ-nl158-m16 (self)                              11_490.11     9_057.28    20_547.38       0.8473          1.0438            1.0304         4.98
IVF-OPQ-nl158-m32-np7 (query)                         13_534.94     1_360.97    14_895.91       0.9369          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32-np12 (query)                        13_534.94     1_703.03    15_237.97       0.9371          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32-np17 (query)                        13_534.94     2_031.43    15_566.37       0.9371          1.0085            1.0036         5.74
IVF-OPQ-nl158-m32 (self)                              13_534.94     9_966.91    23_501.85       0.9074          1.0183            1.0075         5.74
IVF-OPQ-nl158-m64-np7 (query)                         17_754.17     1_700.43    19_454.61       0.9509          1.0052            1.0013         7.27
IVF-OPQ-nl158-m64-np12 (query)                        17_754.17     2_284.88    20_039.06       0.9512          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64-np17 (query)                        17_754.17     2_860.62    20_614.79       0.9512          1.0051            1.0013         7.27
IVF-OPQ-nl158-m64 (self)                              17_754.17    12_697.53    30_451.70       0.9272          1.0115            1.0036         7.27
IVF-OPQ-nl158-m128-np7 (query)                        26_450.12     2_618.93    29_069.05       0.9607          1.0034            1.0000        10.32
IVF-OPQ-nl158-m128-np12 (query)                       26_450.12     3_837.45    30_287.57       0.9610          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128-np17 (query)                       26_450.12     5_007.09    31_457.21       0.9610          1.0033            1.0000        10.32
IVF-OPQ-nl158-m128 (self)                             26_450.12    19_951.05    46_401.17       0.9396          1.0077            1.0017        10.32
IVF-OPQ-nl223-m16-np11 (query)                        10_513.24     1_310.38    11_823.61       0.9001          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16-np14 (query)                        10_513.24     1_436.02    11_949.26       0.9001          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16-np21 (query)                        10_513.24     1_687.27    12_200.50       0.9001          1.0187            1.0134         5.17
IVF-OPQ-nl223-m16 (self)                              10_513.24     8_848.12    19_361.35       0.8592          1.0363            1.0252         5.17
IVF-OPQ-nl223-m32-np11 (query)                        12_706.12     1_573.63    14_279.75       0.9422          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np14 (query)                        12_706.12     1_779.11    14_485.23       0.9423          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32-np21 (query)                        12_706.12     2_220.11    14_926.23       0.9423          1.0071            1.0027         5.93
IVF-OPQ-nl223-m32 (self)                              12_706.12    10_575.69    23_281.81       0.9156          1.0148            1.0059         5.93
IVF-OPQ-nl223-m64-np11 (query)                        17_243.54     2_059.74    19_303.28       0.9544          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np14 (query)                        17_243.54     2_368.74    19_612.28       0.9544          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64-np21 (query)                        17_243.54     3_082.58    20_326.12       0.9544          1.0045            1.0008         7.46
IVF-OPQ-nl223-m64 (self)                              17_243.54    13_503.13    30_746.67       0.9329          1.0097            1.0027         7.46
IVF-OPQ-nl223-m128-np11 (query)                       25_732.05     3_317.40    29_049.45       0.9640          1.0028            1.0000        10.51
IVF-OPQ-nl223-m128-np14 (query)                       25_732.05     3_969.95    29_702.00       0.9641          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128-np21 (query)                       25_732.05     5_539.19    31_271.24       0.9641          1.0027            1.0000        10.51
IVF-OPQ-nl223-m128 (self)                             25_732.05    21_561.63    47_293.68       0.9433          1.0068            1.0011        10.51
IVF-OPQ-nl316-m16-np15 (query)                        10_865.12     1_453.96    12_319.08       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np17 (query)                        10_865.12     1_534.59    12_399.71       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16-np25 (query)                        10_865.12     1_847.44    12_712.56       0.9039          1.0172            1.0120         6.19
IVF-OPQ-nl316-m16 (self)                              10_865.12     9_301.63    20_166.75       0.8640          1.0338            1.0232         6.19
IVF-OPQ-nl316-m32-np15 (query)                        13_033.25     1_812.93    14_846.18       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np17 (query)                        13_033.25     1_931.86    14_965.12       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32-np25 (query)                        13_033.25     2_418.95    15_452.21       0.9447          1.0063            1.0022         6.96
IVF-OPQ-nl316-m32 (self)                              13_033.25    11_233.68    24_266.93       0.9197          1.0135            1.0051         6.96
IVF-OPQ-nl316-m64-np15 (query)                        17_896.01     2_416.45    20_312.46       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64-np17 (query)                        17_896.01     2_600.25    20_496.26       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64-np25 (query)                        17_896.01     3_370.55    21_266.56       0.9563          1.0040            1.0005         8.48
IVF-OPQ-nl316-m64 (self)                              17_896.01    14_471.88    32_367.89       0.9353          1.0090            1.0024         8.48
IVF-OPQ-nl316-m128-np15 (query)                       26_065.09     4_016.90    30_081.99       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np17 (query)                       26_065.09     4_446.68    30_511.77       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128-np25 (query)                       26_065.09     6_120.62    32_185.71       0.9655          1.0024            1.0000        11.54
IVF-OPQ-nl316-m128 (self)                             26_065.09    23_599.12    49_664.21       0.9460          1.0062            1.0009        11.54
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
Exhaustive (query)                                        68.02     1_226.66     1_294.68       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_809.04       126.86     1_935.90       0.3444          1.0764            1.0758         2.24
IVFPQ-m64-nl111-np1                                    2_709.68       225.56     2_935.23       0.4480          1.0516            1.0445         3.77
SOARPQ-shift0.5-m32-nl111-np1                          1_958.12       139.16     2_097.28       0.3188          1.3217            1.0785         4.72
IVFPQ-m32-nl111-np2                                    1_809.04       180.96     1_990.00       0.3458          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np2                                    2_709.68       330.14     3_039.82       0.4506          1.0507            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np2                          1_958.12       187.54     2_145.66       0.3454          1.0766            1.0756         4.72
IVFPQ-m32-nl111-np4                                    1_809.04       267.55     2_076.59       0.3458          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np4                                    2_709.68       493.56     3_203.23       0.4506          1.0507            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np4                          1_958.12       280.50     2_238.62       0.3458          1.0759            1.0755         4.72
IVFPQ-m32-nl111-np5                                    1_809.04       317.08     2_126.12       0.3458          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np5                                    2_709.68       616.40     3_326.07       0.4506          1.0507            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np5                          1_958.12       329.54     2_287.66       0.3458          1.0759            1.0755         4.72
IVFPQ-m32-nl111-np8                                    1_809.04       448.76     2_257.80       0.3458          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np8                                    2_709.68       845.30     3_554.98       0.4506          1.0507            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np8                          1_958.12       474.03     2_432.15       0.3458          1.0759            1.0755         4.72
IVFPQ-m32-nl111-np10                                   1_809.04       544.14     2_353.19       0.3458          1.0759            1.0755         2.24
IVFPQ-m64-nl111-np10                                   2_709.68     1_015.85     3_725.52       0.4506          1.0507            1.0442         3.77
SOARPQ-shift0.5-m32-nl111-np10                         1_958.12       630.34     2_588.46       0.3458          1.0759            1.0755         4.72
IVFPQ-m32-nl158-np1                                    2_883.40       131.73     3_015.13       0.3501          1.0728            1.0731         2.34
IVFPQ-m64-nl158-np1                                    3_782.78       216.73     3_999.50       0.4544          1.0480            1.0435         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_092.31       134.01     3_226.32       0.3119          1.2089            1.0777         4.82
IVFPQ-m32-nl158-np2                                    2_883.40       177.39     3_060.79       0.3544          1.0713            1.0721         2.34
IVFPQ-m64-nl158-np2                                    3_782.78       310.62     4_093.40       0.4622          1.0459            1.0424         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_092.31       186.75     3_279.06       0.3537          1.0726            1.0723         4.82
IVFPQ-m32-nl158-np4                                    2_883.40       272.74     3_156.14       0.3545          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np4                                    3_782.78       479.49     4_262.27       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_092.31       276.00     3_368.31       0.3545          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np7                                    2_883.40       403.21     3_286.60       0.3545          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np7                                    3_782.78       755.54     4_538.32       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_092.31       414.10     3_506.41       0.3545          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np8                                    2_883.40       461.61     3_345.01       0.3545          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np8                                    3_782.78       828.00     4_610.77       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_092.31       462.26     3_554.56       0.3545          1.0712            1.0721         4.82
IVFPQ-m32-nl158-np12                                   2_883.40       643.40     3_526.80       0.3545          1.0712            1.0721         2.34
IVFPQ-m64-nl158-np12                                   3_782.78     1_160.36     4_943.13       0.4625          1.0458            1.0423         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_092.31       659.38     3_751.68       0.3545          1.0712            1.0721         4.82
IVFPQ-m32-nl223-np1                                    2_206.30       106.79     2_313.08       0.3514          1.0704            1.0702         2.46
IVFPQ-m64-nl223-np1                                    3_143.50       177.33     3_320.83       0.4366          1.0504            1.0449         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_463.02       121.49     2_584.51       0.3276          1.1792            1.0736         4.95
IVFPQ-m32-nl223-np2                                    2_206.30       158.83     2_365.13       0.3648          1.0664            1.0669         2.46
IVFPQ-m64-nl223-np2                                    3_143.50       262.28     3_405.78       0.4657          1.0445            1.0404         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_463.02       177.74     2_640.76       0.3617          1.0687            1.0686         4.95
IVFPQ-m32-nl223-np4                                    2_206.30       262.47     2_468.77       0.3682          1.0656            1.0658         2.46
IVFPQ-m64-nl223-np4                                    3_143.50       456.23     3_599.73       0.4753          1.0431            1.0389         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_463.02       274.82     2_737.84       0.3661          1.0669            1.0666         4.95
IVFPQ-m32-nl223-np8                                    2_206.30       459.36     2_665.66       0.3686          1.0655            1.0657         2.46
IVFPQ-m64-nl223-np8                                    3_143.50       828.52     3_972.02       0.4766          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_463.02       467.68     2_930.71       0.3683          1.0658            1.0657         4.95
IVFPQ-m32-nl223-np11                                   2_206.30       602.77     2_809.07       0.3686          1.0655            1.0657         2.46
IVFPQ-m64-nl223-np11                                   3_143.50     1_093.70     4_237.21       0.4766          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_463.02       615.36     3_078.38       0.3686          1.0655            1.0657         4.95
IVFPQ-m32-nl223-np14                                   2_206.30       747.01     2_953.31       0.3686          1.0655            1.0657         2.46
IVFPQ-m64-nl223-np14                                   3_143.50     1_356.97     4_500.47       0.4766          1.0429            1.0387         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_463.02       759.26     3_222.28       0.3686          1.0655            1.0657         4.95
IVFPQ-m32-nl316-np1                                    2_679.79       109.37     2_789.16       0.3523          1.0693            1.0683         2.65
IVFPQ-m64-nl316-np1                                    3_710.25       184.38     3_894.63       0.4283          1.0513            1.0458         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_882.89       116.68     2_999.58       0.3250          1.1522            1.0724         5.13
IVFPQ-m32-nl316-np2                                    2_679.79       158.68     2_838.47       0.3722          1.0628            1.0637         2.65
IVFPQ-m64-nl316-np2                                    3_710.25       248.81     3_959.06       0.4685          1.0425            1.0391         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_882.89       185.96     3_068.86       0.3687          1.0654            1.0658         5.13
IVFPQ-m32-nl316-np4                                    2_679.79       277.00     2_956.80       0.3782          1.0613            1.0621         2.65
IVFPQ-m64-nl316-np4                                    3_710.25       441.85     4_152.10       0.4843          1.0401            1.0368         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_882.89       283.10     3_166.00       0.3748          1.0631            1.0635         5.13
IVFPQ-m32-nl316-np8                                    2_679.79       476.65     3_156.44       0.3790          1.0611            1.0619         2.65
IVFPQ-m64-nl316-np8                                    3_710.25       799.99     4_510.24       0.4878          1.0396            1.0363         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_882.89       477.03     3_359.92       0.3782          1.0616            1.0622         5.13
IVFPQ-m32-nl316-np15                                   2_679.79       815.71     3_495.50       0.3790          1.0612            1.0619         2.65
IVFPQ-m64-nl316-np15                                   3_710.25     1_436.22     5_146.47       0.4880          1.0396            1.0362         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_882.89       834.68     3_717.57       0.3790          1.0612            1.0619         5.13
IVFPQ-m32-nl316-np17                                   2_679.79       911.27     3_591.06       0.3790          1.0612            1.0619         2.65
IVFPQ-m64-nl316-np17                                   3_710.25     1_619.47     5_329.72       0.4880          1.0396            1.0362         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_882.89       911.87     3_794.76       0.3790          1.0612            1.0619         5.13
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
Exhaustive (query)                                        68.02     1_226.66     1_294.68       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        3_257.78       133.65     3_391.43       0.3126          1.2056            1.0776         4.82
SOARPQ-near-np2                                        3_257.78       187.15     3_444.93       0.3538          1.0724            1.0723         4.82
SOARPQ-near-np4                                        3_257.78       283.06     3_540.83       0.3545          1.0712            1.0721         4.82
SOARPQ-near-np7                                        3_257.78       425.22     3_683.00       0.3545          1.0712            1.0721         4.82
SOARPQ-near-np8                                        3_257.78       481.07     3_738.85       0.3545          1.0712            1.0721         4.82
SOARPQ-near-np12                                       3_257.78       667.85     3_925.63       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.3-np1                                    3_156.32       134.20     3_290.53       0.3119          1.2080            1.0776         4.82
SOARPQ-shift0.3-np2                                    3_156.32       189.76     3_346.09       0.3537          1.0725            1.0723         4.82
SOARPQ-shift0.3-np4                                    3_156.32       284.53     3_440.86       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.3-np7                                    3_156.32       433.68     3_590.01       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.3-np8                                    3_156.32       482.40     3_638.73       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.3-np12                                   3_156.32       670.47     3_826.79       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.7-np1                                    3_330.64       137.19     3_467.84       0.3118          1.2101            1.0777         4.82
SOARPQ-shift0.7-np2                                    3_330.64       182.81     3_513.45       0.3537          1.0726            1.0723         4.82
SOARPQ-shift0.7-np4                                    3_330.64       296.38     3_627.02       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.7-np7                                    3_330.64       427.27     3_757.91       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.7-np8                                    3_330.64       481.79     3_812.43       0.3545          1.0712            1.0721         4.82
SOARPQ-shift0.7-np12                                   3_330.64       666.79     3_997.43       0.3545          1.0712            1.0721         4.82
SOARPQ-orth1-np1                                       3_263.57       133.67     3_397.24       0.3131          1.2071            1.0775         4.82
SOARPQ-orth1-np2                                       3_263.57       183.86     3_447.43       0.3538          1.0722            1.0723         4.82
SOARPQ-orth1-np4                                       3_263.57       278.81     3_542.38       0.3545          1.0712            1.0721         4.82
SOARPQ-orth1-np7                                       3_263.57       429.30     3_692.87       0.3545          1.0712            1.0721         4.82
SOARPQ-orth1-np8                                       3_263.57       478.08     3_741.65       0.3545          1.0712            1.0721         4.82
SOARPQ-orth1-np12                                      3_263.57       672.47     3_936.05       0.3545          1.0712            1.0721         4.82
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
Exhaustive (query)                                        68.17     1_352.46     1_420.63       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    2_065.49       126.81     2_192.30       0.4769          1.0785            1.0741         2.24
IVFPQ-m64-nl111-np1                                    3_064.83       220.01     3_284.84       0.6137          1.0403            1.0355         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_254.71       129.69     2_384.40       0.4634          1.1010            1.0768         4.72
IVFPQ-m32-nl111-np2                                    2_065.49       172.60     2_238.08       0.4824          1.0762            1.0733         2.24
IVFPQ-m64-nl111-np2                                    3_064.83       305.00     3_369.83       0.6231          1.0370            1.0349         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_254.71       191.04     2_445.74       0.4814          1.0779            1.0736         4.72
IVFPQ-m32-nl111-np4                                    2_065.49       277.71     2_343.19       0.4826          1.0762            1.0732         2.24
IVFPQ-m64-nl111-np4                                    3_064.83       503.04     3_567.87       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_254.71       289.23     2_543.94       0.4825          1.0762            1.0733         4.72
IVFPQ-m32-nl111-np5                                    2_065.49       331.01     2_396.50       0.4826          1.0762            1.0732         2.24
IVFPQ-m64-nl111-np5                                    3_064.83       574.29     3_639.12       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_254.71       342.21     2_596.92       0.4826          1.0762            1.0732         4.72
IVFPQ-m32-nl111-np8                                    2_065.49       470.90     2_536.38       0.4826          1.0762            1.0732         2.24
IVFPQ-m64-nl111-np8                                    3_064.83       840.96     3_905.79       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_254.71       520.76     2_775.47       0.4826          1.0762            1.0732         4.72
IVFPQ-m32-nl111-np10                                   2_065.49       608.43     2_673.92       0.4826          1.0762            1.0732         2.24
IVFPQ-m64-nl111-np10                                   3_064.83     1_052.06     4_116.89       0.6236          1.0368            1.0348         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_254.71       639.08     2_893.79       0.4826          1.0762            1.0732         4.72
IVFPQ-m32-nl158-np1                                    3_214.15       128.58     3_342.72       0.4826          1.0759            1.0725         2.34
IVFPQ-m64-nl158-np1                                    4_272.46       223.99     4_496.45       0.6177          1.0395            1.0345         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_343.65       135.06     3_478.71       0.4826          1.0800            1.0735         4.82
IVFPQ-m32-nl158-np2                                    3_214.15       186.44     3_400.59       0.4898          1.0730            1.0714         2.34
IVFPQ-m64-nl158-np2                                    4_272.46       317.32     4_589.78       0.6295          1.0354            1.0337         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_343.65       183.59     3_527.24       0.4895          1.0738            1.0717         4.82
IVFPQ-m32-nl158-np4                                    3_214.15       271.42     3_485.56       0.4903          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np4                                    4_272.46       491.65     4_764.12       0.6306          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_343.65       285.73     3_629.38       0.4902          1.0729            1.0713         4.82
IVFPQ-m32-nl158-np7                                    3_214.15       430.14     3_644.29       0.4903          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np7                                    4_272.46       747.75     5_020.21       0.6306          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_343.65       424.77     3_768.42       0.4903          1.0728            1.0712         4.82
IVFPQ-m32-nl158-np8                                    3_214.15       486.83     3_700.98       0.4903          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np8                                    4_272.46       854.98     5_127.44       0.6306          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_343.65       488.99     3_832.64       0.4903          1.0728            1.0712         4.82
IVFPQ-m32-nl158-np12                                   3_214.15       670.68     3_884.82       0.4903          1.0728            1.0712         2.34
IVFPQ-m64-nl158-np12                                   4_272.46     1_202.23     5_474.69       0.6306          1.0351            1.0335         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_343.65       705.11     4_048.76       0.4903          1.0728            1.0712         4.82
IVFPQ-m32-nl223-np1                                    2_522.65       114.92     2_637.57       0.3974          1.1048            1.1003         2.46
IVFPQ-m64-nl223-np1                                    3_616.44       165.37     3_781.81       0.4715          1.0736            1.0670         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_809.68       122.03     2_931.71       0.4486          1.0881            1.0841         4.95
IVFPQ-m32-nl223-np2                                    2_522.65       167.08     2_689.73       0.4565          1.0827            1.0807         2.46
IVFPQ-m64-nl223-np2                                    3_616.44       267.48     3_883.92       0.5691          1.0476            1.0436         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_809.68       178.47     2_988.15       0.4779          1.0767            1.0755         4.95
IVFPQ-m32-nl223-np4                                    2_522.65       268.01     2_790.65       0.4835          1.0746            1.0732         2.46
IVFPQ-m64-nl223-np4                                    3_616.44       462.08     4_078.52       0.6177          1.0374            1.0356         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_809.68       285.30     3_094.98       0.4888          1.0732            1.0718         4.95
IVFPQ-m32-nl223-np8                                    2_522.65       477.34     2_999.99       0.4902          1.0727            1.0714         2.46
IVFPQ-m64-nl223-np8                                    3_616.44       824.65     4_441.09       0.6319          1.0346            1.0332         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_809.68       482.29     3_291.96       0.4902          1.0726            1.0713         4.95
IVFPQ-m32-nl223-np11                                   2_522.65       627.07     3_149.71       0.4904          1.0726            1.0713         2.46
IVFPQ-m64-nl223-np11                                   3_616.44     1_105.88     4_722.32       0.6327          1.0345            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_809.68       633.33     3_443.01       0.4904          1.0726            1.0713         4.95
IVFPQ-m32-nl223-np14                                   2_522.65       773.23     3_295.88       0.4904          1.0726            1.0713         2.46
IVFPQ-m64-nl223-np14                                   3_616.44     1_376.50     4_992.95       0.6327          1.0345            1.0330         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_809.68       791.76     3_601.44       0.4904          1.0726            1.0713         4.95
IVFPQ-m32-nl316-np1                                    2_727.41       109.35     2_836.75       0.3504          1.1213            1.1192         2.65
IVFPQ-m64-nl316-np1                                    3_593.61       152.74     3_746.34       0.3990          1.0929            1.0886         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_983.98       116.25     3_100.23       0.4209          1.0954            1.0940         5.13
IVFPQ-m32-nl316-np2                                    2_727.41       156.04     2_883.45       0.4241          1.0925            1.0914         2.65
IVFPQ-m64-nl316-np2                                    3_593.61       246.41     3_840.02       0.5158          1.0590            1.0566         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_983.98       172.05     3_156.03       0.4619          1.0812            1.0801         5.13
IVFPQ-m32-nl316-np4                                    2_727.41       258.79     2_986.20       0.4687          1.0785            1.0774         2.65
IVFPQ-m64-nl316-np4                                    3_593.61       435.27     4_028.87       0.5953          1.0419            1.0397         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_983.98       281.86     3_265.84       0.4828          1.0745            1.0733         5.13
IVFPQ-m32-nl316-np8                                    2_727.41       465.85     3_193.25       0.4866          1.0734            1.0723         2.65
IVFPQ-m64-nl316-np8                                    3_593.61       803.26     4_396.86       0.6311          1.0347            1.0333         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_983.98       474.79     3_458.77       0.4876          1.0732            1.0722         5.13
IVFPQ-m32-nl316-np15                                   2_727.41       805.58     3_532.99       0.4882          1.0729            1.0718         2.65
IVFPQ-m64-nl316-np15                                   3_593.61     1_418.95     5_012.56       0.6355          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_983.98       804.42     3_788.41       0.4882          1.0729            1.0718         5.13
IVFPQ-m32-nl316-np17                                   2_727.41       906.50     3_633.91       0.4882          1.0729            1.0718         2.65
IVFPQ-m64-nl316-np17                                   3_593.61     1_599.94     5_193.55       0.6355          1.0339            1.0324         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_983.98       899.77     3_883.75       0.4882          1.0729            1.0718         5.13
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
Exhaustive (query)                                        68.17     1_352.46     1_420.63       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        2_952.05       133.70     3_085.75       0.4831          1.0795            1.0733         4.82
SOARPQ-near-np2                                        2_952.05       176.41     3_128.46       0.4897          1.0736            1.0716         4.82
SOARPQ-near-np4                                        2_952.05       268.87     3_220.92       0.4903          1.0729            1.0712         4.82
SOARPQ-near-np7                                        2_952.05       410.27     3_362.32       0.4903          1.0728            1.0712         4.82
SOARPQ-near-np8                                        2_952.05       458.87     3_410.92       0.4903          1.0728            1.0712         4.82
SOARPQ-near-np12                                       2_952.05       661.15     3_613.20       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.3-np1                                    2_940.88       128.72     3_069.60       0.4830          1.0798            1.0734         4.82
SOARPQ-shift0.3-np2                                    2_940.88       178.47     3_119.36       0.4896          1.0737            1.0717         4.82
SOARPQ-shift0.3-np4                                    2_940.88       268.72     3_209.61       0.4903          1.0729            1.0713         4.82
SOARPQ-shift0.3-np7                                    2_940.88       408.08     3_348.96       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.3-np8                                    2_940.88       461.20     3_402.09       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.3-np12                                   2_940.88       659.07     3_599.95       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.7-np1                                    2_939.20       127.82     3_067.02       0.4821          1.0803            1.0736         4.82
SOARPQ-shift0.7-np2                                    2_939.20       177.62     3_116.82       0.4894          1.0739            1.0718         4.82
SOARPQ-shift0.7-np4                                    2_939.20       270.03     3_209.24       0.4902          1.0729            1.0713         4.82
SOARPQ-shift0.7-np7                                    2_939.20       409.13     3_348.33       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.7-np8                                    2_939.20       457.97     3_397.17       0.4903          1.0728            1.0712         4.82
SOARPQ-shift0.7-np12                                   2_939.20       658.21     3_597.41       0.4903          1.0728            1.0712         4.82
SOARPQ-orth1-np1                                       2_949.21       127.48     3_076.69       0.4823          1.0802            1.0736         4.82
SOARPQ-orth1-np2                                       2_949.21       180.80     3_130.00       0.4895          1.0738            1.0717         4.82
SOARPQ-orth1-np4                                       2_949.21       274.02     3_223.23       0.4902          1.0729            1.0713         4.82
SOARPQ-orth1-np7                                       2_949.21       409.39     3_358.60       0.4903          1.0728            1.0712         4.82
SOARPQ-orth1-np8                                       2_949.21       460.19     3_409.40       0.4903          1.0728            1.0712         4.82
SOARPQ-orth1-np12                                      2_949.21       659.09     3_608.29       0.4903          1.0728            1.0712         4.82
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
Exhaustive (query)                                        69.34     1_318.82     1_388.17       1.0000          1.0000            1.0000        97.66
IVFPQ-m32-nl111-np1                                    1_950.66       102.66     2_053.32       0.7055          1.1845            1.0982         2.24
IVFPQ-m64-nl111-np1                                    2_796.45       165.13     2_961.58       0.7175          1.1739            1.0866         3.77
SOARPQ-shift0.5-m32-nl111-np1                          2_141.59       124.05     2_265.64       0.8142          1.0740            1.0458         4.72
IVFPQ-m32-nl111-np2                                    1_950.66       163.65     2_114.31       0.8212          1.0638            1.0402         2.24
IVFPQ-m64-nl111-np2                                    2_796.45       286.10     3_082.55       0.8434          1.0521            1.0275         3.77
SOARPQ-shift0.5-m32-nl111-np2                          2_141.59       207.80     2_349.39       0.8488          1.0446            1.0337         4.72
IVFPQ-m32-nl111-np4                                    1_950.66       284.34     2_235.00       0.8539          1.0384            1.0306         2.24
IVFPQ-m64-nl111-np4                                    2_796.45       530.72     3_327.17       0.8802          1.0260            1.0194         3.77
SOARPQ-shift0.5-m32-nl111-np4                          2_141.59       368.20     2_509.79       0.8555          1.0387            1.0306         4.72
IVFPQ-m32-nl111-np5                                    1_950.66       348.87     2_299.54       0.8556          1.0373            1.0301         2.24
IVFPQ-m64-nl111-np5                                    2_796.45       653.62     3_450.07       0.8821          1.0249            1.0190         3.77
SOARPQ-shift0.5-m32-nl111-np5                          2_141.59       447.09     2_588.68       0.8559          1.0382            1.0302         4.72
IVFPQ-m32-nl111-np8                                    1_950.66       541.72     2_492.38       0.8566          1.0367            1.0298         2.24
IVFPQ-m64-nl111-np8                                    2_796.45     1_025.81     3_822.26       0.8833          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np8                          2_141.59       654.85     2_796.44       0.8565          1.0371            1.0299         4.72
IVFPQ-m32-nl111-np10                                   1_950.66       670.23     2_620.89       0.8566          1.0367            1.0298         2.24
IVFPQ-m64-nl111-np10                                   2_796.45     1_274.97     4_071.42       0.8834          1.0243            1.0187         3.77
SOARPQ-shift0.5-m32-nl111-np10                         2_141.59       792.66     2_934.25       0.8566          1.0369            1.0299         4.72
IVFPQ-m32-nl158-np1                                    3_000.92        98.21     3_099.12       0.7125          1.1767            1.0939         2.34
IVFPQ-m64-nl158-np1                                    3_869.11       149.75     4_018.86       0.7212          1.1686            1.0854         3.86
SOARPQ-shift0.5-m32-nl158-np1                          3_135.48       113.13     3_248.61       0.8237          1.0695            1.0397         4.82
IVFPQ-m32-nl158-np2                                    3_000.92       154.01     3_154.93       0.8339          1.0567            1.0334         2.34
IVFPQ-m64-nl158-np2                                    3_869.11       257.27     4_126.38       0.8510          1.0477            1.0238         3.86
SOARPQ-shift0.5-m32-nl158-np2                          3_135.48       183.74     3_319.22       0.8617          1.0397            1.0274         4.82
IVFPQ-m32-nl158-np4                                    3_000.92       267.48     3_268.39       0.8687          1.0320            1.0241         2.34
IVFPQ-m64-nl158-np4                                    3_869.11       474.67     4_343.79       0.8891          1.0225            1.0159         3.86
SOARPQ-shift0.5-m32-nl158-np4                          3_135.48       318.65     3_454.13       0.8703          1.0331            1.0244         4.82
IVFPQ-m32-nl158-np7                                    3_000.92       437.09     3_438.00       0.8726          1.0297            1.0231         2.34
IVFPQ-m64-nl158-np7                                    3_869.11       807.58     4_676.69       0.8936          1.0202            1.0150         3.86
SOARPQ-shift0.5-m32-nl158-np7                          3_135.48       520.55     3_656.04       0.8724          1.0305            1.0234         4.82
IVFPQ-m32-nl158-np8                                    3_000.92       495.40     3_496.32       0.8730          1.0295            1.0230         2.34
IVFPQ-m64-nl158-np8                                    3_869.11       915.58     4_784.69       0.8939          1.0201            1.0150         3.86
SOARPQ-shift0.5-m32-nl158-np8                          3_135.48       580.21     3_715.70       0.8727          1.0302            1.0233         4.82
IVFPQ-m32-nl158-np12                                   3_000.92       727.18     3_728.09       0.8731          1.0294            1.0230         2.34
IVFPQ-m64-nl158-np12                                   3_869.11     1_360.38     5_229.49       0.8941          1.0200            1.0149         3.86
SOARPQ-shift0.5-m32-nl158-np12                         3_135.48       835.61     3_971.10       0.8731          1.0296            1.0231         4.82
IVFPQ-m32-nl223-np1                                    2_059.55       100.06     2_159.61       0.6875          1.1973            1.1237         2.46
IVFPQ-m64-nl223-np1                                    2_892.43       143.58     3_036.01       0.6935          1.1897            1.1152         3.99
SOARPQ-shift0.5-m32-nl223-np1                          2_243.48       107.74     2_351.22       0.8134          1.0794            1.0465         4.95
IVFPQ-m32-nl223-np2                                    2_059.55       147.78     2_207.33       0.8245          1.0642            1.0373         2.46
IVFPQ-m64-nl223-np2                                    2_892.43       238.72     3_131.15       0.8396          1.0562            1.0281         3.99
SOARPQ-shift0.5-m32-nl223-np2                          2_243.48       167.84     2_411.32       0.8646          1.0397            1.0263         4.95
IVFPQ-m32-nl223-np4                                    2_059.55       252.22     2_311.77       0.8726          1.0303            1.0220         2.46
IVFPQ-m64-nl223-np4                                    2_892.43       432.23     3_324.66       0.8924          1.0218            1.0146         3.99
SOARPQ-shift0.5-m32-nl223-np4                          2_243.48       297.16     2_540.63       0.8758          1.0314            1.0219         4.95
IVFPQ-m32-nl223-np8                                    2_059.55       470.54     2_530.08       0.8792          1.0267            1.0203         2.46
IVFPQ-m64-nl223-np8                                    2_892.43       836.54     3_728.97       0.8998          1.0180            1.0130         3.99
SOARPQ-shift0.5-m32-nl223-np8                          2_243.48       525.77     2_769.25       0.8786          1.0279            1.0206         4.95
IVFPQ-m32-nl223-np11                                   2_059.55       630.35     2_689.90       0.8795          1.0266            1.0202         2.46
IVFPQ-m64-nl223-np11                                   2_892.43     1_132.74     4_025.17       0.9003          1.0178            1.0129         3.99
SOARPQ-shift0.5-m32-nl223-np11                         2_243.48       701.91     2_945.39       0.8792          1.0271            1.0203         4.95
IVFPQ-m32-nl223-np14                                   2_059.55       793.48     2_853.03       0.8795          1.0265            1.0202         2.46
IVFPQ-m64-nl223-np14                                   2_892.43     1_441.71     4_334.14       0.9003          1.0178            1.0129         3.99
SOARPQ-shift0.5-m32-nl223-np14                         2_243.48       878.39     3_121.86       0.8794          1.0268            1.0202         4.95
IVFPQ-m32-nl316-np1                                    2_273.82       105.03     2_378.85       0.6725          1.2102            1.1380         2.65
IVFPQ-m64-nl316-np1                                    3_147.48       143.73     3_291.21       0.6767          1.2049            1.1332         4.17
SOARPQ-shift0.5-m32-nl316-np1                          2_502.65       111.29     2_613.94       0.8093          1.0840            1.0496         5.13
IVFPQ-m32-nl316-np2                                    2_273.82       150.32     2_424.14       0.8228          1.0663            1.0373         2.65
IVFPQ-m64-nl316-np2                                    3_147.48       239.62     3_387.10       0.8330          1.0605            1.0306         4.17
SOARPQ-shift0.5-m32-nl316-np2                          2_502.65       164.91     2_667.57       0.8708          1.0389            1.0233         5.13
IVFPQ-m32-nl316-np4                                    2_273.82       251.23     2_525.05       0.8817          1.0267            1.0187         2.65
IVFPQ-m64-nl316-np4                                    3_147.48       418.31     3_565.79       0.8960          1.0207            1.0134         4.17
SOARPQ-shift0.5-m32-nl316-np4                          2_502.65       273.71     2_776.36       0.8858          1.0287            1.0184         5.13
IVFPQ-m32-nl316-np8                                    2_273.82       459.08     2_732.89       0.8910          1.0218            1.0162         2.65
IVFPQ-m64-nl316-np8                                    3_147.48       791.29     3_938.77       0.9056          1.0158            1.0114         4.17
SOARPQ-shift0.5-m32-nl316-np8                          2_502.65       496.56     2_999.22       0.8900          1.0240            1.0167         5.13
IVFPQ-m32-nl316-np15                                   2_273.82       817.80     3_091.62       0.8917          1.0215            1.0159         2.65
IVFPQ-m64-nl316-np15                                   3_147.48     1_467.72     4_615.21       0.9064          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np15                         2_502.65       881.26     3_383.91       0.8915          1.0218            1.0161         5.13
IVFPQ-m32-nl316-np17                                   2_273.82       928.14     3_201.96       0.8917          1.0215            1.0159         2.65
IVFPQ-m64-nl316-np17                                   3_147.48     1_645.35     4_792.84       0.9064          1.0155            1.0112         4.17
SOARPQ-shift0.5-m32-nl316-np17                         2_502.65     1_020.41     3_523.07       0.8916          1.0216            1.0160         5.13
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
Exhaustive (query)                                        69.34     1_318.82     1_388.17       1.0000          1.0000            1.0000        97.66
SOARPQ-near-np1                                        3_778.55       120.99     3_899.54       0.8232          1.0711            1.0373         4.82
SOARPQ-near-np2                                        3_778.55       205.52     3_984.07       0.8631          1.0379            1.0263         4.82
SOARPQ-near-np4                                        3_778.55       369.07     4_147.62       0.8714          1.0316            1.0238         4.82
SOARPQ-near-np7                                        3_778.55       534.06     4_312.61       0.8727          1.0300            1.0232         4.82
SOARPQ-near-np8                                        3_778.55       613.85     4_392.40       0.8729          1.0298            1.0232         4.82
SOARPQ-near-np12                                       3_778.55       845.25     4_623.80       0.8731          1.0295            1.0230         4.82
SOARPQ-shift0.3-np1                                    3_160.18       114.43     3_274.61       0.8257          1.0677            1.0384         4.82
SOARPQ-shift0.3-np2                                    3_160.18       183.17     3_343.35       0.8627          1.0386            1.0268         4.82
SOARPQ-shift0.3-np4                                    3_160.18       322.17     3_482.35       0.8707          1.0325            1.0242         4.82
SOARPQ-shift0.3-np7                                    3_160.18       518.78     3_678.96       0.8725          1.0304            1.0233         4.82
SOARPQ-shift0.3-np8                                    3_160.18       585.10     3_745.28       0.8727          1.0301            1.0233         4.82
SOARPQ-shift0.3-np12                                   3_160.18       834.31     3_994.49       0.8731          1.0296            1.0230         4.82
SOARPQ-shift0.7-np1                                    3_173.85       115.22     3_289.08       0.8207          1.0725            1.0413         4.82
SOARPQ-shift0.7-np2                                    3_173.85       182.56     3_356.42       0.8608          1.0409            1.0279         4.82
SOARPQ-shift0.7-np4                                    3_173.85       317.71     3_491.56       0.8699          1.0337            1.0246         4.82
SOARPQ-shift0.7-np7                                    3_173.85       516.51     3_690.36       0.8723          1.0308            1.0234         4.82
SOARPQ-shift0.7-np8                                    3_173.85       581.00     3_754.85       0.8726          1.0304            1.0233         4.82
SOARPQ-shift0.7-np12                                   3_173.85       845.26     4_019.11       0.8730          1.0296            1.0231         4.82
SOARPQ-orth1-np1                                       3_165.41       114.36     3_279.77       0.8233          1.0708            1.0386         4.82
SOARPQ-orth1-np2                                       3_165.41       183.96     3_349.37       0.8623          1.0392            1.0268         4.82
SOARPQ-orth1-np4                                       3_165.41       318.28     3_483.69       0.8707          1.0327            1.0241         4.82
SOARPQ-orth1-np7                                       3_165.41       517.61     3_683.02       0.8725          1.0303            1.0233         4.82
SOARPQ-orth1-np8                                       3_165.41       581.14     3_746.55       0.8727          1.0301            1.0232         4.82
SOARPQ-orth1-np12                                      3_165.41       831.34     3_996.75       0.8731          1.0296            1.0231         4.82
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
Exhaustive (query)                                        71.49     1_314.67     1_386.17       1.0000          1.0000            1.0000        97.85
IVFPQ-m32-nl111-np1                                    1_839.73        98.75     1_938.48       0.7687          1.1400            1.0602         2.24
IVFPQ-m64-nl111-np1                                    2_687.21       155.02     2_842.24       0.7773          1.1327            1.0508         3.77
SOARPQ-orth1-m32-nl111-np1                             2_051.74       115.56     2_167.31       0.8461          1.0725            1.0359         4.72
IVFPQ-m32-nl111-np2                                    1_839.73       161.16     2_000.90       0.8642          1.0449            1.0274         2.24
IVFPQ-m64-nl111-np2                                    2_687.21       268.31     2_955.53       0.8774          1.0371            1.0203         3.77
SOARPQ-orth1-m32-nl111-np2                             2_051.74       189.67     2_241.42       0.8750          1.0430            1.0260         4.72
IVFPQ-m32-nl111-np4                                    1_839.73       275.37     2_115.10       0.8829          1.0309            1.0223         2.24
IVFPQ-m64-nl111-np4                                    2_687.21       500.63     3_187.84       0.8971          1.0232            1.0162         3.77
SOARPQ-orth1-m32-nl111-np4                             2_051.74       343.21     2_394.95       0.8820          1.0345            1.0231         4.72
IVFPQ-m32-nl111-np5                                    1_839.73       337.41     2_177.14       0.8838          1.0304            1.0221         2.24
IVFPQ-m64-nl111-np5                                    2_687.21       620.84     3_308.05       0.8981          1.0227            1.0160         3.77
SOARPQ-orth1-m32-nl111-np5                             2_051.74       403.26     2_455.01       0.8830          1.0332            1.0227         4.72
IVFPQ-m32-nl111-np8                                    1_839.73       522.18     2_361.92       0.8844          1.0302            1.0219         2.24
IVFPQ-m64-nl111-np8                                    2_687.21       979.57     3_666.79       0.8987          1.0225            1.0158         3.77
SOARPQ-orth1-m32-nl111-np8                             2_051.74       598.88     2_650.63       0.8840          1.0311            1.0221         4.72
IVFPQ-m32-nl111-np10                                   1_839.73       642.08     2_481.81       0.8844          1.0301            1.0219         2.24
IVFPQ-m64-nl111-np10                                   2_687.21     1_202.67     3_889.88       0.8987          1.0225            1.0158         3.77
SOARPQ-orth1-m32-nl111-np10                            2_051.74       728.89     2_780.63       0.8843          1.0305            1.0220         4.72
IVFPQ-m32-nl158-np1                                    2_866.47        98.93     2_965.39       0.7527          1.1567            1.0752         2.34
IVFPQ-m64-nl158-np1                                    3_731.53       145.20     3_876.73       0.7586          1.1514            1.0685         3.86
SOARPQ-orth1-m32-nl158-np1                             3_097.08       107.33     3_204.41       0.8448          1.0760            1.0352         4.82
IVFPQ-m32-nl158-np2                                    2_866.47       149.24     3_015.71       0.8658          1.0456            1.0252         2.34
IVFPQ-m64-nl158-np2                                    3_731.53       245.60     3_977.13       0.8753          1.0404            1.0199         3.86
SOARPQ-orth1-m32-nl158-np2                             3_097.08       171.74     3_268.82       0.8833          1.0399            1.0221         4.82
IVFPQ-m32-nl158-np4                                    2_866.47       261.61     3_128.07       0.8934          1.0250            1.0182         2.34
IVFPQ-m64-nl158-np4                                    3_731.53       450.23     4_181.77       0.9045          1.0197            1.0138         3.86
SOARPQ-orth1-m32-nl158-np4                             3_097.08       297.07     3_394.15       0.8929          1.0292            1.0188         4.82
IVFPQ-m32-nl158-np7                                    2_866.47       423.53     3_290.00       0.8955          1.0238            1.0178         2.34
IVFPQ-m64-nl158-np7                                    3_731.53       784.23     4_515.77       0.9069          1.0185            1.0134         3.86
SOARPQ-orth1-m32-nl158-np7                             3_097.08       480.78     3_577.86       0.8951          1.0254            1.0180         4.82
IVFPQ-m32-nl158-np8                                    2_866.47       480.28     3_346.75       0.8956          1.0238            1.0177         2.34
IVFPQ-m64-nl158-np8                                    3_731.53       873.30     4_604.83       0.9070          1.0184            1.0133         3.86
SOARPQ-orth1-m32-nl158-np8                             3_097.08       545.34     3_642.42       0.8952          1.0250            1.0179         4.82
IVFPQ-m32-nl158-np12                                   2_866.47       724.63     3_591.09       0.8957          1.0238            1.0177         2.34
IVFPQ-m64-nl158-np12                                   3_731.53     1_298.37     5_029.91       0.9071          1.0184            1.0133         3.86
SOARPQ-orth1-m32-nl158-np12                            3_097.08       785.95     3_883.03       0.8956          1.0241            1.0178         4.82
IVFPQ-m32-nl223-np1                                    2_144.28        97.01     2_241.29       0.7283          1.1815            1.1041         2.46
IVFPQ-m64-nl223-np1                                    2_959.59       139.16     3_098.75       0.7317          1.1772            1.1004         3.99
SOARPQ-orth1-m32-nl223-np1                             2_366.26       104.36     2_470.61       0.8393          1.0777            1.0371         4.95
IVFPQ-m32-nl223-np2                                    2_144.28       146.89     2_291.17       0.8613          1.0501            1.0263         2.46
IVFPQ-m64-nl223-np2                                    2_959.59       232.04     3_191.63       0.8679          1.0460            1.0217         3.99
SOARPQ-orth1-m32-nl223-np2                             2_366.26       161.38     2_527.64       0.8871          1.0370            1.0205         4.95
IVFPQ-m32-nl223-np4                                    2_144.28       247.82     2_392.10       0.8979          1.0231            1.0163         2.46
IVFPQ-m64-nl223-np4                                    2_959.59       421.28     3_380.87       0.9065          1.0191            1.0133         3.99
SOARPQ-orth1-m32-nl223-np4                             2_366.26       278.44     2_644.69       0.8976          1.0276            1.0170         4.95
IVFPQ-m32-nl223-np8                                    2_144.28       452.90     2_597.18       0.9010          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np8                                    2_959.59       814.19     3_773.78       0.9099          1.0173            1.0126         3.99
SOARPQ-orth1-m32-nl223-np8                             2_366.26       491.76     2_858.02       0.9006          1.0227            1.0160         4.95
IVFPQ-m32-nl223-np11                                   2_144.28       614.55     2_758.83       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np11                                   2_959.59     1_093.62     4_053.21       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np11                            2_366.26       657.87     3_024.13       0.9009          1.0218            1.0158         4.95
IVFPQ-m32-nl223-np14                                   2_144.28       771.66     2_915.94       0.9011          1.0214            1.0157         2.46
IVFPQ-m64-nl223-np14                                   2_959.59     1_392.23     4_351.82       0.9101          1.0173            1.0125         3.99
SOARPQ-orth1-m32-nl223-np14                            2_366.26       829.62     3_195.88       0.9011          1.0216            1.0157         4.95
IVFPQ-m32-nl316-np1                                    2_472.90       103.47     2_576.37       0.7037          1.2091            1.1328         2.65
IVFPQ-m64-nl316-np1                                    3_303.26       143.01     3_446.27       0.7071          1.2044            1.1272         4.17
SOARPQ-orth1-m32-nl316-np1                             2_762.29       107.62     2_869.91       0.8250          1.0882            1.0452         5.13
IVFPQ-m32-nl316-np2                                    2_472.90       155.01     2_627.91       0.8495          1.0587            1.0313         2.65
IVFPQ-m64-nl316-np2                                    3_303.26       231.64     3_534.89       0.8573          1.0541            1.0258         4.17
SOARPQ-orth1-m32-nl316-np2                             2_762.29       159.75     2_922.04       0.8855          1.0375            1.0212         5.13
IVFPQ-m32-nl316-np4                                    2_472.90       254.22     2_727.12       0.8982          1.0231            1.0160         2.65
IVFPQ-m64-nl316-np4                                    3_303.26       411.11     3_714.37       0.9091          1.0184            1.0119         4.17
SOARPQ-orth1-m32-nl316-np4                             2_762.29       272.29     3_034.58       0.8994          1.0269            1.0166         5.13
IVFPQ-m32-nl316-np8                                    2_472.90       460.15     2_933.05       0.9033          1.0203            1.0148         2.65
IVFPQ-m64-nl316-np8                                    3_303.26       774.27     4_077.53       0.9144          1.0156            1.0109         4.17
SOARPQ-orth1-m32-nl316-np8                             2_762.29       467.52     3_229.81       0.9028          1.0220            1.0152         5.13
IVFPQ-m32-nl316-np15                                   2_472.90       793.05     3_265.95       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np15                                   3_303.26     1_428.03     4_731.28       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np15                            2_762.29       838.92     3_601.21       0.9035          1.0204            1.0148         5.13
IVFPQ-m32-nl316-np17                                   2_472.90       900.73     3_373.63       0.9036          1.0202            1.0147         2.65
IVFPQ-m64-nl316-np17                                   3_303.26     1_696.34     4_999.59       0.9147          1.0155            1.0108         4.17
SOARPQ-orth1-m32-nl316-np17                            2_762.29       943.65     3_705.93       0.9035          1.0203            1.0148         5.13
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
Exhaustive (query)                                        71.49     1_314.67     1_386.17       1.0000          1.0000            1.0000        97.85
SOARPQ-near-np1                                        3_047.70       107.61     3_155.30       0.8512          1.0652            1.0326         4.82
SOARPQ-near-np2                                        3_047.70       171.20     3_218.89       0.8858          1.0343            1.0213         4.82
SOARPQ-near-np4                                        3_047.70       300.19     3_347.89       0.8938          1.0267            1.0185         4.82
SOARPQ-near-np7                                        3_047.70       479.61     3_527.31       0.8953          1.0244            1.0179         4.82
SOARPQ-near-np8                                        3_047.70       541.12     3_588.82       0.8954          1.0242            1.0178         4.82
SOARPQ-near-np12                                       3_047.70       783.75     3_831.45       0.8957          1.0239            1.0177         4.82
SOARPQ-shift0.3-np1                                    3_139.36       107.82     3_247.18       0.8500          1.0681            1.0342         4.82
SOARPQ-shift0.3-np2                                    3_139.36       172.82     3_312.19       0.8840          1.0373            1.0220         4.82
SOARPQ-shift0.3-np4                                    3_139.36       295.46     3_434.82       0.8927          1.0284            1.0189         4.82
SOARPQ-shift0.3-np7                                    3_139.36       484.54     3_623.90       0.8950          1.0250            1.0181         4.82
SOARPQ-shift0.3-np8                                    3_139.36       544.41     3_683.78       0.8952          1.0247            1.0180         4.82
SOARPQ-shift0.3-np12                                   3_139.36       776.32     3_915.68       0.8956          1.0240            1.0178         4.82
SOARPQ-shift0.7-np1                                    3_007.04       109.50     3_116.54       0.8446          1.0765            1.0365         4.82
SOARPQ-shift0.7-np2                                    3_007.04       170.45     3_177.50       0.8805          1.0432            1.0229         4.82
SOARPQ-shift0.7-np4                                    3_007.04       295.78     3_302.82       0.8912          1.0314            1.0195         4.82
SOARPQ-shift0.7-np7                                    3_007.04       478.72     3_485.76       0.8945          1.0262            1.0183         4.82
SOARPQ-shift0.7-np8                                    3_007.04       541.60     3_548.65       0.8949          1.0255            1.0181         4.82
SOARPQ-shift0.7-np12                                   3_007.04       777.91     3_784.95       0.8955          1.0242            1.0178         4.82
SOARPQ-orth1-np1                                       3_052.14       107.84     3_159.97       0.8448          1.0760            1.0352         4.82
SOARPQ-orth1-np2                                       3_052.14       176.73     3_228.87       0.8833          1.0399            1.0221         4.82
SOARPQ-orth1-np4                                       3_052.14       293.62     3_345.75       0.8929          1.0292            1.0188         4.82
SOARPQ-orth1-np7                                       3_052.14       480.33     3_532.47       0.8951          1.0254            1.0180         4.82
SOARPQ-orth1-np8                                       3_052.14       541.89     3_594.02       0.8952          1.0250            1.0179         4.82
SOARPQ-orth1-np12                                      3_052.14       795.33     3_847.46       0.8956          1.0241            1.0178         4.82
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
Exhaustive (query)                                        67.55     1_207.56     1_275.11       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_500.48       453.86     7_954.34       0.3575          1.0701            1.0698         3.49
IVFOPQ-m64-nl111-np1                                  11_458.13       552.69    12_010.82       0.4572          1.0464            1.0431         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_747.32       472.92     9_220.24       0.3377          1.2404            1.0721         5.98
IVFOPQ-m32-nl111-np2                                   7_500.48       501.47     8_001.95       0.3594          1.0693            1.0695         3.49
IVFOPQ-m64-nl111-np2                                  11_458.13       638.40    12_096.53       0.4603          1.0453            1.0427         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_747.32       515.13     9_262.45       0.3590          1.0699            1.0696         5.98
IVFOPQ-m32-nl111-np4                                   7_500.48       592.48     8_092.96       0.3594          1.0693            1.0695         3.49
IVFOPQ-m64-nl111-np4                                  11_458.13       808.67    12_266.80       0.4603          1.0453            1.0427         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_747.32       616.46     9_363.78       0.3594          1.0693            1.0695         5.98
IVFOPQ-m32-nl111-np5                                   7_500.48       636.27     8_136.75       0.3594          1.0693            1.0695         3.49
IVFOPQ-m64-nl111-np5                                  11_458.13       900.10    12_358.23       0.4603          1.0453            1.0427         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_747.32       666.73     9_414.05       0.3594          1.0693            1.0695         5.98
IVFOPQ-m32-nl111-np8                                   7_500.48       785.59     8_286.07       0.3594          1.0693            1.0695         3.49
IVFOPQ-m64-nl111-np8                                  11_458.13     1_159.31    12_617.44       0.4603          1.0453            1.0427         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_747.32       822.67     9_570.00       0.3594          1.0693            1.0695         5.98
IVFOPQ-m32-nl111-np10                                  7_500.48       865.08     8_365.56       0.3594          1.0693            1.0695         3.49
IVFOPQ-m64-nl111-np10                                 11_458.13     1_337.83    12_795.96       0.4603          1.0453            1.0427         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_747.32       915.09     9_662.41       0.3594          1.0693            1.0695         5.98
IVFOPQ-m32-nl158-np1                                   8_832.51       458.47     9_290.98       0.3635          1.0673            1.0680         3.84
IVFOPQ-m64-nl158-np1                                  12_897.13       539.65    13_436.78       0.4675          1.0437            1.0411         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_098.89       461.98    10_560.87       0.3322          1.1498            1.0715         6.32
IVFOPQ-m32-nl158-np2                                   8_832.51       502.06     9_334.57       0.3686          1.0656            1.0671         3.84
IVFOPQ-m64-nl158-np2                                  12_897.13       636.75    13_533.88       0.4760          1.0414            1.0399         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_098.89       513.20    10_612.08       0.3681          1.0665            1.0672         6.32
IVFOPQ-m32-nl158-np4                                   8_832.51       589.09     9_421.60       0.3688          1.0656            1.0670         3.84
IVFOPQ-m64-nl158-np4                                  12_897.13       809.70    13_706.83       0.4762          1.0414            1.0398         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_098.89       606.21    10_705.10       0.3688          1.0656            1.0670         6.32
IVFOPQ-m32-nl158-np7                                   8_832.51       724.06     9_556.57       0.3688          1.0656            1.0670         3.84
IVFOPQ-m64-nl158-np7                                  12_897.13     1_058.76    13_955.89       0.4762          1.0414            1.0398         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_098.89       756.28    10_855.17       0.3688          1.0656            1.0670         6.32
IVFOPQ-m32-nl158-np8                                   8_832.51       770.67     9_603.18       0.3688          1.0656            1.0670         3.84
IVFOPQ-m64-nl158-np8                                  12_897.13     1_158.61    14_055.74       0.4762          1.0414            1.0398         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_098.89       799.37    10_898.25       0.3688          1.0656            1.0670         6.32
IVFOPQ-m32-nl158-np12                                  8_832.51     1_012.29     9_844.80       0.3688          1.0656            1.0670         3.84
IVFOPQ-m64-nl158-np12                                 12_897.13     1_470.45    14_367.58       0.4762          1.0414            1.0398         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_098.89       980.40    11_079.29       0.3688          1.0656            1.0670         6.32
IVFOPQ-m32-nl223-np1                                   8_738.19       443.21     9_181.40       0.3598          1.0669            1.0674         3.96
IVFOPQ-m64-nl223-np1                                  12_553.87       490.53    13_044.40       0.4472          1.0462            1.0434         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_919.98       444.84    10_364.82       0.3444          1.1271            1.0695         6.45
IVFOPQ-m32-nl223-np2                                   8_738.19       491.98     9_230.17       0.3740          1.0627            1.0640         3.96
IVFOPQ-m64-nl223-np2                                  12_553.87       587.00    13_140.87       0.4768          1.0400            1.0388         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_919.98       511.53    10_431.51       0.3721          1.0642            1.0654         6.45
IVFOPQ-m32-nl223-np4                                   8_738.19       616.99     9_355.19       0.3776          1.0619            1.0631         3.96
IVFOPQ-m64-nl223-np4                                  12_553.87       772.92    13_326.79       0.4868          1.0386            1.0373         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_919.98       605.06    10_525.04       0.3762          1.0627            1.0636         6.45
IVFOPQ-m32-nl223-np8                                   8_738.19       765.45     9_503.65       0.3782          1.0617            1.0630         3.96
IVFOPQ-m64-nl223-np8                                  12_553.87     1_128.62    13_682.50       0.4880          1.0384            1.0371         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_919.98       801.29    10_721.26       0.3780          1.0619            1.0630         6.45
IVFOPQ-m32-nl223-np11                                  8_738.19       905.56     9_643.75       0.3782          1.0617            1.0630         3.96
IVFOPQ-m64-nl223-np11                                 12_553.87     1_385.71    13_939.59       0.4880          1.0384            1.0371         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_919.98       937.10    10_857.07       0.3782          1.0617            1.0630         6.45
IVFOPQ-m32-nl223-np14                                  8_738.19     1_041.56     9_779.75       0.3782          1.0617            1.0630         3.96
IVFOPQ-m64-nl223-np14                                 12_553.87     1_638.41    14_192.28       0.4880          1.0384            1.0371         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_919.98     1_083.51    11_003.49       0.3782          1.0617            1.0630         6.45
IVFOPQ-m32-nl316-np1                                   8_703.22       442.52     9_145.74       0.3579          1.0669            1.0661         4.65
IVFOPQ-m64-nl316-np1                                  12_998.22       481.07    13_479.30       0.4341          1.0484            1.0450         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_310.56       441.88    10_752.44       0.3374          1.1218            1.0693         7.13
IVFOPQ-m32-nl316-np2                                   8_703.22       481.51     9_184.73       0.3783          1.0603            1.0617         4.65
IVFOPQ-m64-nl316-np2                                  12_998.22       572.61    13_570.83       0.4757          1.0395            1.0385         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_310.56       500.58    10_811.14       0.3760          1.0622            1.0634         7.13
IVFOPQ-m32-nl316-np4                                   8_703.22       575.69     9_278.91       0.3849          1.0587            1.0602         4.65
IVFOPQ-m64-nl316-np4                                  12_998.22       755.19    13_753.41       0.4920          1.0370            1.0360         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_310.56       600.64    10_911.20       0.3818          1.0602            1.0613         7.13
IVFOPQ-m32-nl316-np8                                   8_703.22       766.88     9_470.10       0.3859          1.0585            1.0600         4.65
IVFOPQ-m64-nl316-np8                                  12_998.22     1_104.86    14_103.08       0.4956          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_310.56       794.64    11_105.20       0.3852          1.0589            1.0602         7.13
IVFOPQ-m32-nl316-np15                                  8_703.22     1_087.12     9_790.34       0.3859          1.0585            1.0600         4.65
IVFOPQ-m64-nl316-np15                                 12_998.22     1_730.96    14_729.18       0.4958          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_310.56     1_186.97    11_497.53       0.3859          1.0585            1.0600         7.13
IVFOPQ-m32-nl316-np17                                  8_703.22     1_178.92     9_882.14       0.3859          1.0585            1.0600         4.65
IVFOPQ-m64-nl316-np17                                 12_998.22     1_892.08    14_890.30       0.4958          1.0365            1.0355         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_310.56     1_240.69    11_551.25       0.3859          1.0585            1.0600         7.13
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
Exhaustive (query)                                        67.55     1_207.56     1_275.11       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_099.99       468.95    10_568.94       0.3329          1.1482            1.0715         6.32
SOAROPQ-near-np2                                      10_099.99       512.09    10_612.08       0.3682          1.0663            1.0672         6.32
SOAROPQ-near-np4                                      10_099.99       609.36    10_709.36       0.3688          1.0656            1.0670         6.32
SOAROPQ-near-np7                                      10_099.99       752.03    10_852.02       0.3688          1.0656            1.0670         6.32
SOAROPQ-near-np8                                      10_099.99       801.78    10_901.77       0.3688          1.0656            1.0670         6.32
SOAROPQ-near-np12                                     10_099.99       999.21    11_099.20       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.3-np1                                  10_326.00       464.45    10_790.46       0.3323          1.1492            1.0715         6.32
SOAROPQ-shift0.3-np2                                  10_326.00       518.36    10_844.36       0.3681          1.0665            1.0672         6.32
SOAROPQ-shift0.3-np4                                  10_326.00       605.91    10_931.91       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.3-np7                                  10_326.00       748.87    11_074.87       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.3-np8                                  10_326.00       803.21    11_129.21       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.3-np12                                 10_326.00       982.58    11_308.59       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.7-np1                                  10_093.33       466.12    10_559.44       0.3322          1.1502            1.0715         6.32
SOAROPQ-shift0.7-np2                                  10_093.33       520.99    10_614.32       0.3681          1.0666            1.0672         6.32
SOAROPQ-shift0.7-np4                                  10_093.33       614.80    10_708.13       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.7-np7                                  10_093.33       754.95    10_848.28       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.7-np8                                  10_093.33       795.88    10_889.21       0.3688          1.0656            1.0670         6.32
SOAROPQ-shift0.7-np12                                 10_093.33       995.36    11_088.69       0.3688          1.0656            1.0670         6.32
SOAROPQ-orth1-np1                                     10_112.93       477.04    10_589.96       0.3334          1.1478            1.0714         6.32
SOAROPQ-orth1-np2                                     10_112.93       514.70    10_627.62       0.3683          1.0661            1.0672         6.32
SOAROPQ-orth1-np4                                     10_112.93       607.37    10_720.30       0.3688          1.0656            1.0670         6.32
SOAROPQ-orth1-np7                                     10_112.93       755.48    10_868.41       0.3688          1.0656            1.0670         6.32
SOAROPQ-orth1-np8                                     10_112.93       804.37    10_917.30       0.3688          1.0656            1.0670         6.32
SOAROPQ-orth1-np12                                    10_112.93       996.43    11_109.36       0.3688          1.0656            1.0670         6.32
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
Exhaustive (query)                                        67.67     1_277.06     1_344.73       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_778.07       460.04     8_238.11       0.6644          1.0303            1.0250         3.49
IVFOPQ-m64-nl111-np1                                  11_545.26       539.70    12_084.96       0.7567          1.0171            1.0113         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         8_975.33       463.93     9_439.26       0.6652          1.0333            1.0253         5.98
IVFOPQ-m32-nl111-np2                                   7_778.07       498.34     8_276.41       0.6771          1.0261            1.0246         3.49
IVFOPQ-m64-nl111-np2                                  11_545.26       656.92    12_202.18       0.7721          1.0126            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         8_975.33       526.57     9_501.91       0.6772          1.0262            1.0246         5.98
IVFOPQ-m32-nl111-np4                                   7_778.07       589.82     8_367.90       0.6779          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np4                                  11_545.26       800.40    12_345.66       0.7732          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         8_975.33       610.02     9_585.35       0.6779          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np5                                   7_778.07       634.81     8_412.89       0.6779          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np5                                  11_545.26       885.35    12_430.61       0.7732          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         8_975.33       663.43     9_638.76       0.6779          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np8                                   7_778.07       774.80     8_552.88       0.6779          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np8                                  11_545.26     1_141.78    12_687.04       0.7732          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         8_975.33       846.14     9_821.47       0.6779          1.0259            1.0245         5.98
IVFOPQ-m32-nl111-np10                                  7_778.07       874.54     8_652.61       0.6779          1.0259            1.0245         3.49
IVFOPQ-m64-nl111-np10                                 11_545.26     1_324.65    12_869.91       0.7732          1.0123            1.0110         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        8_975.33       954.85     9_930.19       0.6779          1.0259            1.0245         5.98
IVFOPQ-m32-nl158-np1                                   8_758.71       452.14     9_210.85       0.6656          1.0303            1.0243         3.84
IVFOPQ-m64-nl158-np1                                  12_834.09       541.52    13_375.61       0.7560          1.0178            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_056.54       459.73    10_516.26       0.6763          1.0276            1.0242         6.32
IVFOPQ-m32-nl158-np2                                   8_758.71       508.26     9_266.98       0.6817          1.0251            1.0236         3.84
IVFOPQ-m64-nl158-np2                                  12_834.09       625.70    13_459.79       0.7751          1.0123            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_056.54       510.72    10_567.26       0.6830          1.0248            1.0235         6.32
IVFOPQ-m32-nl158-np4                                   8_758.71       585.96     9_344.67       0.6837          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np4                                  12_834.09       793.27    13_627.36       0.7777          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_056.54       600.46    10_656.99       0.6838          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np7                                   8_758.71       719.91     9_478.62       0.6838          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np7                                  12_834.09     1_042.97    13_877.06       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_056.54       743.89    10_800.43       0.6838          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np8                                   8_758.71       770.21     9_528.92       0.6838          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np8                                  12_834.09     1_127.52    13_961.62       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_056.54       793.13    10_849.67       0.6838          1.0246            1.0234         6.32
IVFOPQ-m32-nl158-np12                                  8_758.71       950.56     9_709.28       0.6838          1.0246            1.0234         3.84
IVFOPQ-m64-nl158-np12                                 12_834.09     1_471.37    14_305.46       0.7778          1.0117            1.0106         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_056.54       980.36    11_036.90       0.6838          1.0246            1.0234         6.32
IVFOPQ-m32-nl223-np1                                   8_474.17       432.38     8_906.55       0.4957          1.0650            1.0575         3.96
IVFOPQ-m64-nl223-np1                                  12_631.63       487.41    13_119.04       0.5346          1.0542            1.0467         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_733.87       449.24    10_183.11       0.6014          1.0413            1.0349         6.45
IVFOPQ-m32-nl223-np2                                   8_474.17       491.02     8_965.19       0.6138          1.0371            1.0319         3.96
IVFOPQ-m64-nl223-np2                                  12_631.63       588.62    13_220.25       0.6808          1.0254            1.0187         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_733.87       504.47    10_238.33       0.6631          1.0282            1.0260         6.45
IVFOPQ-m32-nl223-np4                                   8_474.17       583.34     9_057.51       0.6726          1.0264            1.0245         3.96
IVFOPQ-m64-nl223-np4                                  12_631.63       771.76    13_403.38       0.7578          1.0143            1.0120         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_733.87       603.31    10_337.18       0.6866          1.0240            1.0228         6.45
IVFOPQ-m32-nl223-np8                                   8_474.17       764.19     9_238.36       0.6900          1.0235            1.0223         3.96
IVFOPQ-m64-nl223-np8                                  12_631.63     1_113.64    13_745.26       0.7800          1.0113            1.0103         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_733.87       791.23    10_525.10       0.6909          1.0233            1.0222         6.45
IVFOPQ-m32-nl223-np11                                  8_474.17       899.42     9_373.59       0.6912          1.0233            1.0222         3.96
IVFOPQ-m64-nl223-np11                                 12_631.63     1_366.18    13_997.81       0.7814          1.0111            1.0102         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_733.87       931.69    10_665.55       0.6912          1.0233            1.0222         6.45
IVFOPQ-m32-nl223-np14                                  8_474.17     1_042.46     9_516.63       0.6912          1.0233            1.0222         3.96
IVFOPQ-m64-nl223-np14                                 12_631.63     1_619.53    14_251.16       0.7814          1.0111            1.0102         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_733.87     1_082.84    10_816.70       0.6912          1.0233            1.0222         6.45
IVFOPQ-m32-nl316-np1                                   9_027.97       439.79     9_467.75       0.4098          1.0851            1.0798         4.65
IVFOPQ-m64-nl316-np1                                  13_247.14       489.70    13_736.84       0.4287          1.0751            1.0695         6.17
SOAROPQ-shift0.5-m32-nl316-np1                        10_588.13       444.03    11_032.15       0.5373          1.0529            1.0490         7.13
IVFOPQ-m32-nl316-np2                                   9_027.97       480.71     9_508.68       0.5477          1.0493            1.0457         4.65
IVFOPQ-m64-nl316-np2                                  13_247.14       570.41    13_817.54       0.5950          1.0382            1.0341         6.17
SOAROPQ-shift0.5-m32-nl316-np2                        10_588.13       501.88    11_090.01       0.6302          1.0339            1.0312         7.13
IVFOPQ-m32-nl316-np4                                   9_027.97       580.05     9_608.02       0.6428          1.0314            1.0287         4.65
IVFOPQ-m64-nl316-np4                                  13_247.14       755.98    14_003.11       0.7195          1.0197            1.0162         6.17
SOAROPQ-shift0.5-m32-nl316-np4                        10_588.13       602.41    11_190.53       0.6783          1.0253            1.0239         7.13
IVFOPQ-m32-nl316-np8                                   9_027.97       764.57     9_792.54       0.6877          1.0238            1.0224         4.65
IVFOPQ-m64-nl316-np8                                  13_247.14     1_100.16    14_347.29       0.7779          1.0118            1.0105         6.17
SOAROPQ-shift0.5-m32-nl316-np8                        10_588.13       795.69    11_383.82       0.6922          1.0231            1.0217         7.13
IVFOPQ-m32-nl316-np15                                  9_027.97     1_079.76    10_107.72       0.6934          1.0229            1.0216         4.65
IVFOPQ-m64-nl316-np15                                 13_247.14     1_690.14    14_937.28       0.7856          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np15                       10_588.13     1_118.51    11_706.63       0.6934          1.0229            1.0216         7.13
IVFOPQ-m32-nl316-np17                                  9_027.97     1_185.98    10_213.94       0.6934          1.0229            1.0216         4.65
IVFOPQ-m64-nl316-np17                                 13_247.14     1_885.14    15_132.27       0.7856          1.0108            1.0098         6.17
SOAROPQ-shift0.5-m32-nl316-np17                       10_588.13     1_221.25    11_809.38       0.6934          1.0229            1.0216         7.13
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
Exhaustive (query)                                        67.67     1_277.06     1_344.73       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_088.05       466.00    10_554.05       0.6763          1.0275            1.0242         6.32
SOAROPQ-near-np2                                      10_088.05       508.02    10_596.07       0.6832          1.0248            1.0235         6.32
SOAROPQ-near-np4                                      10_088.05       608.29    10_696.34       0.6838          1.0246            1.0234         6.32
SOAROPQ-near-np7                                      10_088.05       746.74    10_834.79       0.6838          1.0246            1.0234         6.32
SOAROPQ-near-np8                                      10_088.05       796.76    10_884.81       0.6838          1.0246            1.0234         6.32
SOAROPQ-near-np12                                     10_088.05       983.77    11_071.82       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np1                                  10_050.95       464.73    10_515.67       0.6765          1.0275            1.0242         6.32
SOAROPQ-shift0.3-np2                                  10_050.95       508.00    10_558.95       0.6831          1.0248            1.0235         6.32
SOAROPQ-shift0.3-np4                                  10_050.95       603.56    10_654.51       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np7                                  10_050.95       757.66    10_808.60       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np8                                  10_050.95       787.37    10_838.31       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.3-np12                                 10_050.95       983.38    11_034.32       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np1                                  10_199.76       468.44    10_668.20       0.6760          1.0277            1.0243         6.32
SOAROPQ-shift0.7-np2                                  10_199.76       511.82    10_711.57       0.6829          1.0249            1.0236         6.32
SOAROPQ-shift0.7-np4                                  10_199.76       602.38    10_802.13       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np7                                  10_199.76       739.50    10_939.25       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np8                                  10_199.76       793.61    10_993.36       0.6838          1.0246            1.0234         6.32
SOAROPQ-shift0.7-np12                                 10_199.76       982.87    11_182.63       0.6838          1.0246            1.0234         6.32
SOAROPQ-orth1-np1                                     10_343.41       466.52    10_809.93       0.6758          1.0278            1.0242         6.32
SOAROPQ-orth1-np2                                     10_343.41       511.34    10_854.75       0.6829          1.0249            1.0236         6.32
SOAROPQ-orth1-np4                                     10_343.41       603.83    10_947.24       0.6838          1.0246            1.0234         6.32
SOAROPQ-orth1-np7                                     10_343.41       754.36    11_097.77       0.6838          1.0246            1.0234         6.32
SOAROPQ-orth1-np8                                     10_343.41       800.01    11_143.42       0.6838          1.0246            1.0234         6.32
SOAROPQ-orth1-np12                                    10_343.41       985.66    11_329.06       0.6838          1.0246            1.0234         6.32
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
Exhaustive (query)                                        74.15     1_281.18     1_355.33       1.0000          1.0000            1.0000        97.66
IVFOPQ-m32-nl111-np1                                   7_890.01       426.60     8_316.60       0.7216          1.1712            1.0832         3.49
IVFOPQ-m64-nl111-np1                                  11_736.10       489.27    12_225.37       0.7255          1.1681            1.0809         5.02
SOAROPQ-shift0.5-m32-nl111-np1                         9_046.26       453.68     9_499.94       0.8436          1.0561            1.0277         5.98
IVFOPQ-m32-nl111-np2                                   7_890.01       486.91     8_376.92       0.8515          1.0485            1.0239         3.49
IVFOPQ-m64-nl111-np2                                  11_736.10       610.82    12_346.92       0.8597          1.0453            1.0198         5.02
SOAROPQ-shift0.5-m32-nl111-np2                         9_046.26       540.92     9_587.18       0.8854          1.0261            1.0173         5.98
IVFOPQ-m32-nl111-np4                                   7_890.01       608.81     8_498.82       0.8900          1.0220            1.0159         3.49
IVFOPQ-m64-nl111-np4                                  11_736.10       857.99    12_594.09       0.9006          1.0185            1.0126         5.02
SOAROPQ-shift0.5-m32-nl111-np4                         9_046.26       698.61     9_744.87       0.8923          1.0213            1.0156         5.98
IVFOPQ-m32-nl111-np5                                   7_890.01       675.22     8_565.23       0.8919          1.0209            1.0155         3.49
IVFOPQ-m64-nl111-np5                                  11_736.10       989.00    12_725.10       0.9028          1.0174            1.0121         5.02
SOAROPQ-shift0.5-m32-nl111-np5                         9_046.26       778.84     9_825.10       0.8927          1.0209            1.0155         5.98
IVFOPQ-m32-nl111-np8                                   7_890.01       852.54     8_742.55       0.8932          1.0202            1.0153         3.49
IVFOPQ-m64-nl111-np8                                  11_736.10     1_367.42    13_103.53       0.9042          1.0167            1.0118         5.02
SOAROPQ-shift0.5-m32-nl111-np8                         9_046.26       983.87    10_030.13       0.8931          1.0205            1.0153         5.98
IVFOPQ-m32-nl111-np10                                  7_890.01       974.54     8_864.55       0.8933          1.0202            1.0152         3.49
IVFOPQ-m64-nl111-np10                                 11_736.10     1_607.54    13_343.64       0.9042          1.0167            1.0118         5.02
SOAROPQ-shift0.5-m32-nl111-np10                        9_046.26     1_129.95    10_176.22       0.8932          1.0203            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   9_135.17       432.02     9_567.18       0.7234          1.1670            1.0836         3.84
IVFOPQ-m64-nl158-np1                                  13_056.07       475.14    13_531.20       0.7265          1.1644            1.0812         5.36
SOAROPQ-shift0.5-m32-nl158-np1                        10_302.23       445.99    10_748.22       0.8467          1.0546            1.0256         6.32
IVFOPQ-m32-nl158-np2                                   9_135.17       476.35     9_611.52       0.8571          1.0452            1.0213         3.84
IVFOPQ-m64-nl158-np2                                  13_056.07       584.90    13_640.97       0.8632          1.0426            1.0180         5.36
SOAROPQ-shift0.5-m32-nl158-np2                        10_302.23       511.21    10_813.44       0.8916          1.0239            1.0151         6.32
IVFOPQ-m32-nl158-np4                                   9_135.17       621.67     9_756.83       0.8972          1.0195            1.0134         3.84
IVFOPQ-m64-nl158-np4                                  13_056.07       796.64    13_852.70       0.9048          1.0170            1.0110         5.36
SOAROPQ-shift0.5-m32-nl158-np4                        10_302.23       653.09    10_955.32       0.9004          1.0186            1.0130         6.32
IVFOPQ-m32-nl158-np7                                   9_135.17       754.59     9_889.75       0.9020          1.0171            1.0125         3.84
IVFOPQ-m64-nl158-np7                                  13_056.07     1_124.67    14_180.74       0.9096          1.0145            1.0102         5.36
SOAROPQ-shift0.5-m32-nl158-np7                        10_302.23       846.20    11_148.43       0.9021          1.0173            1.0126         6.32
IVFOPQ-m32-nl158-np8                                   9_135.17       825.88     9_961.05       0.9024          1.0169            1.0125         3.84
IVFOPQ-m64-nl158-np8                                  13_056.07     1_238.88    14_294.94       0.9100          1.0144            1.0101         5.36
SOAROPQ-shift0.5-m32-nl158-np8                        10_302.23       907.39    11_209.63       0.9023          1.0172            1.0125         6.32
IVFOPQ-m32-nl158-np12                                  9_135.17     1_037.68    10_172.85       0.9026          1.0169            1.0124         3.84
IVFOPQ-m64-nl158-np12                                 13_056.07     1_668.67    14_724.74       0.9102          1.0143            1.0100         5.36
SOAROPQ-shift0.5-m32-nl158-np12                       10_302.23     1_172.00    11_474.23       0.9025          1.0169            1.0124         6.32
IVFOPQ-m32-nl223-np1                                   8_347.87       424.13     8_772.00       0.6955          1.1880            1.1139         3.96
IVFOPQ-m64-nl223-np1                                  12_525.29       468.87    12_994.16       0.6983          1.1856            1.1114         5.49
SOAROPQ-shift0.5-m32-nl223-np1                         9_759.73       434.34    10_194.07       0.8331          1.0653            1.0319         6.45
IVFOPQ-m32-nl223-np2                                   8_347.87       473.41     8_821.28       0.8441          1.0543            1.0260         3.96
IVFOPQ-m64-nl223-np2                                  12_525.29       568.45    13_093.74       0.8506          1.0516            1.0224         5.49
SOAROPQ-shift0.5-m32-nl223-np2                         9_759.73       495.05    10_254.79       0.8917          1.0253            1.0151         6.45
IVFOPQ-m32-nl223-np4                                   8_347.87       572.65     8_920.52       0.8994          1.0194            1.0126         3.96
IVFOPQ-m64-nl223-np4                                  12_525.29       756.33    13_281.62       0.9080          1.0165            1.0099         5.49
SOAROPQ-shift0.5-m32-nl223-np4                         9_759.73       616.33    10_376.06       0.9047          1.0178            1.0117         6.45
IVFOPQ-m32-nl223-np8                                   8_347.87       774.33     9_122.21       0.9071          1.0156            1.0110         3.96
IVFOPQ-m64-nl223-np8                                  12_525.29     1_144.52    13_669.80       0.9164          1.0125            1.0084         5.49
SOAROPQ-shift0.5-m32-nl223-np8                         9_759.73       868.32    10_628.05       0.9071          1.0159            1.0111         6.45
IVFOPQ-m32-nl223-np11                                  8_347.87       926.75     9_274.62       0.9075          1.0154            1.0109         3.96
IVFOPQ-m64-nl223-np11                                 12_525.29     1_434.34    13_959.63       0.9168          1.0124            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np11                        9_759.73     1_019.69    10_779.42       0.9075          1.0156            1.0109         6.45
IVFOPQ-m32-nl223-np14                                  8_347.87     1_075.55     9_423.42       0.9075          1.0154            1.0109         3.96
IVFOPQ-m64-nl223-np14                                 12_525.29     1_725.57    14_250.86       0.9169          1.0123            1.0083         5.49
SOAROPQ-shift0.5-m32-nl223-np14                        9_759.73     1_194.83    10_954.56       0.9075          1.0154            1.0109         6.45
IVFOPQ-m32-nl316-np1                                   8_784.70       427.26     9_211.96       0.6783          1.2033            1.1315         4.65
IVFOPQ-m64-nl316-np1                                  12_905.06       469.64    13_374.70       0.6798          1.2013            1.1297         6.17
SOAROPQ-shift0.5-m32-nl316-np1                         9_913.11       435.12    10_348.24       0.8246          1.0715            1.0366         7.13
IVFOPQ-m32-nl316-np2                                   8_784.70       484.12     9_268.82       0.8377          1.0586            1.0283         4.65
IVFOPQ-m64-nl316-np2                                  12_905.06       564.18    13_469.24       0.8418          1.0567            1.0253         6.17
SOAROPQ-shift0.5-m32-nl316-np2                         9_913.11       490.13    10_403.24       0.8939          1.0255            1.0140         7.13
IVFOPQ-m32-nl316-np4                                   8_784.70       568.82     9_353.52       0.9032          1.0184            1.0113         4.65
IVFOPQ-m64-nl316-np4                                  12_905.06       738.46    13_643.52       0.9089          1.0165            1.0095         6.17
SOAROPQ-shift0.5-m32-nl316-np4                         9_913.11       598.68    10_511.79       0.9102          1.0163            1.0103         7.13
IVFOPQ-m32-nl316-np8                                   8_784.70       767.72     9_552.42       0.9137          1.0133            1.0093         4.65
IVFOPQ-m64-nl316-np8                                  12_905.06     1_097.39    14_002.45       0.9199          1.0113            1.0076         6.17
SOAROPQ-shift0.5-m32-nl316-np8                         9_913.11       814.21    10_727.32       0.9136          1.0139            1.0094         7.13
IVFOPQ-m32-nl316-np15                                  8_784.70     1_097.15     9_881.85       0.9145          1.0130            1.0092         4.65
IVFOPQ-m64-nl316-np15                                 12_905.06     1_746.52    14_651.58       0.9208          1.0110            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np15                        9_913.11     1_246.15    11_159.26       0.9144          1.0131            1.0092         7.13
IVFOPQ-m32-nl316-np17                                  8_784.70     1_201.84     9_986.54       0.9145          1.0130            1.0092         4.65
IVFOPQ-m64-nl316-np17                                 12_905.06     1_961.14    14_866.20       0.9208          1.0110            1.0075         6.17
SOAROPQ-shift0.5-m32-nl316-np17                        9_913.11     1_304.42    11_217.53       0.9145          1.0130            1.0092         7.13
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
Exhaustive (query)                                        74.15     1_281.18     1_355.33       1.0000          1.0000            1.0000        97.66
SOAROPQ-near-np1                                      10_490.32       449.55    10_939.87       0.8452          1.0573            1.0241         6.32
SOAROPQ-near-np2                                      10_490.32       516.17    11_006.49       0.8917          1.0237            1.0148         6.32
SOAROPQ-near-np4                                      10_490.32       648.96    11_139.28       0.9010          1.0181            1.0129         6.32
SOAROPQ-near-np7                                      10_490.32       856.67    11_346.99       0.9023          1.0171            1.0125         6.32
SOAROPQ-near-np8                                      10_490.32       945.30    11_435.62       0.9024          1.0170            1.0125         6.32
SOAROPQ-near-np12                                     10_490.32     1_237.64    11_727.96       0.9026          1.0169            1.0124         6.32
SOAROPQ-shift0.3-np1                                  11_335.60       448.30    11_783.90       0.8485          1.0533            1.0247         6.32
SOAROPQ-shift0.3-np2                                  11_335.60       516.35    11_851.95       0.8922          1.0234            1.0148         6.32
SOAROPQ-shift0.3-np4                                  11_335.60       649.28    11_984.88       0.9007          1.0183            1.0130         6.32
SOAROPQ-shift0.3-np7                                  11_335.60       855.67    12_191.27       0.9022          1.0172            1.0126         6.32
SOAROPQ-shift0.3-np8                                  11_335.60       927.14    12_262.74       0.9024          1.0171            1.0125         6.32
SOAROPQ-shift0.3-np12                                 11_335.60     1_156.48    12_492.08       0.9025          1.0169            1.0124         6.32
SOAROPQ-shift0.7-np1                                  10_361.53       450.99    10_812.52       0.8435          1.0574            1.0265         6.32
SOAROPQ-shift0.7-np2                                  10_361.53       512.58    10_874.11       0.8909          1.0247            1.0153         6.32
SOAROPQ-shift0.7-np4                                  10_361.53       651.41    11_012.94       0.9001          1.0189            1.0131         6.32
SOAROPQ-shift0.7-np7                                  10_361.53       845.06    11_206.59       0.9021          1.0174            1.0126         6.32
SOAROPQ-shift0.7-np8                                  10_361.53       909.82    11_271.35       0.9023          1.0172            1.0125         6.32
SOAROPQ-shift0.7-np12                                 10_361.53     1_253.86    11_615.39       0.9025          1.0169            1.0124         6.32
SOAROPQ-orth1-np1                                     10_398.16       445.66    10_843.82       0.8460          1.0560            1.0248         6.32
SOAROPQ-orth1-np2                                     10_398.16       520.15    10_918.31       0.8917          1.0239            1.0149         6.32
SOAROPQ-orth1-np4                                     10_398.16       653.57    11_051.73       0.9006          1.0184            1.0130         6.32
SOAROPQ-orth1-np7                                     10_398.16       859.58    11_257.74       0.9022          1.0172            1.0125         6.32
SOAROPQ-orth1-np8                                     10_398.16       912.23    11_310.39       0.9024          1.0171            1.0125         6.32
SOAROPQ-orth1-np12                                    10_398.16     1_172.25    11_570.41       0.9025          1.0169            1.0124         6.32
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
Exhaustive (query)                                        74.19     1_296.75     1_370.94       1.0000          1.0000            1.0000        97.85
IVFOPQ-m32-nl111-np1                                   7_665.71       423.64     8_089.35       0.7779          1.1326            1.0517         3.49
IVFOPQ-m64-nl111-np1                                  11_745.56       480.49    12_226.04       0.7817          1.1296            1.0467         5.02
SOAROPQ-orth1-m32-nl111-np1                            8_874.39       453.43     9_327.82       0.8616          1.0578            1.0267         5.98
IVFOPQ-m32-nl111-np2                                   7_665.71       480.37     8_146.08       0.8790          1.0367            1.0199         3.49
IVFOPQ-m64-nl111-np2                                  11_745.56       595.11    12_340.67       0.8858          1.0334            1.0167         5.02
SOAROPQ-orth1-m32-nl111-np2                            8_874.39       522.21     9_396.60       0.8926          1.0300            1.0180         5.98
IVFOPQ-m32-nl111-np4                                   7_665.71       597.47     8_263.18       0.8990          1.0225            1.0156         3.49
IVFOPQ-m64-nl111-np4                                  11_745.56       844.50    12_590.06       0.9071          1.0192            1.0127         5.02
SOAROPQ-orth1-m32-nl111-np4                            8_874.39       674.56     9_548.95       0.8988          1.0242            1.0160         5.98
IVFOPQ-m32-nl111-np5                                   7_665.71       659.49     8_325.19       0.9000          1.0220            1.0153         3.49
IVFOPQ-m64-nl111-np5                                  11_745.56       941.32    12_686.88       0.9082          1.0187            1.0125         5.02
SOAROPQ-orth1-m32-nl111-np5                            8_874.39       739.78     9_614.17       0.8995          1.0234            1.0157         5.98
IVFOPQ-m32-nl111-np8                                   7_665.71       829.99     8_495.70       0.9005          1.0218            1.0152         3.49
IVFOPQ-m64-nl111-np8                                  11_745.56     1_283.81    13_029.37       0.9088          1.0185            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np8                            8_874.39       941.96     9_816.35       0.9003          1.0222            1.0153         5.98
IVFOPQ-m32-nl111-np10                                  7_665.71       944.44     8_610.14       0.9005          1.0217            1.0152         3.49
IVFOPQ-m64-nl111-np10                                 11_745.56     1_511.01    13_256.57       0.9088          1.0184            1.0123         5.02
SOAROPQ-orth1-m32-nl111-np10                           8_874.39     1_066.70     9_941.09       0.9005          1.0220            1.0153         5.98
IVFOPQ-m32-nl158-np1                                   9_121.15       419.80     9_540.95       0.7594          1.1510            1.0684         3.84
IVFOPQ-m64-nl158-np1                                  13_081.83       469.78    13_551.61       0.7621          1.1486            1.0644         5.36
SOAROPQ-orth1-m32-nl158-np1                           10_228.11       438.40    10_666.52       0.8568          1.0634            1.0270         6.32
IVFOPQ-m32-nl158-np2                                   9_121.15       475.49     9_596.64       0.8773          1.0396            1.0193         3.84
IVFOPQ-m64-nl158-np2                                  13_081.83       572.21    13_654.04       0.8816          1.0375            1.0169         5.36
SOAROPQ-orth1-m32-nl158-np2                           10_228.11       501.63    10_729.74       0.8979          1.0286            1.0157         6.32
IVFOPQ-m32-nl158-np4                                   9_121.15       577.95     9_699.10       0.9070          1.0189            1.0131         3.84
IVFOPQ-m64-nl158-np4                                  13_081.83       788.74    13_870.57       0.9125          1.0166            1.0111         5.36
SOAROPQ-orth1-m32-nl158-np4                           10_228.11       628.70    10_856.81       0.9073          1.0207            1.0133         6.32
IVFOPQ-m32-nl158-np7                                   9_121.15       738.95     9_860.10       0.9093          1.0176            1.0126         3.84
IVFOPQ-m64-nl158-np7                                  13_081.83     1_091.83    14_173.65       0.9151          1.0153            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np7                           10_228.11       821.55    11_049.67       0.9090          1.0184            1.0127         6.32
IVFOPQ-m32-nl158-np8                                   9_121.15       791.68     9_912.84       0.9094          1.0176            1.0126         3.84
IVFOPQ-m64-nl158-np8                                  13_081.83     1_191.96    14_273.78       0.9153          1.0152            1.0106         5.36
SOAROPQ-orth1-m32-nl158-np8                           10_228.11       879.25    11_107.36       0.9092          1.0182            1.0127         6.32
IVFOPQ-m32-nl158-np12                                  9_121.15     1_007.93    10_129.08       0.9095          1.0176            1.0125         3.84
IVFOPQ-m64-nl158-np12                                 13_081.83     1_629.95    14_711.78       0.9154          1.0152            1.0105         5.36
SOAROPQ-orth1-m32-nl158-np12                          10_228.11     1_114.87    11_342.98       0.9094          1.0177            1.0126         6.32
IVFOPQ-m32-nl223-np1                                   8_466.88       427.38     8_894.25       0.7330          1.1767            1.0999         3.96
IVFOPQ-m64-nl223-np1                                  12_605.91       464.55    13_070.46       0.7347          1.1749            1.0978         5.49
SOAROPQ-orth1-m32-nl223-np1                            9_706.90       440.53    10_147.43       0.8494          1.0687            1.0294         6.45
IVFOPQ-m32-nl223-np2                                   8_466.88       470.05     8_936.92       0.8709          1.0451            1.0207         3.96
IVFOPQ-m64-nl223-np2                                  12_605.91       557.60    13_163.51       0.8741          1.0434            1.0185         5.49
SOAROPQ-orth1-m32-nl223-np2                            9_706.90       491.85    10_198.75       0.9000          1.0280            1.0149         6.45
IVFOPQ-m32-nl223-np4                                   8_466.88       574.66     9_041.54       0.9099          1.0179            1.0120         3.96
IVFOPQ-m64-nl223-np4                                  12_605.91       747.38    13_353.28       0.9149          1.0162            1.0103         5.49
SOAROPQ-orth1-m32-nl223-np4                            9_706.90       606.14    10_313.05       0.9108          1.0198            1.0121         6.45
IVFOPQ-m32-nl223-np8                                   8_466.88       764.21     9_231.09       0.9134          1.0162            1.0113         3.96
IVFOPQ-m64-nl223-np8                                  12_605.91     1_125.03    13_730.94       0.9185          1.0144            1.0095         5.49
SOAROPQ-orth1-m32-nl223-np8                            9_706.90       822.10    10_529.00       0.9132          1.0168            1.0114         6.45
IVFOPQ-m32-nl223-np11                                  8_466.88       910.16     9_377.04       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np11                                 12_605.91     1_403.28    14_009.19       0.9187          1.0144            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np11                           9_706.90       999.73    10_706.63       0.9135          1.0163            1.0113         6.45
IVFOPQ-m32-nl223-np14                                  8_466.88     1_055.51     9_522.38       0.9136          1.0161            1.0112         3.96
IVFOPQ-m64-nl223-np14                                 12_605.91     1_687.31    14_293.22       0.9188          1.0143            1.0094         5.49
SOAROPQ-orth1-m32-nl223-np14                           9_706.90     1_157.58    10_864.49       0.9135          1.0162            1.0112         6.45
IVFOPQ-m32-nl316-np1                                   8_816.51       433.24     9_249.75       0.7080          1.2039            1.1272         4.65
IVFOPQ-m64-nl316-np1                                  12_957.93       469.04    13_426.97       0.7097          1.2018            1.1246         6.17
SOAROPQ-orth1-m32-nl316-np1                           10_248.63       436.76    10_685.39       0.8347          1.0799            1.0371         7.13
IVFOPQ-m32-nl316-np2                                   8_816.51       474.78     9_291.29       0.8590          1.0536            1.0252         4.65
IVFOPQ-m64-nl316-np2                                  12_957.93       558.38    13_516.31       0.8632          1.0515            1.0223         6.17
SOAROPQ-orth1-m32-nl316-np2                           10_248.63       486.28    10_734.91       0.8984          1.0293            1.0154         7.13
IVFOPQ-m32-nl316-np4                                   8_816.51       568.26     9_384.78       0.9114          1.0177            1.0112         4.65
IVFOPQ-m64-nl316-np4                                  12_957.93       735.08    13_693.01       0.9173          1.0157            1.0093         6.17
SOAROPQ-orth1-m32-nl316-np4                           10_248.63       593.61    10_842.24       0.9135          1.0191            1.0112         7.13
IVFOPQ-m32-nl316-np8                                   8_816.51       752.28     9_568.80       0.9170          1.0148            1.0101         4.65
IVFOPQ-m64-nl316-np8                                  12_957.93     1_088.97    14_046.90       0.9234          1.0128            1.0082         6.17
SOAROPQ-orth1-m32-nl316-np8                           10_248.63       803.18    11_051.81       0.9166          1.0159            1.0103         7.13
IVFOPQ-m32-nl316-np15                                  8_816.51     1_083.33     9_899.84       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np15                                 12_957.93     1_719.18    14_677.11       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np15                          10_248.63     1_232.23    11_480.86       0.9173          1.0148            1.0100         7.13
IVFOPQ-m32-nl316-np17                                  8_816.51     1_189.60    10_006.11       0.9174          1.0147            1.0100         4.65
IVFOPQ-m64-nl316-np17                                 12_957.93     1_911.48    14_869.41       0.9239          1.0127            1.0081         6.17
SOAROPQ-orth1-m32-nl316-np17                          10_248.63     1_284.77    11_533.40       0.9174          1.0147            1.0100         7.13
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
Exhaustive (query)                                        74.19     1_296.75     1_370.94       1.0000          1.0000            1.0000        97.85
SOAROPQ-near-np1                                      10_176.59       435.68    10_612.27       0.8631          1.0549            1.0252         6.32
SOAROPQ-near-np2                                      10_176.59       501.27    10_677.86       0.8998          1.0253            1.0153         6.32
SOAROPQ-near-np4                                      10_176.59       630.51    10_807.10       0.9078          1.0194            1.0131         6.32
SOAROPQ-near-np7                                      10_176.59       816.63    10_993.22       0.9092          1.0179            1.0127         6.32
SOAROPQ-near-np8                                      10_176.59       879.72    11_056.31       0.9093          1.0178            1.0126         6.32
SOAROPQ-near-np12                                     10_176.59     1_125.37    11_301.96       0.9095          1.0176            1.0126         6.32
SOAROPQ-shift0.3-np1                                  10_160.20       446.39    10_606.59       0.8624          1.0567            1.0262         6.32
SOAROPQ-shift0.3-np2                                  10_160.20       501.91    10_662.11       0.8984          1.0272            1.0156         6.32
SOAROPQ-shift0.3-np4                                  10_160.20       626.20    10_786.40       0.9069          1.0204            1.0134         6.32
SOAROPQ-shift0.3-np7                                  10_160.20       847.07    11_007.27       0.9089          1.0183            1.0127         6.32
SOAROPQ-shift0.3-np8                                  10_160.20       875.89    11_036.09       0.9091          1.0181            1.0127         6.32
SOAROPQ-shift0.3-np12                                 10_160.20     1_118.03    11_278.23       0.9094          1.0177            1.0126         6.32
SOAROPQ-shift0.7-np1                                  10_182.21       442.42    10_624.63       0.8569          1.0638            1.0280         6.32
SOAROPQ-shift0.7-np2                                  10_182.21       501.57    10_683.78       0.8955          1.0311            1.0165         6.32
SOAROPQ-shift0.7-np4                                  10_182.21       632.62    10_814.83       0.9057          1.0223            1.0137         6.32
SOAROPQ-shift0.7-np7                                  10_182.21       831.69    11_013.90       0.9085          1.0190            1.0128         6.32
SOAROPQ-shift0.7-np8                                  10_182.21       881.04    11_063.25       0.9088          1.0187            1.0128         6.32
SOAROPQ-shift0.7-np12                                 10_182.21     1_130.54    11_312.76       0.9094          1.0178            1.0126         6.32
SOAROPQ-orth1-np1                                     10_217.65       439.49    10_657.14       0.8568          1.0634            1.0270         6.32
SOAROPQ-orth1-np2                                     10_217.65       502.56    10_720.21       0.8979          1.0286            1.0157         6.32
SOAROPQ-orth1-np4                                     10_217.65       627.88    10_845.53       0.9073          1.0207            1.0133         6.32
SOAROPQ-orth1-np7                                     10_217.65       822.46    11_040.11       0.9090          1.0184            1.0127         6.32
SOAROPQ-orth1-np8                                     10_217.65       886.66    11_104.31       0.9092          1.0182            1.0127         6.32
SOAROPQ-orth1-np12                                    10_217.65     1_117.69    11_335.34       0.9094          1.0177            1.0126         6.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
