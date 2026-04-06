## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown for the GPU-accelerated code. If you wish to run
the version with GPU-accelerated exhaustive and IVF script, please use:

```bash
cargo run --example gridsearch_gpu --features gpu --release
```

For the CAGRA style search, use:

```bash
cargo run --example gridsearch_cagra --features gpu --release
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided (however, GPU memory is not
reported). To note also, every benchmark here is run on the wgpu backend.
Other backends like cuda might provide even more speed benefits.

## Table of Contents

- [GPU exhaustive and IVF](#gpu-accelerated-exhaustive-and-ivf-vs-cpu-exhaustive)
- [Comparison on larger data sets against the CPU](#comparison-against-ivf-cpu)
- [CAGRA style index](#cagra-type-querying)
- [CAGRA index on larger data](#larger-data-sets)
- [CAGRA for kNN generation](#two-tier-knn-generation)

### GPU-accelerated exhaustive and IVF vs CPU exhaustive

The GPU acceleration is particularly notable for the exhaustive index. The
IVF-GPU reaches very fast speeds here, but not much faster actually than the
IVF-CPU version (or exhaustive GPU index). The advantages for the IVF-GPU index
become more apparent in larger data sets (more to that below). Also to note is
that the data is kept on the GPU for easier access and less frequent transfer
between CPU and GPU, hence, the apparent reduced memory footprint. The data
lives on the GPU for this version. (Be aware of your VRAM limits!).

<details>
<summary><b>GPU - Euclidean (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.80     1_791.21     1_795.01       1.0000     0.000000        18.31
Exhaustive (self)                                          3.80    18_470.35    18_474.16       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     6.50       667.32       673.82       1.0000     0.000005        18.31
GPU-Exhaustive (self)                                      6.50     5_541.03     5_547.52       1.0000     0.000005        18.31
IVF-GPU-nl273-np13 (query)                               415.69       439.40       855.09       0.9942     0.017444         1.15
IVF-GPU-nl273-np16 (query)                               415.69       347.30       762.99       0.9987     0.002855         1.15
IVF-GPU-nl273-np23 (query)                               415.69       436.81       852.50       1.0000     0.000005         1.15
IVF-GPU-nl273 (self)                                     415.69     1_591.85     2_007.55       1.0000     0.000005         1.15
IVF-GPU-nl387-np19 (query)                               827.52       290.97     1_118.49       0.9976     0.012096         1.15
IVF-GPU-nl387-np27 (query)                               827.52       414.30     1_241.82       1.0000     0.000005         1.15
IVF-GPU-nl387 (self)                                     827.52     1_468.08     2_295.60       1.0000     0.000005         1.15
IVF-GPU-nl547-np23 (query)                             1_643.80       305.17     1_948.97       0.9958     0.015835         1.15
IVF-GPU-nl547-np27 (query)                             1_643.80       341.03     1_984.83       0.9991     0.002102         1.15
IVF-GPU-nl547-np33 (query)                             1_643.80       342.78     1_986.58       0.9999     0.000146         1.15
IVF-GPU-nl547 (self)                                   1_643.80     1_393.69     3_037.48       0.9999     0.000153         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.06     1_860.57     1_864.63       1.0000     0.000000        18.88
Exhaustive (self)                                          4.06    18_632.97    18_637.04       1.0000     0.000000        18.88
GPU-Exhaustive (query)                                     6.66       663.55       670.21       1.0000     0.000000        18.88
GPU-Exhaustive (self)                                      6.66     5_620.87     5_627.53       1.0000     0.000000        18.88
IVF-GPU-nl273-np13 (query)                               405.87       297.73       703.60       0.9946     0.000011         1.15
IVF-GPU-nl273-np16 (query)                               405.87       353.59       759.46       0.9989     0.000002         1.15
IVF-GPU-nl273-np23 (query)                               405.87       430.81       836.68       1.0000     0.000000         1.15
IVF-GPU-nl273 (self)                                     405.87     1_739.60     2_145.47       1.0000     0.000000         1.15
IVF-GPU-nl387-np19 (query)                               798.26       310.78     1_109.04       0.9979     0.000007         1.15
IVF-GPU-nl387-np27 (query)                               798.26       405.70     1_203.96       1.0000     0.000000         1.15
IVF-GPU-nl387 (self)                                     798.26     1_409.94     2_208.20       1.0000     0.000000         1.15
IVF-GPU-nl547-np23 (query)                             1_517.79       312.87     1_830.65       0.9957     0.000012         1.15
IVF-GPU-nl547-np27 (query)                             1_517.79       334.90     1_852.69       0.9991     0.000002         1.15
IVF-GPU-nl547-np33 (query)                             1_517.79       343.09     1_860.88       0.9999     0.000000         1.15
IVF-GPU-nl547 (self)                                   1_517.79     1_550.20     3_067.99       0.9999     0.000000         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.77     1_825.44     1_829.20       1.0000     0.000000        18.31
Exhaustive (self)                                          3.77    18_879.73    18_883.50       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.27       648.45       653.72       1.0000     0.000000        18.31
GPU-Exhaustive (self)                                      5.27     5_558.20     5_563.47       1.0000     0.000000        18.31
IVF-GPU-nl273-np13 (query)                               448.70       282.71       731.41       1.0000     0.000008         1.15
IVF-GPU-nl273-np16 (query)                               448.70       263.99       712.69       1.0000     0.000000         1.15
IVF-GPU-nl273-np23 (query)                               448.70       263.47       712.17       1.0000     0.000000         1.15
IVF-GPU-nl273 (self)                                     448.70     1_661.07     2_109.77       1.0000     0.000000         1.15
IVF-GPU-nl387-np19 (query)                               857.88       288.84     1_146.72       1.0000     0.000000         1.15
IVF-GPU-nl387-np27 (query)                               857.88       412.75     1_270.63       1.0000     0.000000         1.15
IVF-GPU-nl387 (self)                                     857.88     1_323.27     2_181.16       1.0000     0.000000         1.15
IVF-GPU-nl547-np23 (query)                             1_597.03       163.09     1_760.12       1.0000     0.000000         1.15
IVF-GPU-nl547-np27 (query)                             1_597.03       293.37     1_890.40       1.0000     0.000000         1.15
IVF-GPU-nl547-np33 (query)                             1_597.03       350.64     1_947.68       1.0000     0.000000         1.15
IVF-GPU-nl547 (self)                                   1_597.03     1_264.16     2_861.19       1.0000     0.000000         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.22     1_870.88     1_874.10       1.0000     0.000000        18.31
Exhaustive (self)                                          3.22    19_627.22    19_630.44       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.86       671.57       677.43       1.0000     0.000001        18.31
GPU-Exhaustive (self)                                      5.86     5_509.97     5_515.84       1.0000     0.000001        18.31
IVF-GPU-nl273-np13 (query)                               439.81       289.73       729.54       1.0000     0.000004         1.15
IVF-GPU-nl273-np16 (query)                               439.81       336.54       776.35       1.0000     0.000001         1.15
IVF-GPU-nl273-np23 (query)                               439.81       418.48       858.30       1.0000     0.000001         1.15
IVF-GPU-nl273 (self)                                     439.81     1_579.41     2_019.22       1.0000     0.000001         1.15
IVF-GPU-nl387-np19 (query)                               845.17       306.11     1_151.28       1.0000     0.000001         1.15
IVF-GPU-nl387-np27 (query)                               845.17       401.82     1_246.99       1.0000     0.000001         1.15
IVF-GPU-nl387 (self)                                     845.17     1_272.35     2_117.51       1.0000     0.000001         1.15
IVF-GPU-nl547-np23 (query)                             1_926.66       168.04     2_094.70       1.0000     0.000001         1.15
IVF-GPU-nl547-np27 (query)                             1_926.66       327.13     2_253.79       1.0000     0.000001         1.15
IVF-GPU-nl547-np33 (query)                             1_926.66       368.67     2_295.33       1.0000     0.000001         1.15
IVF-GPU-nl547 (self)                                   1_926.66     1_299.61     3_226.27       1.0000     0.000001         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.36     7_214.45     7_229.81       1.0000     0.000000        73.24
Exhaustive (self)                                         15.36    75_703.45    75_718.80       1.0000     0.000000        73.24
GPU-Exhaustive (query)                                    23.41     1_410.10     1_433.52       1.0000     0.000027        73.24
GPU-Exhaustive (self)                                     23.41    12_824.02    12_847.43       1.0000     0.000027        73.24
IVF-GPU-nl273-np13 (query)                               510.47       503.97     1_014.45       0.9996     0.006405         1.15
IVF-GPU-nl273-np16 (query)                               510.47       538.44     1_048.91       1.0000     0.000161         1.15
IVF-GPU-nl273-np23 (query)                               510.47       644.51     1_154.98       1.0000     0.000027         1.15
IVF-GPU-nl273 (self)                                     510.47     4_095.10     4_605.57       1.0000     0.000027         1.15
IVF-GPU-nl387-np19 (query)                               885.71       466.95     1_352.66       1.0000     0.000027         1.15
IVF-GPU-nl387-np27 (query)                               885.71       585.93     1_471.64       1.0000     0.000027         1.15
IVF-GPU-nl387 (self)                                     885.71     3_454.05     4_339.76       1.0000     0.000027         1.15
IVF-GPU-nl547-np23 (query)                             1_759.02       332.39     2_091.41       0.9998     0.002357         1.15
IVF-GPU-nl547-np27 (query)                             1_759.02       461.53     2_220.55       1.0000     0.000211         1.15
IVF-GPU-nl547-np33 (query)                             1_759.02       593.08     2_352.10       1.0000     0.000027         1.15
IVF-GPU-nl547 (self)                                   1_759.02     3_257.61     5_016.63       1.0000     0.000027         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### Comparison against IVF CPU

In this case, the IVF CPU implementation is being compared against the GPU
version. GPU acceleration shines with larger data sets and larger dimensions,
hence, the number of samples was increased to 250_000 and dimensions to 64 or
128 for these benchmarks.

#### With 250k samples and 64 dimensions

<details>
<summary><b>CPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 64D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.44     5_841.76     5_853.19       1.0000     0.000000        61.04
Exhaustive (self)                                         11.44    96_574.81    96_586.25       1.0000     0.000000        61.04
IVF-nl353-np17 (query)                                 1_343.44       370.39     1_713.84       1.0000     0.000000        61.12
IVF-nl353-np18 (query)                                 1_343.44       387.33     1_730.77       1.0000     0.000000        61.12
IVF-nl353-np26 (query)                                 1_343.44       568.19     1_911.63       1.0000     0.000000        61.12
IVF-nl353 (self)                                       1_343.44     7_802.13     9_145.57       1.0000     0.000000        61.12
IVF-nl500-np22 (query)                                 2_398.44       364.71     2_763.14       1.0000     0.000016        61.16
IVF-nl500-np25 (query)                                 2_398.44       392.52     2_790.96       1.0000     0.000000        61.16
IVF-nl500-np31 (query)                                 2_398.44       494.50     2_892.93       1.0000     0.000000        61.16
IVF-nl500 (self)                                       2_398.44     6_645.71     9_044.15       1.0000     0.000000        61.16
IVF-nl707-np26 (query)                                 4_862.39       304.73     5_167.12       1.0000     0.000025        61.21
IVF-nl707-np35 (query)                                 4_862.39       405.50     5_267.90       1.0000     0.000000        61.21
IVF-nl707-np37 (query)                                 4_862.39       417.14     5_279.54       1.0000     0.000000        61.21
IVF-nl707 (self)                                       4_862.39     5_331.26    10_193.66       1.0000     0.000000        61.21
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:lowrank:64:250000 -->
</code></pre>
</details>

---

The results here are more favourable of the GPU acceleration. We go from ~90
seconds with exhaustive search on CPU to ~20 seconds on GPU for full kNN
generation; with the IVF variants, we can go from 10 seconds for the CPU based
version to ~7 seconds on the GPU one, a smaller effect than on for the
exhaustive search.

---

<details>
<summary><b>CPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        25.59    12_511.72    12_537.31       1.0000     0.000000       122.07
Exhaustive (self)                                         25.59   215_978.71   216_004.30       1.0000     0.000000       122.07
IVF-nl353-np17 (query)                                   758.37       788.87     1_547.24       0.9999     0.001614       122.25
IVF-nl353-np18 (query)                                   758.37       806.36     1_564.73       0.9999     0.000721       122.25
IVF-nl353-np26 (query)                                   758.37     1_157.94     1_916.31       1.0000     0.000000       122.25
IVF-nl353 (self)                                         758.37    18_657.56    19_415.92       1.0000     0.000000       122.25
IVF-nl500-np22 (query)                                 1_520.02       694.91     2_214.93       1.0000     0.000569       122.32
IVF-nl500-np25 (query)                                 1_520.02       756.89     2_276.91       1.0000     0.000000       122.32
IVF-nl500-np31 (query)                                 1_520.02       923.85     2_443.87       1.0000     0.000000       122.32
IVF-nl500 (self)                                       1_520.02    15_436.23    16_956.25       1.0000     0.000000       122.32
IVF-nl707-np26 (query)                                 3_114.40       620.14     3_734.54       0.9999     0.001561       122.42
IVF-nl707-np35 (query)                                 3_114.40       825.55     3_939.95       1.0000     0.000000       122.42
IVF-nl707-np37 (query)                                 3_114.40       828.65     3_943.05       1.0000     0.000000       122.42
IVF-nl707 (self)                                       3_114.40    13_859.73    16_974.13       1.0000     0.000000       122.42
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:lowrank:128:250000 -->
</code></pre>
</details>

The exhaustive kNN search on the CPU takes ~200 seconds (3+ minutes). Leveraging
the GPU, we cut this down to 30 seconds, a 4x speedup. The IVF CPU as a highly
optimised version takes 15 seconds, we can cut this down to 10 seconds. In
this case, the acceleration is more modest (similar as before) – the exhaustiv
search benefits from the large volume of data.

#### Increasing the number of samples

Results are becoming more pronounced with more samples and showing the
advantage of the GPU acceleration.

<details>
<summary><b>CPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 64D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.59    13_210.02    13_231.61       1.0000     0.000000       122.07
Exhaustive (self)                                         21.59   442_828.88   442_850.47       1.0000     0.000000       122.07
IVF-nl500-np22 (query)                                 2_603.33       668.39     3_271.72       1.0000     0.000000       122.20
IVF-nl500-np25 (query)                                 2_603.33       763.88     3_367.21       1.0000     0.000000       122.20
IVF-nl500-np31 (query)                                 2_603.33       951.55     3_554.88       1.0000     0.000000       122.20
IVF-nl500 (self)                                       2_603.33    30_943.79    33_547.12       1.0000     0.000000       122.20
IVF-nl707-np26 (query)                                 4_729.74       589.27     5_319.01       1.0000     0.000014       122.25
IVF-nl707-np35 (query)                                 4_729.74       792.58     5_522.32       1.0000     0.000000       122.25
IVF-nl707-np37 (query)                                 4_729.74       835.47     5_565.20       1.0000     0.000000       122.25
IVF-nl707 (self)                                       4_729.74    26_747.77    31_477.51       1.0000     0.000000       122.25
IVF-nl1000-np31 (query)                                8_947.82       516.37     9_464.19       1.0000     0.000033       122.32
IVF-nl1000-np44 (query)                                8_947.82       713.16     9_660.98       1.0000     0.000000       122.32
IVF-nl1000-np50 (query)                                8_947.82       808.19     9_756.01       1.0000     0.000000       122.32
IVF-nl1000 (self)                                      8_947.82    22_506.22    31_454.04       1.0000     0.000000       122.32
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:lowrank:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:ivf:euclidean:lowrank:128:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:lowrank:128:500000 -->
</code></pre>
</details>

---

The overall trends hold true. The exhaustive search becomes much faster on the
GPU, the IVF-based version gets a decent 2x bonus here. In this case, the
dimensionality starts being large enough that the GPU has enough data to
churn through and the difference with CPU versions becomes more apparent.

### CAGRA-type querying

The crate also offers a [CAGRA-style index](https://arxiv.org/abs/2308.15136),
combining GPU-accelerated NNDescent graph construction with CAGRA navigational
graph optimisation and beam search. The index is built in four phases:

1. **Random graph initialisation**: each node gets `build_k` random neighbours
   with computed distances, providing a baseline graph even before the forest
   runs.
2. **GPU forest initialisation**: a shallow random partition forest (default 20
   trees) groups nearby points into leaves. All-pairs distances within each
   leaf are computed on the GPU and merged into the graph via a proposal
   buffer. Leaf sizes are dynamically capped to fit within the GPU's shared
   memory budget (32 KB), so this scales correctly to high dimensions.
3. **GPU NNDescent iterations**: the standard local join loop runs entirely on
   the GPU. Each iteration builds reverse edges, evaluates (new, new) and
   (new, old) candidate pairs in shared memory, and merges proposals into the
   sorted graph. Convergence is checked by downloading a single `u32` counter
   per iteration. Typically converges in 4-6 iterations.
4. **CAGRA graph optimisation**: the NNDescent graph (at degree `build_k`) is
   pruned to degree `k` using rank-based detour counting, reverse edge
   construction, and forward/reverse merge. This produces a directed
   navigational graph with improved long-range reachability for beam search.

Querying uses a GPU beam search kernel: one workgroup per query, with the query
vector in shared memory, a sorted candidate queue, and a linear-probing hash
table for visited-node deduplication. Beam width and iteration limits are
scaled automatically based on `k` and the graph degree via
`CagraGpuSearchParams::from_graph()`. For small individual queries, a CPU
path is used that doesn't have the overhead of the GPU kernel launches.
Generally speaking, this index does not perform too well on very well separated
data. However, it does perform well on low-rank data.

#### Parameter guidance

The two key build parameters are `build_k` (internal NNDescent degree before
CAGRA pruning) and `refine_knn` (number of 2-hop refinement sweeps after
NNDescent convergence).

**Key parameters:**

* `build_k`: Internal NNDescent degree before CAGRA pruning. Defaults to 2 * k.
  Higher values give CAGRA more edges to select from when building the
  navigational graph, at the cost of build time. 3 * k shows diminishing returns.
* `refine_knn`: Number of 2-hop refinement sweeps after NNDescent convergence.
  Each sweep evaluates all neighbours-of-neighbours and merges improvements.
  Defaults to 0. Marginal benefit for beam search recall; primarily improves
  extract graph quality up to the forest ceiling.
* `n_trees`: Number of random partition trees for forest initialisation.
  Defaults to 5 + n^0.25, capped at 20. More trees raise the raw graph quality
  ceiling but increase build time linearly.
* `beam_width`: Number of active candidates maintained during beam search.
  Defaults to 2 * max(k_out, k_graph). Wider beams improve recall at the cost
  of query latency. Auto-scaled when using CagraGpuSearchParams::from_graph().
* `max_beam_iters`: Safety cap on beam search iterations. Defaults to
  3 * beam_width. Most queries terminate naturally well before this limit; it
  only fires for pathological cases where the search keeps discovering better
  candidates.
* `n_entry_points`: Number of seed nodes per query for beam search. Defaults
  to 8. Entry points are sourced from a small Annoy forest (external queries)
  or from the kNN graph's closest neighbours (self-query).

Generally speaking CAGRA allows for very fast querying; however, the generation
of the index takes a bit more time compared to IVF for example. Also, it "fails"
in very well clustered data. It works better in data sets

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Cosine (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:cosine:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Correlated)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:correlated:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:128 -->
</code></pre>
</details>

#### Larger data sets

Let's test CAGRA similar to IVF GPU on larger data sets.

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:64:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:128:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:128:500000 -->
</code></pre>
</details>

#### Two-tier kNN generation

For downstream tasks that require a full kNN graph (e.g. BBKNN, MNN, UMAP,
Leiden clustering), the index offers three paths with different speed/accuracy
trade-offs:

| Method | Mechanism | Typical recall | Use case |
|--------|-----------|---------------|----------|
| **Extract** | Direct reshape of the NNDescent graph. No search performed. | ~0.9 | Fast, however, lowever precision. |
| **Self-beam** | GPU beam search over the CAGRA navigational graph for every vector in the index. | 0.99 | Production kNN graphs for all types of applications. |

Below are examples of kNN generation. The dimensions are specifically kept
quite low to mimic single cell situations. This is where the CAGRA-style part
is quite performant and can be used to quickly generate kNN graphs from the
data... To run these, you can use:

```bash
cargo run --example knn_comparison_cagra --features gpu --release
```

The application idea here is to use these for large single cell data sets in
which the kNN can be further accelerated.

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 32 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:32:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:64:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 32 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:32:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 32 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:32:1000000 -->
</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra_knn:euclidean:lowrank:64:1000000 -->
</code></pre>
</details>

Let's do one large data set with 2.5m samples at 32 dimensions and see what
happens ... ?

<details>
<summary><b>Generation of a kNN graph with CAGRA (2.5m samples; 32 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             57.25 1_468_501.04 1_468_558.29       1.0000     0.000000       305.18
CPU-NNDescent (k=15)                                  65_984.14    20_250.36    86_234.49       0.9991     0.000418      3235.77
--------------------------------------------------------------------------------------------------------------------------------
GPU-NND bk=1x refine=0 (extract)                      19_919.09       442.76    20_361.85       0.8489     0.583217      1449.59
GPU-NND bk=1x refine=0 (self-beam)                    19_919.09    12_834.35    32_753.43       0.9807     0.010239      1449.59
GPU-NND bk=1x refine=1 (extract)                      23_022.03       449.21    23_471.24       0.8991     0.537751      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    23_022.03    12_792.96    35_814.99       0.9831     0.008733      1449.59
GPU-NND bk=1x refine=2 (extract)                      26_379.19       449.91    26_829.09       0.9043     0.533820      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    26_379.19    12_955.97    39_335.16       0.9835     0.008434      1449.59
--------------------------------------------------------------------------------------------------------------------------------
GPU-NND bk=2x refine=0 (extract)                      27_630.71       439.61    28_070.32       0.9113     0.529758      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    27_630.71    12_704.59    40_335.30       0.9902     0.003774      1449.59
GPU-NND bk=2x refine=1 (extract)                      37_544.14       438.75    37_982.89       0.9301     0.515632      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    37_544.14    12_614.01    50_158.15       0.9925     0.002372      1449.59
GPU-NND bk=2x refine=2 (extract)                      50_440.12       441.07    50_881.19       0.9308     0.515140      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    50_440.12    13_043.14    63_483.26       0.9928     0.002233      1449.59
--------------------------------------------------------------------------------------------------------------------------------
GPU-NND bk=3x refine=0 (extract)                      50_051.65       455.02    50_506.68       0.9255     0.518781      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    50_051.65    12_844.15    62_895.81       0.9925     0.002232      1449.59
GPU-NND bk=3x refine=1 (extract)                      68_104.28       439.79    68_544.07       0.9329     0.513755      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    68_104.28    12_767.82    80_872.10       0.9937     0.001551      1449.59
GPU-NND bk=3x refine=2 (extract)                      91_675.37       437.76    92_113.14       0.9330     0.513689      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    91_675.37    12_860.18   104_535.56       0.9937     0.001526      1449.59
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

Especially on larger data sets, we can accelerate the queries substantially
and get up to 2x to 3x speed increases to generate the full kNN graph with
Recall@k of ≥0.99. If you are okay with a graph that has Recall ≥0.9 you
can do that in <10 seconds on a million samples or ~30 seconds on 2.5 million
samples (with n_dim = 32 dim). Also, the data is very contrived here... On real
data, NNDescent will have to do quite a few iterations. The Annoy
initialisations are already very good, so the CPU version basically has no
need for refining the kNN graph. On real data, the GPU outperforms more
substantially.

## Conclusions

GPU acceleration in the setting of the `wgpu` backend only starts making sense
with large indices and large dimensionality (assuming you can hold the data
in VRAM or unified memory for Apple Silicon). With smaller dimensionalities and
less samples, the overhead of launching the GPU kernels does not give
substantial performance benefits over the highly optimised CPU code. Exhaustive
searches over larger data sets however become more viable with GPU acceleration
and it here where some of the biggest gains can be observed. To note, these
implemetations are not designed (and cannot) compete with what is possible
on data centre GPUs with cuBLAS under the hood! They serve as an acceleration
in specific situations and were designed to enable fast kNN generation for
1m to 10m sample situations with lower dimensions (think single cell).

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
*Last update: 2026/04/04 with version **0.2.11***
