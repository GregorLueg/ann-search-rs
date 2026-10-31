## Self-kNN graph benchmarks

Every index in this crate can produce a full self-kNN graph by querying itself,
but a handful build one as a by-product of construction and can hand it back
without a search pass at all. That is the cheap path, and it is what downstream
single-cell work (BBKNN, MNN, UMAP, Leiden) actually wants. This page collects
those paths and the searched ones they compete against.

```bash
# CPU NN-Descent: build, self-beam and raw extract in one table
cargo run --example gridsearch_nndescent --release

# GPU NN-Descent: the raw kNN graph across build_k and refinement
cargo run --example knn_comparison_gpu --features gpu --release

# Clustered GPU NN-Descent: cluster-count sweep for datasets past the binding limit
cargo run --example gridsearch_clustered_nndescent --features gpu --release
```

## Table of Contents

- [The three paths](#the-three-paths)
- [CPU NN-Descent](#cpu-nn-descent)
- [GPU NN-Descent](#gpu-nn-descent)
- [Scaling to millions of points](#scaling-to-millions-of-points)
- [Clustered GPU NN-Descent](#clustered-gpu-nn-descent)

### The three paths

| Path | API | Mechanism |
|---|---|---|
| **Extract** | `extract_nndescent_knn`, `extract_nndescent_knn_gpu`, `extract_knn_graph_gpu` | Reshapes the graph the descent already built. No search runs. |
| **Self-beam** | `query_nndescent_self`, `query_nndescent_index_gpu_self` | Beam search over the graph for every point in the index. |
| **Any other index** | `query_*_self` | The index's own self-query fast path. Costs a full search. |

Extract rows can come back shorter than `k` where the descent never filled a
row, which the search-based paths never produce. The extract path is also
capped by the build-time degree, so asking for more neighbours than the graph
holds gets you what it has.

All three extract functions take `include_self`. A kNN graph stores no `i -> i`
edge, but every `query_*_self` and any exhaustive ground truth counts a point as
its own nearest neighbour at distance zero. Set the flag to compare like for
like; leave it unset for a graph of true neighbours only. `k` is the total row
length either way, so the self-edge takes a slot rather than being added on top.

`build_knn_graph_gpu` and `build_clustered_knn_graph_gpu` are the slim
counterparts: they return a bare `KnnGraphGpu` with no query functions at all,
for NSG feeders and raw-kNN consumers. `extract_knn_graph_gpu` is the way out of
one.

### CPU NN-Descent

A random-projection forest seeds the graph by joining every leaf exhaustively,
then local joins over neighbours-of-neighbours refine it until the improving
fraction drops below `delta`. Three rows per configuration: `(query)` against held-out data,
`(self)` for the full self-kNN via beam search, and `(extract)` for the descent
graph as-is. The gap between `(self)` and `(extract)` is exactly what the beam
search buys on top of the graph.

The `(extract)` row is taken with `include_self`, so the trivial self-edge is
back before scoring. Without it the row would lose a flat `1/k` against every
other row and every other gridsearch.

**Tunable parameters:** see
[the standard benchmarks](benchmarks_standard.md#nndescent). The one that
matters most here is the graph degree `k`, which is the ceiling on what
`(extract)` can return.

<details>
<summary><b>CPU NN-Descent - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.36       680.31       691.66       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.36     6_747.19     6_758.55       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_492.43        39.51     2_531.94       0.9989          1.0002            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (self)                        2_492.43       357.32     2_849.74       0.9998          1.0000            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (extract)                     2_492.43         3.12     2_495.54       0.9997          1.0000            1.0000        74.67
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_072.25        37.77     2_110.02       0.9988          1.0003            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (self)                        2_072.25       351.03     2_423.28       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (extract)                     2_072.25         2.75     2_075.00       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_160.04        65.04     2_225.08       0.9996          1.0001            1.0000        98.92
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_160.04        76.40     2_236.44       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_160.04        43.84     2_203.89       0.9988          1.0003            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (self)                    2_160.04       407.28     2_567.33       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_160.04         2.89     2_162.93       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_471.50        52.24     2_523.74       0.9990          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_471.50       486.85     2_958.35       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_471.50         3.38     2_474.87       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_648.01        53.75     2_701.76       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_648.01       523.13     3_171.14       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_648.01         2.83     2_650.84       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_247.25        57.73     2_304.97       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (self)                    2_247.25       552.48     2_799.73       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_247.25         2.64     2_249.89       0.9998          1.0000            1.0000       116.09
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.22       673.39       684.61       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.22     6_739.49     6_750.71       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_496.40        52.71     1_549.11       0.9989          1.0001            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (self)                        1_496.40       479.72     1_976.12       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (extract)                     1_496.40         2.95     1_499.34       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_347.32        51.46     1_398.77       0.9989          1.0001            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (self)                        1_347.32       480.48     1_827.79       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (extract)                     1_347.32         2.60     1_349.92       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_141.89        72.34     1_214.23       0.9994          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_141.89        91.44     1_233.33       0.9997          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_141.89        51.77     1_193.66       0.9989          1.0001            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (self)                    1_141.89       486.82     1_628.71       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_141.89         2.56     1_144.45       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_314.27        57.40     1_371.67       0.9998          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_314.27       529.07     1_843.34       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_314.27         3.03     1_317.31       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_286.68        56.59     1_343.27       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_286.68       551.90     1_838.58       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_286.68         2.69     1_289.37       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_286.60        56.07     1_342.67       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (self)                    1_286.60       538.50     1_825.09       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_286.60         2.90     1_289.50       1.0000          1.0000            1.0000       115.90
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (NN embeddings; 128 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.59     1_240.98     1_290.57       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.59    12_644.18    12_693.77       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_929.44        93.37     2_022.81       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (self)                        1_929.44       893.07     2_822.52       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (extract)                     1_929.44         4.04     1_933.48       0.9999          1.0000            1.0000       135.29
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_779.54        91.95     1_871.49       0.9999          1.0001            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (self)                        1_779.54       887.28     2_666.82       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (extract)                     1_779.54         2.80     1_782.34       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_797.28       125.25     1_922.53       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_797.28       156.56     1_953.84       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_797.28        91.71     1_888.99       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (self)                    1_797.28       883.82     2_681.11       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_797.28         2.77     1_800.05       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_082.52        96.59     2_179.11       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_082.52       931.37     3_013.89       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_082.52         3.16     2_085.68       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_110.09        95.31     2_205.40       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_110.09       913.21     3_023.29       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_110.09         3.60     2_113.69       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_062.95        93.42     2_156.38       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (self)                    2_062.95       910.06     2_973.02       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_062.95         2.99     2_065.95       1.0000          1.0000            1.0000       205.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### GPU NN-Descent

The same algorithm with the local join on the GPU. `build_knn_graph_gpu` runs
the descent and stops there: no CAGRA rank-prune, no reverse-merge, no second
graph copy in memory. `extract_knn_graph_gpu` reshapes what comes out. The sweep
varies `build_k` (the internal working degree, as a multiple of `k`) and
`refine_knn` (2-hop refinement sweeps after convergence), with a CPU NN-Descent
row and a GPU exhaustive ground truth for reference.

Dimensions are kept deliberately low here to mimic single-cell embeddings.

As in the CPU section, the extract rows put the trivial self-edge back before
scoring. The GPU graph stores non-self neighbours only, so without the fix-up
every row here would lose a flat `1/k` against the ground truth and the numbers
would say nothing about the graph.

Where the GPU descent does genuinely differ from the CPU one: the forest
initialisation also joins leaves but runs fewer trees and keeps a capped number
of leaf proposals per point, reverse edges are capped at `build_k` per node, and proposals past
`MAX_PROPOSALS = 128` per node per iteration are dropped in arrival order.
`refine_knn` is the knob that buys those back.

**Tunable parameters:**

- *`build_k`*: Internal NN-Descent working degree, defaults to `1.5 * k`. A
  wider degree gives the descent more room to improve, at linear build cost.
- *`refine_knn`*: 2-hop refinement sweeps after convergence. Each sweep
  evaluates all neighbours-of-neighbours and merges improvements.
- *`n_trees`*: Random-partition trees for the forest initialisation. Defaults to
  `5 + n^0.25`, capped at 20.
- *`delta`*: Convergence threshold on the improving fraction.

<details>
<summary><b>kNN generation (250k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             26.31     3_977.78     4_004.09       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                     897.92         0.00       897.92       0.9968          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   700.86         0.00       700.86       0.9814          1.0016            1.0000        30.52
GPU-kNN bk=1x refine=1                                   443.50         0.00       443.50       0.9867          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   414.40         0.00       414.40       0.9870          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                   777.62         0.00       777.62       0.9989          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                   756.92         0.00       756.92       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                   823.74         0.00       823.74       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_142.50         0.00     1_142.50       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 1_207.03         0.00     1_207.03       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 1_372.08         0.00     1_372.08       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             58.72     6_614.58     6_673.30       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   1_224.39         0.00     1_224.39       0.9968          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   819.19         0.00       819.19       0.9810          1.0016            1.0000        30.52
GPU-kNN bk=1x refine=1                                   847.62         0.00       847.62       0.9864          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   969.17         0.00       969.17       0.9867          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                 1_098.72         0.00     1_098.72       0.9989          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                 1_840.90         0.00     1_840.90       0.9994          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                 2_607.88         0.00     2_607.88       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_781.65         0.00     1_781.65       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 3_506.45         0.00     3_506.45       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 5_297.95         0.00     5_297.95       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             59.12    14_734.98    14_794.10       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   2_095.45         0.00     2_095.45       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                   998.11         0.00       998.11       0.9721          1.0025            1.0000        61.04
GPU-kNN bk=1x refine=1                                 1_037.52         0.00     1_037.52       0.9791          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                 1_176.05         0.00     1_176.05       0.9797          1.0017            1.0000        61.04
GPU-kNN bk=2x refine=0                                 1_425.43         0.00     1_425.43       0.9982          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 2_136.72         0.00     2_136.72       0.9990          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 2_918.63         0.00     2_918.63       0.9991          1.0001            1.0000        61.04
GPU-kNN bk=3x refine=0                                 2_195.92         0.00     2_195.92       0.9994          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                 3_867.82         0.00     3_867.82       0.9999          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                 5_609.92         0.00     5_609.92       0.9999          1.0000            1.0000        61.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            152.23    25_224.67    25_376.91       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   3_005.01         0.00     3_005.01       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                 1_464.25         0.00     1_464.25       0.9714          1.0025            1.0000        61.04
GPU-kNN bk=1x refine=1                                 2_045.25         0.00     2_045.25       0.9785          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                 2_659.49         0.00     2_659.49       0.9791          1.0018            1.0000        61.04
GPU-kNN bk=2x refine=0                                 2_137.13         0.00     2_137.13       0.9982          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 5_044.11         0.00     5_044.11       0.9990          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 8_011.07         0.00     8_011.07       0.9991          1.0001            1.0000        61.04
GPU-kNN bk=3x refine=0                                 3_591.92         0.00     3_591.92       0.9993          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                10_280.14         0.00    10_280.14       0.9998          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                16_982.85         0.00    16_982.85       0.9999          1.0000            1.0000        61.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Scaling to millions of points

Same benchmark, more data. Note the synthetic data here is contrived: the Annoy
initialisation on the CPU side is already close to right, so the CPU descent has
little left to refine. On real data it has to work considerably harder, and the
gap widens accordingly.

<details>
<summary><b>kNN generation (1m samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            123.02    56_613.21    56_736.23       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   5_243.44         0.00     5_243.44       0.9963          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 2_006.29         0.00     2_006.29       0.9605          1.0037            1.0000       122.07
GPU-kNN bk=1x refine=1                                 2_442.64         0.00     2_442.64       0.9691          1.0027            1.0000       122.07
GPU-kNN bk=1x refine=2                                 2_982.48         0.00     2_982.48       0.9700          1.0026            1.0000       122.07
GPU-kNN bk=2x refine=0                                 2_981.52         0.00     2_981.52       0.9974          1.0002            1.0000       122.07
GPU-kNN bk=2x refine=1                                 5_656.67         0.00     5_656.67       0.9984          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                 8_389.00         0.00     8_389.00       0.9985          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 4_918.30         0.00     4_918.30       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                11_027.03         0.00    11_027.03       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                17_267.69         0.00    17_267.69       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (1m samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            311.06    98_463.25    98_774.31       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                   7_968.77         0.00     7_968.77       0.9963          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 3_302.54         0.00     3_302.54       0.9599          1.0037            1.0000       122.07
GPU-kNN bk=1x refine=1                                 4_963.41         0.00     4_963.41       0.9687          1.0028            1.0000       122.07
GPU-kNN bk=1x refine=2                                 6_704.55         0.00     6_704.55       0.9696          1.0027            1.0000       122.07
GPU-kNN bk=2x refine=0                                 4_456.21         0.00     4_456.21       0.9974          1.0002            1.0000       122.07
GPU-kNN bk=2x refine=1                                13_438.16         0.00    13_438.16       0.9984          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                22_497.89         0.00    22_497.89       0.9985          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 8_057.19         0.00     8_057.19       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                28_652.44         0.00    28_652.44       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                49_313.52         0.00    49_313.52       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (2.5m samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            286.19   344_710.20   344_996.38       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  15_356.08         0.00    15_356.08       0.9943          1.0004            1.0000       305.18
GPU-kNN bk=1x refine=0                                 5_383.92         0.00     5_383.92       0.9393          1.0060            1.0000       305.18
GPU-kNN bk=1x refine=1                                 7_385.91         0.00     7_385.91       0.9505          1.0047            1.0000       305.18
GPU-kNN bk=1x refine=2                                 9_787.47         0.00     9_787.47       0.9520          1.0045            1.0000       305.18
GPU-kNN bk=2x refine=0                                 8_535.86         0.00     8_535.86       0.9958          1.0003            1.0000       305.18
GPU-kNN bk=2x refine=1                                19_213.90         0.00    19_213.90       0.9973          1.0002            1.0000       305.18
GPU-kNN bk=2x refine=2                                30_182.20         0.00    30_182.20       0.9974          1.0002            1.0000       305.18
GPU-kNN bk=3x refine=0                                13_322.32         0.00    13_322.32       0.9989          1.0001            1.0000       305.18
GPU-kNN bk=3x refine=1                                38_059.35         0.00    38_059.35       0.9996          1.0000            1.0000       305.18
GPU-kNN bk=3x refine=2                                62_898.46         0.00    62_898.46       0.9996          1.0000            1.0000       305.18
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Clustered GPU NN-Descent

The whole dataset goes onto the device as one tensor for the plain GPU path, so
it is bounded by the per-binding limit. Past that, `build_clustered_knn_graph_gpu`
runs balanced k-means on a subsample, has every point join its two nearest
clusters, runs NN-Descent per cluster against a shared client, and merges the
subgraphs on the host. The overlap is what stitches the batch boundaries back
together. `C = 1` dispatches straight to `build_knn_graph_gpu`, since the
overlap is pure cost when the data already fits.

**Tunable parameters:**

- *Cluster count (C)*: How many batches to split into. `plan_cluster_count`
  picks one from the device limits if you do not. The sweep runs 1, 2, 4, 8 and
  16, with `C = 1` as the unbatched baseline.
- *Sample fraction*: Fraction of the data used to train the batching centroids.
  10% here.
- *Assignments per point*: Clusters each point joins. Two here; one is the
  pessimistic case rather than the sane one.

Ground truth here is a CPU exhaustive self-query rather than the GPU one the
unbatched comparison uses, which is why the sizes stop lower on this table.

**Read the fill column first.** Every launch in this crate is
`launch_unchecked`, so a dispatch that busts a device limit does no work,
returns zeros and reports no error: the panic lands on a cubecl background
thread. A batched build that silently did nothing looks like a spectacular
speed-up. The timings only mean something once the fill count is at 100%.

<details>
<summary><b>Clustered GPU NN-Descent (250k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================
Benchmark: 250k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         541.39       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         454.46       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1047.05       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1022.91       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8         992.37       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1109.30       0.9992        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================
Benchmark: 250k samples, 64D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         805.16       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         707.80       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1533.37       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1486.59       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1508.80       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1561.80       0.9993        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================
Benchmark: 500k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -        1096.43       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1         984.42       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2077.22       0.9987        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2011.99       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        1930.27       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        1993.20       0.9986        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.12.0 (commit v0.11.2-37-ge44af67), run on 2026-10-31.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
