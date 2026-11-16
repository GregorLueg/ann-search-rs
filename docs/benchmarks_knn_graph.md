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
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        12.02       678.55       690.57       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.02     6_503.19     6_515.22       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_468.45        39.69     2_508.14       0.9989          1.0002            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (self)                        2_468.45       373.87     2_842.32       0.9998          1.0000            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (extract)                     2_468.45         3.99     2_472.44       0.9997          1.0000            1.0000        74.67
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_160.64        37.87     2_198.51       0.9988          1.0003            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (self)                        2_160.64       344.22     2_504.86       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (extract)                     2_160.64         2.87     2_163.50       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_074.19        50.15     2_124.35       0.9996          1.0001            1.0000        98.92
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_074.19        63.59     2_137.79       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_074.19        36.87     2_111.06       0.9988          1.0003            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (self)                    2_074.19       335.65     2_409.84       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_074.19         2.76     2_076.95       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_222.22        51.28     2_273.50       0.9990          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_222.22       524.21     2_746.42       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_222.22         3.36     2_225.58       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_209.67        53.79     2_263.46       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_209.67       519.43     2_729.10       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_209.67         2.64     2_212.31       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_150.52        57.23     2_207.75       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (self)                    2_150.52       556.76     2_707.28       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_150.52         2.73     2_153.25       0.9998          1.0000            1.0000       116.09
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        12.10       646.68       658.78       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         12.10     6_298.07     6_310.17       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_420.39        55.45     1_475.85       0.9989          1.0001            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (self)                        1_420.39       497.67     1_918.06       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (extract)                     1_420.39         4.01     1_424.41       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_267.54        51.87     1_319.41       0.9989          1.0001            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (self)                        1_267.54       507.43     1_774.98       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (extract)                     1_267.54         2.95     1_270.49       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_080.22        73.27     1_153.49       0.9994          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_080.22        93.61     1_173.83       0.9997          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_080.22        52.68     1_132.90       0.9989          1.0001            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (self)                    1_080.22       495.06     1_575.28       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_080.22         2.98     1_083.19       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_308.71        55.74     1_364.45       0.9998          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_308.71       540.77     1_849.48       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_308.71         3.59     1_312.30       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_264.17        57.07     1_321.24       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_264.17       564.10     1_828.27       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_264.17         2.87     1_267.04       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_227.69        57.23     1_284.92       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (self)                    1_227.69       553.16     1_780.85       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_227.69         2.61     1_230.30       1.0000          1.0000            1.0000       115.90
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (NN embeddings; 128 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.16     1_235.79     1_284.95       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.16    12_387.75    12_436.91       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_912.11        96.08     2_008.18       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (self)                        1_912.11       924.87     2_836.98       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (extract)                     1_912.11         4.01     1_916.11       0.9999          1.0000            1.0000       135.29
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_759.88        96.72     1_856.60       0.9999          1.0001            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (self)                        1_759.88       918.83     2_678.71       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (extract)                     1_759.88         2.84     1_762.72       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_799.16       128.17     1_927.33       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_799.16       168.25     1_967.41       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_799.16        92.89     1_892.05       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (self)                    1_799.16       952.03     2_751.19       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_799.16         2.70     1_801.85       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_185.03        99.17     2_284.20       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_185.03     1_066.70     3_251.73       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_185.03         4.00     2_189.03       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_102.62        97.07     2_199.69       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_102.62       939.63     3_042.25       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_102.62         2.64     2_105.27       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_029.89        96.11     2_126.00       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (self)                    2_029.89       923.61     2_953.50       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_029.89         2.66     2_032.55       1.0000          1.0000            1.0000       205.32
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
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             25.80     4_149.98     4_175.78       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                     687.73         0.00       687.73       0.9968          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   622.52         0.00       622.52       0.9825          1.0015            1.0000        30.52
GPU-kNN bk=1x refine=1                                   433.98         0.00       433.98       0.9868          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   410.06         0.00       410.06       0.9871          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                   775.72         0.00       775.72       0.9991          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                   729.85         0.00       729.85       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                   788.73         0.00       788.73       0.9996          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_123.31         0.00     1_123.31       0.9994          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 1_183.71         0.00     1_183.71       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 1_321.67         0.00     1_321.67       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             58.51     6_694.56     6_753.07       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                     911.46         0.00       911.46       0.9967          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   860.87         0.00       860.87       0.9822          1.0015            1.0000        30.52
GPU-kNN bk=1x refine=1                                   705.09         0.00       705.09       0.9865          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   694.78         0.00       694.78       0.9868          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                 1_134.50         0.00     1_134.50       0.9991          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                 1_178.87         0.00     1_178.87       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                 1_318.33         0.00     1_318.33       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_773.98         0.00     1_773.98       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 2_089.95         0.00     2_089.95       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 2_369.22         0.00     2_369.22       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             58.37    14_748.84    14_807.21       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   1_601.75         0.00     1_601.75       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                   973.31         0.00       973.31       0.9737          1.0023            1.0000        61.04
GPU-kNN bk=1x refine=1                                   868.58         0.00       868.58       0.9793          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                   881.33         0.00       881.33       0.9798          1.0017            1.0000        61.04
GPU-kNN bk=2x refine=0                                 1_310.18         0.00     1_310.18       0.9986          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 1_452.35         0.00     1_452.35       0.9992          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 1_611.36         0.00     1_611.36       0.9992          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=0                                 2_202.91         0.00     2_202.91       0.9993          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                 2_525.09         0.00     2_525.09       0.9999          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                 2_892.03         0.00     2_892.03       0.9999          1.0000            1.0000        61.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            149.09    25_224.60    25_373.68       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   2_254.70         0.00     2_254.70       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                 1_541.88         0.00     1_541.88       0.9730          1.0023            1.0000        61.04
GPU-kNN bk=1x refine=1                                 1_497.61         0.00     1_497.61       0.9787          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                 1_527.12         0.00     1_527.12       0.9792          1.0017            1.0000        61.04
GPU-kNN bk=2x refine=0                                 2_103.86         0.00     2_103.86       0.9986          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 2_404.32         0.00     2_404.32       0.9992          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 2_741.23         0.00     2_741.23       0.9992          1.0001            1.0000        61.04
GPU-kNN bk=3x refine=0                                 3_869.75         0.00     3_869.75       0.9993          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                 4_680.56         0.00     4_680.56       0.9999          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                 5_334.06         0.00     5_334.06       0.9999          1.0000            1.0000        61.04
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
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            120.64    58_443.28    58_563.93       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   4_267.51         0.00     4_267.51       0.9963          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 1_982.82         0.00     1_982.82       0.9623          1.0034            1.0000       122.07
GPU-kNN bk=1x refine=1                                 1_857.86         0.00     1_857.86       0.9694          1.0027            1.0000       122.07
GPU-kNN bk=1x refine=2                                 1_885.72         0.00     1_885.72       0.9701          1.0026            1.0000       122.07
GPU-kNN bk=2x refine=0                                 2_806.32         0.00     2_806.32       0.9979          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=1                                 3_231.32         0.00     3_231.32       0.9986          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                 3_319.24         0.00     3_319.24       0.9987          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 4_558.38         0.00     4_558.38       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                 5_239.53         0.00     5_239.53       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                 5_968.42         0.00     5_968.42       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (1m samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            311.91    98_705.40    99_017.30       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                   5_449.69         0.00     5_449.69       0.9962          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 3_110.65         0.00     3_110.65       0.9620          1.0034            1.0000       122.07
GPU-kNN bk=1x refine=1                                 3_094.10         0.00     3_094.10       0.9691          1.0027            1.0000       122.07
GPU-kNN bk=1x refine=2                                 3_146.77         0.00     3_146.77       0.9698          1.0026            1.0000       122.07
GPU-kNN bk=2x refine=0                                 4_370.47         0.00     4_370.47       0.9979          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=1                                 5_025.39         0.00     5_025.39       0.9987          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                 5_659.71         0.00     5_659.71       0.9987          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 7_904.51         0.00     7_904.51       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                 9_363.99         0.00     9_363.99       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                10_862.34         0.00    10_862.34       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (2.5m samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            289.87   346_840.63   347_130.50       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  11_129.02         0.00    11_129.02       0.9943          1.0004            1.0000       305.18
GPU-kNN bk=1x refine=0                                 4_932.69         0.00     4_932.69       0.9415          1.0057            1.0000       305.18
GPU-kNN bk=1x refine=1                                 4_903.45         0.00     4_903.45       0.9510          1.0046            1.0000       305.18
GPU-kNN bk=1x refine=2                                 5_013.75         0.00     5_013.75       0.9523          1.0044            1.0000       305.18
GPU-kNN bk=2x refine=0                                 7_269.19         0.00     7_269.19       0.9965          1.0002            1.0000       305.18
GPU-kNN bk=2x refine=1                                 8_033.60         0.00     8_033.60       0.9976          1.0002            1.0000       305.18
GPU-kNN bk=2x refine=2                                 8_900.04         0.00     8_900.04       0.9977          1.0002            1.0000       305.18
GPU-kNN bk=3x refine=0                                12_056.01         0.00    12_056.01       0.9989          1.0001            1.0000       305.18
GPU-kNN bk=3x refine=1                                13_888.32         0.00    13_888.32       0.9996          1.0000            1.0000       305.18
GPU-kNN bk=3x refine=2                                15_764.37         0.00    15_764.37       0.9997          1.0000            1.0000       305.18
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
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
===================================================================================================================
Benchmark: 250k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         690.28       0.9957        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         450.65       0.9957        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1057.65       0.9995        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1012.86       0.9994        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1007.59       0.9994        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1130.88       0.9993        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
===================================================================================================================
Benchmark: 250k samples, 64D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         935.55       0.9955        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         723.22       0.9956        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1513.65       0.9995        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1469.71       0.9994        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1509.50       0.9994        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1588.07       0.9993        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-38-g4bc6780), run on 2026-11-16
===================================================================================================================
Benchmark: 500k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -        1048.33       0.9931        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1         940.66       0.9932        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        1917.97       0.9991        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        1920.30       0.9988        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        1937.20       0.9989        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2012.34       0.9989        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
