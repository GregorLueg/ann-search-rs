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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.23       645.74       656.96       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.23     6_687.57     6_698.80       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_461.22        39.03     2_500.25       0.9989          1.0002            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (self)                        2_461.22       353.89     2_815.11       0.9998          1.0000            1.0000        74.67
NNDescent-k:auto-nt4-dp0 (extract)                     2_461.22         4.39     2_465.61       0.9997          1.0000            1.0000        74.67
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_074.98        35.67     2_110.64       0.9988          1.0003            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (self)                        2_074.98       325.64     2_400.62       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt8-dp0 (extract)                     2_074.98         3.25     2_078.23       0.9997          1.0000            1.0000        79.51
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_056.86        48.87     2_105.73       0.9996          1.0001            1.0000        98.92
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_056.86        63.95     2_120.81       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_056.86        34.84     2_091.70       0.9988          1.0003            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (self)                    2_056.86       335.07     2_391.93       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_056.86         2.65     2_059.51       0.9998          1.0000            1.0000        98.92
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_212.49        51.45     2_263.94       0.9990          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_212.49       482.73     2_695.22       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_212.49         3.23     2_215.72       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_174.91        53.57     2_228.48       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_174.91       511.78     2_686.70       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_174.91         2.66     2_177.58       0.9998          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_164.67        56.70     2_221.37       0.9992          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (self)                    2_164.67       550.56     2_715.23       0.9994          1.0000            1.0000       116.09
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_164.67         2.71     2_167.38       0.9998          1.0000            1.0000       116.09
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 32D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.25       652.83       664.08       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.25     6_637.32     6_648.57       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_422.71        52.11     1_474.83       0.9989          1.0001            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (self)                        1_422.71       489.26     1_911.97       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt4-dp0 (extract)                     1_422.71         4.24     1_426.96       1.0000          1.0000            1.0000        74.63
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_264.89        52.41     1_317.30       0.9989          1.0001            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (self)                        1_264.89       483.90     1_748.79       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt8-dp0 (extract)                     1_264.89         2.86     1_267.75       1.0000          1.0000            1.0000        79.45
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_110.82        71.51     1_182.32       0.9994          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_110.82        88.15     1_198.97       0.9997          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_110.82        50.29     1_161.10       0.9989          1.0001            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (self)                    1_110.82       483.96     1_594.77       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_110.82         2.76     1_113.57       1.0000          1.0000            1.0000        98.73
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_312.69        56.73     1_369.41       0.9998          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_312.69       528.09     1_840.78       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_312.69         3.23     1_315.91       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_255.77        56.65     1_312.42       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_255.77       537.23     1_793.00       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_255.77         2.64     1_258.41       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_239.30        56.31     1_295.61       0.9999          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (self)                    1_239.30       532.90     1_772.20       1.0000          1.0000            1.0000       115.90
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_239.30         2.72     1_242.02       1.0000          1.0000            1.0000       115.90
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (NN embeddings; 128 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 150k samples, 128D
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.88     1_221.59     1_270.47       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.88    12_928.23    12_977.11       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_957.75        93.05     2_050.80       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (self)                        1_957.75       887.62     2_845.37       1.0000          1.0000            1.0000       135.29
NNDescent-k:auto-nt4-dp0 (extract)                     1_957.75         4.49     1_962.24       0.9999          1.0000            1.0000       135.29
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_764.16        92.21     1_856.38       0.9999          1.0001            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (self)                        1_764.16       887.83     2_651.99       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt8-dp0 (extract)                     1_764.16         2.70     1_766.87       1.0000          1.0000            1.0000       145.87
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_805.26       125.04     1_930.30       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_805.26       156.82     1_962.08       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_805.26        91.69     1_896.95       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (self)                    1_805.26       887.02     2_692.28       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_805.26         2.78     1_808.04       1.0000          1.0000            1.0000       188.15
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_083.37        96.75     2_180.12       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_083.37       931.92     3_015.29       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_083.37         3.31     2_086.68       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_044.17        95.39     2_139.56       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_044.17       918.36     2_962.53       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_044.17         2.74     2_046.91       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_017.71        93.57     2_111.28       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (self)                    2_017.71       900.57     2_918.28       1.0000          1.0000            1.0000       205.32
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_017.71         2.95     2_020.66       1.0000          1.0000            1.0000       205.32
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             25.92     3_980.45     4_006.37       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                     890.15         0.00       890.15       0.9968          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   589.27         0.00       589.27       0.9814          1.0016            1.0000        30.52
GPU-kNN bk=1x refine=1                                   430.41         0.00       430.41       0.9866          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   415.91         0.00       415.91       0.9870          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                   792.00         0.00       792.00       0.9989          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                   759.19         0.00       759.19       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                   824.09         0.00       824.09       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_133.72         0.00     1_133.72       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 1_207.02         0.00     1_207.02       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 1_367.14         0.00     1_367.14       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             60.05     6_620.84     6_680.89       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   1_204.04         0.00     1_204.04       0.9968          1.0002            1.0000        30.52
GPU-kNN bk=1x refine=0                                   813.18         0.00       813.18       0.9810          1.0016            1.0000        30.52
GPU-kNN bk=1x refine=1                                   847.69         0.00       847.69       0.9864          1.0011            1.0000        30.52
GPU-kNN bk=1x refine=2                                   967.67         0.00       967.67       0.9868          1.0011            1.0000        30.52
GPU-kNN bk=2x refine=0                                 1_099.62         0.00     1_099.62       0.9989          1.0001            1.0000        30.52
GPU-kNN bk=2x refine=1                                 1_825.43         0.00     1_825.43       0.9994          1.0000            1.0000        30.52
GPU-kNN bk=2x refine=2                                 2_606.29         0.00     2_606.29       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=0                                 1_781.70         0.00     1_781.70       0.9995          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=1                                 3_508.09         0.00     3_508.09       0.9999          1.0000            1.0000        30.52
GPU-kNN bk=3x refine=2                                 5_296.13         0.00     5_296.13       0.9999          1.0000            1.0000        30.52
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             57.95    14_743.07    14_801.02       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   2_077.40         0.00     2_077.40       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                   993.48         0.00       993.48       0.9721          1.0025            1.0000        61.04
GPU-kNN bk=1x refine=1                                 1_032.07         0.00     1_032.07       0.9791          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                 1_172.52         0.00     1_172.52       0.9797          1.0017            1.0000        61.04
GPU-kNN bk=2x refine=0                                 1_388.25         0.00     1_388.25       0.9982          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 2_135.35         0.00     2_135.35       0.9990          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 2_918.04         0.00     2_918.04       0.9991          1.0001            1.0000        61.04
GPU-kNN bk=3x refine=0                                 2_163.81         0.00     2_163.81       0.9994          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                 3_869.83         0.00     3_869.83       0.9998          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                 5_605.09         0.00     5_605.09       0.9999          1.0000            1.0000        61.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            149.95    25_227.56    25_377.51       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   2_984.11         0.00     2_984.11       0.9962          1.0002            1.0000        61.04
GPU-kNN bk=1x refine=0                                 1_458.50         0.00     1_458.50       0.9714          1.0025            1.0000        61.04
GPU-kNN bk=1x refine=1                                 2_036.23         0.00     2_036.23       0.9786          1.0018            1.0000        61.04
GPU-kNN bk=1x refine=2                                 2_843.06         0.00     2_843.06       0.9791          1.0018            1.0000        61.04
GPU-kNN bk=2x refine=0                                 2_132.50         0.00     2_132.50       0.9982          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=1                                 5_219.27         0.00     5_219.27       0.9990          1.0001            1.0000        61.04
GPU-kNN bk=2x refine=2                                 8_472.25         0.00     8_472.25       0.9990          1.0001            1.0000        61.04
GPU-kNN bk=3x refine=0                                 3_598.86         0.00     3_598.86       0.9993          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=1                                10_243.34         0.00    10_243.34       0.9998          1.0000            1.0000        61.04
GPU-kNN bk=3x refine=2                                17_825.54         0.00    17_825.54       0.9999          1.0000            1.0000        61.04
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            126.24    56_632.46    56_758.70       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   5_299.96         0.00     5_299.96       0.9963          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 2_017.92         0.00     2_017.92       0.9605          1.0037            1.0000       122.07
GPU-kNN bk=1x refine=1                                 2_454.28         0.00     2_454.28       0.9691          1.0027            1.0000       122.07
GPU-kNN bk=1x refine=2                                 3_150.17         0.00     3_150.17       0.9700          1.0026            1.0000       122.07
GPU-kNN bk=2x refine=0                                 2_989.53         0.00     2_989.53       0.9974          1.0002            1.0000       122.07
GPU-kNN bk=2x refine=1                                 5_646.28         0.00     5_646.28       0.9984          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                 8_428.44         0.00     8_428.44       0.9985          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 4_892.36         0.00     4_892.36       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                10_974.05         0.00    10_974.05       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                17_278.97         0.00    17_278.97       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (1m samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            314.82    98_502.42    98_817.23       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                   7_932.82         0.00     7_932.82       0.9963          1.0002            1.0000       122.07
GPU-kNN bk=1x refine=0                                 3_151.67         0.00     3_151.67       0.9599          1.0037            1.0000       122.07
GPU-kNN bk=1x refine=1                                 4_914.28         0.00     4_914.28       0.9687          1.0028            1.0000       122.07
GPU-kNN bk=1x refine=2                                 6_788.04         0.00     6_788.04       0.9696          1.0027            1.0000       122.07
GPU-kNN bk=2x refine=0                                 4_485.79         0.00     4_485.79       0.9974          1.0002            1.0000       122.07
GPU-kNN bk=2x refine=1                                13_457.60         0.00    13_457.60       0.9984          1.0001            1.0000       122.07
GPU-kNN bk=2x refine=2                                22_776.16         0.00    22_776.16       0.9985          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=0                                 8_103.33         0.00     8_103.33       0.9992          1.0001            1.0000       122.07
GPU-kNN bk=3x refine=1                                28_943.79         0.00    28_943.79       0.9998          1.0000            1.0000       122.07
GPU-kNN bk=3x refine=2                                49_890.17         0.00    49_890.17       0.9998          1.0000            1.0000       122.07
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>kNN generation (2.5m samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            295.81   344_754.32   345_050.13       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  15_596.26         0.00    15_596.26       0.9943          1.0004            1.0000       305.18
GPU-kNN bk=1x refine=0                                 5_525.58         0.00     5_525.58       0.9393          1.0060            1.0000       305.18
GPU-kNN bk=1x refine=1                                 7_429.86         0.00     7_429.86       0.9505          1.0047            1.0000       305.18
GPU-kNN bk=1x refine=2                                 9_714.33         0.00     9_714.33       0.9520          1.0045            1.0000       305.18
GPU-kNN bk=2x refine=0                                 8_444.01         0.00     8_444.01       0.9958          1.0003            1.0000       305.18
GPU-kNN bk=2x refine=1                                19_363.97         0.00    19_363.97       0.9973          1.0002            1.0000       305.18
GPU-kNN bk=2x refine=2                                30_861.33         0.00    30_861.33       0.9974          1.0002            1.0000       305.18
GPU-kNN bk=3x refine=0                                13_337.47         0.00    13_337.47       0.9989          1.0001            1.0000       305.18
GPU-kNN bk=3x refine=1                                38_111.82         0.00    38_111.82       0.9996          1.0000            1.0000       305.18
GPU-kNN bk=3x refine=2                                64_046.00         0.00    64_046.00       0.9996          1.0000            1.0000       305.18
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
===================================================================================================================
Benchmark: 250k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         546.25       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         468.90       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1031.93       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1015.92       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8         980.88       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1090.39       0.9992        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
===================================================================================================================
Benchmark: 250k samples, 64D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -         814.65       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         721.84       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1501.01       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1459.87       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1485.00       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1539.63       0.9993        3_750_000 / 3_750_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Clustered GPU NN-Descent (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
===================================================================================================================
Benchmark: 500k samples, 32D, k=15 (clustered vs unbatched GPU NN-Descent)
===================================================================================================================
Method                             Clusters     Build (ms)     Recall@k                       Filled   Fill (%)
-------------------------------------------------------------------------------------------------------------------
NNDescent-GPU (unbatched)                 -        1107.32       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1013.08       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2078.50       0.9987        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        1971.45       0.9985        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        1904.71       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        1966.59       0.9987        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
