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
Exhaustive (query)                                        11.84       641.38       653.22       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.84     6_242.25     6_254.09       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_681.24        41.69     2_722.93       0.9989          1.0002            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (self)                        2_681.24       382.54     3_063.78       0.9998          1.0000            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (extract)                     2_681.24         5.43     2_686.67       0.9997          1.0000            1.0000       101.64
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_262.25        42.05     2_304.30       0.9988          1.0003            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (self)                        2_262.25       383.75     2_646.00       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (extract)                     2_262.25         4.04     2_266.29       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_307.90        57.19     2_365.09       0.9996          1.0001            1.0000       139.77
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_307.90        73.78     2_381.68       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_307.90        41.48     2_349.38       0.9988          1.0003            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0 (self)                    2_307.90       385.27     2_693.16       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_307.90         3.67     2_311.57       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_471.78        66.80     2_538.58       0.9990          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_471.78       533.50     3_005.28       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_471.78         3.50     2_475.28       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_502.84        54.32     2_557.16       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_502.84       521.67     3_024.51       0.9994          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_502.84         3.58     2_506.42       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_424.56        57.41     2_481.97       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp1 (self)                    2_424.56       555.27     2_979.83       0.9994          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_424.56         3.83     2_428.39       0.9998          1.0000            1.0000       139.77
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
Exhaustive (query)                                        11.26       671.83       683.09       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.26     6_181.53     6_192.78       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_607.44        57.07     1_664.50       0.9989          1.0001            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (self)                        1_607.44       505.67     2_113.11       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (extract)                     1_607.44         4.20     1_611.64       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_416.75        55.26     1_472.00       0.9989          1.0001            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (self)                        1_416.75       497.98     1_914.73       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (extract)                     1_416.75         4.00     1_420.75       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_238.94        78.75     1_317.69       0.9994          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_238.94        99.45     1_338.39       0.9997          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_238.94        54.86     1_293.80       0.9989          1.0001            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0 (self)                    1_238.94       529.46     1_768.40       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_238.94         3.09     1_242.03       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_469.03        62.64     1_531.67       0.9998          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_469.03       580.24     2_049.27       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_469.03         3.63     1_472.65       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_449.91        62.76     1_512.67       0.9999          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_449.91       591.70     2_041.61       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_449.91         4.29     1_454.20       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_421.11        64.96     1_486.07       0.9999          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp1 (self)                    1_421.11       581.74     2_002.85       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_421.11         3.45     1_424.55       1.0000          1.0000            1.0000       140.73
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
Exhaustive (query)                                        50.58     1_253.72     1_304.29       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         50.58    12_612.92    12_663.50       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_230.29       111.31     2_341.61       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (self)                        2_230.29     1_073.67     3_303.97       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (extract)                     2_230.29         5.10     2_235.39       0.9999          1.0000            1.0000       213.55
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_970.86       111.66     2_082.52       0.9999          1.0001            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (self)                        1_970.86     1_056.29     3_027.14       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (extract)                     1_970.86         4.05     1_974.91       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_045.33       157.64     2_202.97       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_045.33       188.74     2_234.06       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_045.33       109.47     2_154.79       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0 (self)                    2_045.33     1_057.28     3_102.61       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_045.33         3.16     2_048.48       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_426.21       146.47     2_572.68       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_426.21     1_171.50     3_597.70       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_426.21         8.27     2_434.47       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_323.82       112.06     2_435.88       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_323.82     1_084.31     3_408.13       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_323.82         3.83     2_327.65       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_297.53       113.68     2_411.21       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp1 (self)                    2_297.53     1_072.12     3_369.65       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_297.53         4.88     2_302.41       1.0000          1.0000            1.0000       313.53
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
GPU-Exhaustive (ground truth)                             26.53     4_725.44     4_751.96       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   2_167.87       997.60     3_165.46       1.0000          1.0000            1.0000       300.95
GPU-kNN bk=1x refine=0                                   601.02         4.55       605.58       0.9814          1.0016            1.0000        59.13
GPU-kNN bk=1x refine=1                                   434.30         4.25       438.55       0.9867          1.0011            1.0000        59.13
GPU-kNN bk=1x refine=2                                   427.89         4.09       431.98       0.9870          1.0011            1.0000        59.13
GPU-kNN bk=2x refine=0                                   760.23         4.38       764.62       0.9989          1.0001            1.0000        59.13
GPU-kNN bk=2x refine=1                                   768.35         4.23       772.58       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=2x refine=2                                   825.20         4.71       829.91       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=0                                 1_094.47         4.20     1_098.67       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=1                                 1_212.13         4.06     1_216.19       0.9999          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=2                                 1_380.16         4.13     1_384.29       0.9999          1.0000            1.0000        59.13
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
GPU-Exhaustive (ground truth)                             58.55     7_329.00     7_387.54       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   2_613.21     1_457.49     4_070.70       1.0000          1.0000            1.0000       339.49
GPU-kNN bk=1x refine=0                                   800.29         4.50       804.78       0.9810          1.0016            1.0000        89.65
GPU-kNN bk=1x refine=1                                   807.75         4.23       811.99       0.9864          1.0011            1.0000        89.65
GPU-kNN bk=1x refine=2                                   958.44         4.73       963.16       0.9868          1.0011            1.0000        89.65
GPU-kNN bk=2x refine=0                                 1_139.27         4.36     1_143.63       0.9989          1.0001            1.0000        89.65
GPU-kNN bk=2x refine=1                                 1_858.40         4.23     1_862.63       0.9994          1.0000            1.0000        89.65
GPU-kNN bk=2x refine=2                                 2_599.39         4.07     2_603.46       0.9995          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=0                                 1_735.09         4.11     1_739.20       0.9995          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=1                                 3_465.95         4.54     3_470.49       0.9999          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=2                                 5_152.09         4.09     5_156.18       0.9999          1.0000            1.0000        89.65
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
GPU-Exhaustive (ground truth)                             59.94    17_250.21    17_310.15       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   4_567.22     2_448.12     7_015.34       1.0000          1.0000            1.0000       649.98
GPU-kNN bk=1x refine=0                                   955.46         9.29       964.75       0.9721          1.0025            1.0000       118.26
GPU-kNN bk=1x refine=1                                 1_027.17         9.05     1_036.22       0.9792          1.0018            1.0000       118.26
GPU-kNN bk=1x refine=2                                 1_168.92         9.02     1_177.93       0.9797          1.0017            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_462.80         9.07     1_471.87       0.9982          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=1                                 2_160.29         9.54     2_169.83       0.9990          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_903.92         9.86     2_913.78       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 2_263.44         9.58     2_273.02       0.9993          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_854.37         9.27     3_863.64       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_566.81         8.96     5_575.78       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                            149.65    27_725.58    27_875.22       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   5_763.58     3_710.24     9_473.83       1.0000          1.0000            1.0000       900.05
GPU-kNN bk=1x refine=0                                 1_488.88         8.98     1_497.86       0.9714          1.0025            1.0000       179.29
GPU-kNN bk=1x refine=1                                 2_008.53         8.76     2_017.29       0.9786          1.0018            1.0000       179.29
GPU-kNN bk=1x refine=2                                 2_644.87         8.99     2_653.86       0.9791          1.0018            1.0000       179.29
GPU-kNN bk=2x refine=0                                 2_326.27         8.80     2_335.08       0.9982          1.0001            1.0000       179.29
GPU-kNN bk=2x refine=1                                 5_105.18         9.25     5_114.43       0.9990          1.0001            1.0000       179.29
GPU-kNN bk=2x refine=2                                 7_993.98         8.69     8_002.66       0.9990          1.0001            1.0000       179.29
GPU-kNN bk=3x refine=0                                 3_695.28        10.09     3_705.38       0.9993          1.0000            1.0000       179.29
GPU-kNN bk=3x refine=1                                10_116.95        10.76    10_127.71       0.9998          1.0000            1.0000       179.29
GPU-kNN bk=3x refine=2                                16_579.06         9.13    16_588.20       0.9999          1.0000            1.0000       179.29
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
GPU-Exhaustive (ground truth)                            121.36    66_271.39    66_392.76       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  10_180.99     6_244.14    16_425.14       0.9999          1.0000            1.0000      1696.23
GPU-kNN bk=1x refine=0                                 2_014.70        20.73     2_035.43       0.9605          1.0037            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_515.86        24.36     2_540.22       0.9691          1.0027            1.0000       236.51
GPU-kNN bk=1x refine=2                                 3_102.74        23.16     3_125.91       0.9700          1.0027            1.0000       236.51
GPU-kNN bk=2x refine=0                                 3_155.52        20.74     3_176.26       0.9974          1.0002            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_866.47        21.63     5_888.09       0.9984          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_618.78        20.57     8_639.35       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 5_102.23        20.18     5_122.41       0.9992          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                11_283.73        20.34    11_304.07       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                17_631.32        24.29    17_655.61       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            319.94   110_441.37   110_761.31       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                  13_701.95     9_168.20    22_870.14       0.9999          1.0000            1.0000      1940.39
GPU-kNN bk=1x refine=0                                 3_108.25        19.98     3_128.22       0.9599          1.0037            1.0000       358.58
GPU-kNN bk=1x refine=1                                 4_781.92        20.14     4_802.05       0.9687          1.0028            1.0000       358.58
GPU-kNN bk=1x refine=2                                 6_500.63        18.67     6_519.30       0.9696          1.0027            1.0000       358.58
GPU-kNN bk=2x refine=0                                 4_952.15        18.63     4_970.78       0.9974          1.0002            1.0000       358.58
GPU-kNN bk=2x refine=1                                12_745.20        18.49    12_763.69       0.9984          1.0001            1.0000       358.58
GPU-kNN bk=2x refine=2                                20_856.54        18.57    20_875.11       0.9985          1.0001            1.0000       358.58
GPU-kNN bk=3x refine=0                                 8_420.57        18.09     8_438.67       0.9992          1.0001            1.0000       358.58
GPU-kNN bk=3x refine=1                                26_808.19        18.78    26_826.97       0.9998          1.0000            1.0000       358.58
GPU-kNN bk=3x refine=2                                46_021.22        16.05    46_037.27       0.9998          1.0000            1.0000       358.58
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
GPU-Exhaustive (ground truth)                            300.01   413_128.83   413_428.85       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  30_335.55    18_290.27    48_625.82       0.9999          1.0000            1.0000      3890.93
GPU-kNN bk=1x refine=0                                 5_369.95        72.04     5_441.99       0.9393          1.0060            1.0000       591.28
GPU-kNN bk=1x refine=1                                 7_536.37        57.18     7_593.55       0.9505          1.0047            1.0000       591.28
GPU-kNN bk=1x refine=2                                 9_905.19        49.99     9_955.18       0.9520          1.0045            1.0000       591.28
GPU-kNN bk=2x refine=0                                 9_028.88        48.59     9_077.47       0.9958          1.0003            1.0000       591.28
GPU-kNN bk=2x refine=1                                19_457.06        50.72    19_507.78       0.9973          1.0002            1.0000       591.28
GPU-kNN bk=2x refine=2                                29_950.37        47.24    29_997.62       0.9974          1.0002            1.0000       591.28
GPU-kNN bk=3x refine=0                                13_772.30        46.54    13_818.84       0.9989          1.0001            1.0000       591.28
GPU-kNN bk=3x refine=1                                37_631.64        52.23    37_683.86       0.9996          1.0000            1.0000       591.28
GPU-kNN bk=3x refine=2                                61_592.87        51.69    61_644.55       0.9996          1.0000            1.0000       591.28
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
NNDescent-GPU (unbatched)                 -         668.98       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         486.42       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1136.59       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1150.63       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1165.87       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1348.03       0.9992        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -         933.60       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         794.92       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1709.42       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1658.68       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1691.43       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1852.25       0.9993        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1161.43       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1095.01       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2358.07       0.9987        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2197.58       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2147.15       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2251.41       0.9986        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.11.0 (commit v0.10.1-78-gec8a881), run on 2026-10-11.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
