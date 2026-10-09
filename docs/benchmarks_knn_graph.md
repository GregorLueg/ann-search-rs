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

A random-projection forest seeds the graph, then local joins over
neighbours-of-neighbours refine it until the improving fraction drops below
`delta`. Three rows per configuration: `(query)` against held-out data,
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
Exhaustive (query)                                        11.30       670.68       681.98       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.30     6_873.26     6_884.56       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_832.59        42.43     2_875.01       0.9989          1.0002            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (self)                        2_832.59       386.21     3_218.80       0.9997          1.0000            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (extract)                     2_832.59         4.46     2_837.05       0.9997          1.0000            1.0000       101.64
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_539.34        41.94     2_581.29       0.9988          1.0003            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (self)                        2_539.34       382.29     2_921.63       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (extract)                     2_539.34         3.56     2_542.91       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_421.36        56.61     2_477.97       0.9995          1.0001            1.0000       114.51
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_421.36        74.42     2_495.78       0.9996          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_421.36        41.09     2_462.45       0.9988          1.0003            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0 (self)                    2_421.36       387.17     2_808.53       0.9998          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_421.36         4.78     2_426.14       0.9997          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_617.32        52.35     2_669.67       0.9989          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_617.32       508.04     3_125.36       0.9991          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_617.32         3.25     2_620.57       0.9997          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_570.29        53.98     2_624.28       0.9992          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_570.29       514.48     3_084.78       0.9994          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_570.29         3.64     2_573.93       0.9997          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_617.64        56.66     2_674.30       0.9992          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp1 (self)                    2_617.64       540.99     3_158.62       0.9993          1.0000            1.0000       114.51
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_617.64         3.72     2_621.36       0.9997          1.0000            1.0000       114.51
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
Exhaustive (query)                                        11.21       665.78       676.99       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.21     6_824.70     6_835.91       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_783.67        54.18     1_837.85       0.9989          1.0001            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (self)                        1_783.67       524.67     2_308.34       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (extract)                     1_783.67         4.34     1_788.01       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_558.43        53.79     1_612.22       0.9990          1.0001            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (self)                        1_558.43       505.51     2_063.95       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (extract)                     1_558.43         3.44     1_561.87       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_544.76        78.67     1_623.43       0.9994          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_544.76       100.32     1_645.07       0.9997          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_544.76        55.86     1_600.62       0.9989          1.0001            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0 (self)                    1_544.76       523.54     2_068.30       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_544.76         3.47     1_548.23       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_787.74        58.32     1_846.06       0.9998          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_787.74       566.70     2_354.45       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_787.74         3.29     1_791.03       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_729.72        63.17     1_792.90       0.9999          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_729.72       598.53     2_328.25       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_729.72         4.85     1_734.57       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_723.31        57.89     1_781.20       0.9998          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp1 (self)                    1_723.31       570.55     2_293.86       1.0000          1.0000            1.0000       115.00
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_723.31         3.61     1_726.92       1.0000          1.0000            1.0000       115.00
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
Exhaustive (query)                                        49.81     1_271.94     1_321.75       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.81    12_747.97    12_797.78       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_413.91       113.26     2_527.17       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (self)                        2_413.91     1_056.89     3_470.80       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (extract)                     2_413.91         3.56     2_417.47       0.9999          1.0000            1.0000       213.55
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_373.85       110.50     2_484.35       0.9999          1.0001            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (self)                        2_373.85     1_055.24     3_429.09       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (extract)                     2_373.85         3.67     2_377.52       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_545.34       149.62     2_694.96       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_545.34       187.54     2_732.88       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_545.34       108.18     2_653.52       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0 (self)                    2_545.34     1_048.89     3_594.23       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_545.34         4.13     2_549.47       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_843.32       113.56     2_956.88       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_843.32     1_102.19     3_945.52       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_843.32         3.29     2_846.61       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_839.19       113.10     2_952.29       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_839.19     1_092.63     3_931.82       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_839.19         3.63     2_842.82       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_770.61       113.34     2_883.96       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp1 (self)                    2_770.61     1_062.94     3_833.56       1.0000          1.0000            1.0000       256.33
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_770.61         3.82     2_774.44       1.0000          1.0000            1.0000       256.33
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
initialisation only proposes within leaves rather than running a full forest
query per point, reverse edges are capped at `build_k` per node, and proposals
past `MAX_PROPOSALS = 128` per node per iteration are dropped in arrival order.
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
GPU-Exhaustive (ground truth)                             26.51     4_730.82     4_757.33       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   2_684.13     1_102.69     3_786.82       1.0000          1.0000            1.0000       195.75
GPU-kNN bk=1x refine=0                                   643.86         6.21       650.07       0.9813          1.0016            1.0000        87.74
GPU-kNN bk=1x refine=1                                   483.34         5.11       488.45       0.9866          1.0011            1.0000        87.74
GPU-kNN bk=1x refine=2                                   426.32         4.43       430.75       0.9870          1.0011            1.0000        87.74
GPU-kNN bk=2x refine=0                                   718.47         7.01       725.48       0.9973          1.0002            1.0000        87.74
GPU-kNN bk=2x refine=1                                   874.57         4.99       879.57       0.9991          1.0001            1.0000        87.74
GPU-kNN bk=2x refine=2                                   828.49         5.52       834.01       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=0                                 1_113.75         5.21     1_118.97       0.9984          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=1                                 1_329.51         5.36     1_334.87       0.9998          1.0000            1.0000        87.74
GPU-kNN bk=3x refine=2                                 1_476.92         4.90     1_481.82       0.9998          1.0000            1.0000        87.74
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
GPU-Exhaustive (ground truth)                             61.86     7_363.21     7_425.06       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   3_369.07     1_516.54     4_885.61       1.0000          1.0000            1.0000       274.03
GPU-kNN bk=1x refine=0                                   820.17         5.26       825.44       0.9810          1.0016            1.0000       118.26
GPU-kNN bk=1x refine=1                                   851.85         5.38       857.23       0.9864          1.0011            1.0000       118.26
GPU-kNN bk=1x refine=2                                   987.92         4.68       992.60       0.9867          1.0011            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_179.81         4.45     1_184.26       0.9972          1.0002            1.0000       118.26
GPU-kNN bk=2x refine=1                                 1_945.50         4.30     1_949.80       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_716.23         4.22     2_720.45       0.9992          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 1_922.21         4.40     1_926.61       0.9984          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_734.37         4.21     3_738.58       0.9998          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_585.09         4.20     5_589.29       0.9998          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                             59.22    17_679.46    17_738.67       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   5_483.96     2_528.32     8_012.28       0.9999          1.0000            1.0000       386.48
GPU-kNN bk=1x refine=0                                 1_041.74        11.58     1_053.32       0.9721          1.0025            1.0000       175.48
GPU-kNN bk=1x refine=1                                 1_102.60        12.29     1_114.89       0.9791          1.0018            1.0000       175.48
GPU-kNN bk=1x refine=2                                 1_222.18        11.74     1_233.92       0.9797          1.0017            1.0000       175.48
GPU-kNN bk=2x refine=0                                 1_575.34        10.47     1_585.81       0.9963          1.0003            1.0000       175.48
GPU-kNN bk=2x refine=1                                 2_344.17        10.23     2_354.40       0.9985          1.0001            1.0000       175.48
GPU-kNN bk=2x refine=2                                 3_103.75         9.67     3_113.42       0.9986          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=0                                 2_394.62         9.70     2_404.31       0.9980          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=1                                 4_153.45        12.32     4_165.78       0.9996          1.0000            1.0000       175.48
GPU-kNN bk=3x refine=2                                 5_963.88        10.13     5_974.01       0.9997          1.0000            1.0000       175.48
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
GPU-Exhaustive (ground truth)                            151.87    28_360.44    28_512.31       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   7_069.71     3_829.73    10_899.44       0.9999          1.0000            1.0000       540.55
GPU-kNN bk=1x refine=0                                 1_603.38        11.33     1_614.72       0.9714          1.0025            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_124.14        11.57     2_135.71       0.9785          1.0018            1.0000       236.51
GPU-kNN bk=1x refine=2                                 2_689.96        11.96     2_701.92       0.9791          1.0018            1.0000       236.51
GPU-kNN bk=2x refine=0                                 2_553.88        12.10     2_565.98       0.9962          1.0003            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_493.09        11.13     5_504.22       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_350.62        11.26     8_361.88       0.9986          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 4_111.83        12.44     4_124.27       0.9980          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                10_763.00        10.35    10_773.36       0.9996          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                17_479.68        12.47    17_492.15       0.9997          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            125.84    67_378.27    67_504.11       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  12_145.41     6_112.10    18_257.51       0.9998          1.0000            1.0000       768.94
GPU-kNN bk=1x refine=0                                 2_145.41        24.46     2_169.87       0.9605          1.0037            1.0000       350.95
GPU-kNN bk=1x refine=1                                 2_689.43        21.07     2_710.50       0.9691          1.0027            1.0000       350.95
GPU-kNN bk=1x refine=2                                 3_118.07        21.26     3_139.34       0.9700          1.0027            1.0000       350.95
GPU-kNN bk=2x refine=0                                 3_411.24        22.37     3_433.60       0.9951          1.0004            1.0000       350.95
GPU-kNN bk=2x refine=1                                 6_216.94        22.51     6_239.45       0.9977          1.0002            1.0000       350.95
GPU-kNN bk=2x refine=2                                 9_172.11        20.39     9_192.49       0.9979          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=0                                 5_476.98        18.67     5_495.65       0.9977          1.0002            1.0000       350.95
GPU-kNN bk=3x refine=1                                11_881.41        20.75    11_902.16       0.9995          1.0000            1.0000       350.95
GPU-kNN bk=3x refine=2                                18_498.32        21.84    18_520.16       0.9995          1.0000            1.0000       350.95
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
GPU-Exhaustive (ground truth)                            324.35   110_207.15   110_531.50       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                  15_526.05     9_216.81    24_742.85       0.9998          1.0000            1.0000      1113.09
GPU-kNN bk=1x refine=0                                 3_260.68        24.29     3_284.97       0.9599          1.0037            1.0000       473.02
GPU-kNN bk=1x refine=1                                 5_005.32        20.88     5_026.20       0.9687          1.0028            1.0000       473.02
GPU-kNN bk=1x refine=2                                 6_798.79        22.15     6_820.94       0.9696          1.0027            1.0000       473.02
GPU-kNN bk=2x refine=0                                 5_565.60        23.31     5_588.91       0.9951          1.0004            1.0000       473.02
GPU-kNN bk=2x refine=1                                13_864.16        22.60    13_886.75       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=2x refine=2                                22_170.27        21.44    22_191.71       0.9978          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=0                                 9_202.03        21.53     9_223.56       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=3x refine=1                                28_702.76        21.36    28_724.12       0.9994          1.0000            1.0000       473.02
GPU-kNN bk=3x refine=2                                48_477.45        16.79    48_494.24       0.9995          1.0000            1.0000       473.02
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
GPU-Exhaustive (ground truth)                            292.10   413_218.15   413_510.25       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  32_124.75    18_449.03    50_573.78       0.9996          1.0000            1.0000      1952.33
GPU-kNN bk=1x refine=0                                 6_022.26        70.54     6_092.80       0.9393          1.0060            1.0000       877.38
GPU-kNN bk=1x refine=1                                 7_971.41        58.66     8_030.07       0.9505          1.0047            1.0000       877.38
GPU-kNN bk=1x refine=2                                10_190.24        58.06    10_248.29       0.9520          1.0045            1.0000       877.38
GPU-kNN bk=2x refine=0                                 9_990.03        45.97    10_036.00       0.9932          1.0005            1.0000       877.38
GPU-kNN bk=2x refine=1                                20_378.26        54.55    20_432.80       0.9963          1.0003            1.0000       877.38
GPU-kNN bk=2x refine=2                                30_634.21        45.28    30_679.50       0.9965          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=0                                15_046.38        58.08    15_104.46       0.9972          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=1                                38_632.18        56.14    38_688.32       0.9992          1.0001            1.0000       877.38
GPU-kNN bk=3x refine=2                                62_545.72        57.36    62_603.08       0.9992          1.0001            1.0000       877.38
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
NNDescent-GPU (unbatched)                 -         663.59       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         504.16       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1090.24       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1119.55       0.9990        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1110.42       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1215.27       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -         997.37       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         831.49       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1727.47       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1673.58       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1685.14       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1708.62       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1247.69       0.9909        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1162.88       0.9909        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2545.07       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2442.04       0.9983        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2086.25       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2150.78       0.9985        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.10.1 (commit v0.10.1-18-g25372f1), run on 2026-10-09.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
