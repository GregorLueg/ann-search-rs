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
Exhaustive (query)                                        11.32       654.52       665.83       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.32     6_646.96     6_658.28       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_912.44        46.14     2_958.58       0.9989          1.0002            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (self)                        2_912.44       428.46     3_340.91       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (extract)                     2_912.44         4.11     2_916.55       0.9996          1.0000            1.0000       118.81
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_594.31        45.53     2_639.83       0.9987          1.0003            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (self)                        2_594.31       428.22     3_022.53       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (extract)                     2_594.31         3.09     2_597.39       0.9996          1.0000            1.0000       131.18
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_465.94        64.23     2_530.17       0.9995          1.0001            1.0000       131.68
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_465.94        82.39     2_548.33       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_465.94        48.78     2_514.72       0.9988          1.0003            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (self)                    2_465.94       428.63     2_894.57       0.9997          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_465.94         4.51     2_470.45       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_647.32        57.76     2_705.08       0.9989          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_647.32       567.72     3_215.05       0.9991          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_647.32         2.51     2_649.83       0.9220          1.0197            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_606.98        57.74     2_664.72       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_606.98       554.19     3_161.17       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_606.98         1.96     2_608.94       0.9225          1.0496            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_592.47        65.07     2_657.54       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (self)                    2_592.47       630.38     3_222.86       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_592.47         2.77     2_595.24       0.9252          1.0974            1.0000       131.68
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
Exhaustive (query)                                        11.24       653.51       664.75       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.24     6_670.79     6_682.03       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_786.31        59.55     1_845.86       0.9985          1.0001            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (self)                        1_786.31       551.87     2_338.18       0.9994          1.0000            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (extract)                     1_786.31         4.37     1_790.68       0.9994          1.0000            1.0000       119.05
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_563.11        58.00     1_621.11       0.9984          1.0001            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (self)                        1_563.11       546.30     2_109.40       0.9994          1.0000            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (extract)                     1_563.11         3.55     1_566.66       0.9994          1.0000            1.0000       131.67
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_606.01        86.05     1_692.06       0.9989          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_606.01       107.09     1_713.10       0.9992          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_606.01        57.92     1_663.93       0.9984          1.0001            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (self)                    1_606.01       543.55     2_149.56       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_606.01         3.06     1_609.07       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_793.89        62.03     1_855.92       0.9993          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_793.89       598.00     2_391.88       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_793.89         2.48     1_796.37       0.9452          1.0062            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_781.02        62.58     1_843.60       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_781.02       596.66     2_377.68       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_781.02         2.98     1_784.00       0.9572          1.0055            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_772.50        66.80     1_839.30       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (self)                    1_772.50       614.80     2_387.30       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_772.50         2.84     1_775.34       0.9869          1.0017            1.0000       132.16
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
Exhaustive (query)                                        48.78     1_232.78     1_281.56       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.78    12_541.79    12_590.57       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_546.21       114.76     2_660.97       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (self)                        2_546.21     1_100.09     3_646.30       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (extract)                     2_546.21         4.57     2_550.78       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_468.76       119.82     2_588.58       0.9999          1.0001            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (self)                        2_468.76     1_102.91     3_571.67       0.9999          1.0000            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (extract)                     2_468.76         2.97     2_471.73       0.9999          1.0000            1.0000       245.14
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_745.32       152.81     2_898.13       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_745.32       201.83     2_947.16       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_745.32       113.21     2_858.54       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (self)                    2_745.32     1_100.07     3_845.39       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_745.32         2.98     2_748.30       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         3_055.82       119.21     3_175.02       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (self)                 3_055.82     1_132.58     4_188.39       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (extract)              3_055.82         2.35     3_058.17       0.9970          1.0005            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_989.90       116.30     3_106.20       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_989.90     1_115.82     4_105.72       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_989.90         2.55     2_992.45       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_974.99       114.83     3_089.81       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (self)                    2_974.99     1_100.84     4_075.82       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_974.99         2.78     2_977.77       1.0000          1.0000            1.0000       273.50
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
GPU-Exhaustive (ground truth)                             24.69     8_922.18     8_946.87       1.0000          1.0000            1.0000        30.52
CPU-NNDescent (k=15)                                   2_732.27     1_096.23     3_828.50       1.0000          1.0000            1.0000       224.36
GPU-kNN bk=1x refine=0                                   583.50         3.86       587.36       0.9813          1.0016            1.0000        87.74
GPU-kNN bk=1x refine=1                                   459.25         3.51       462.76       0.9866          1.0011            1.0000        87.74
GPU-kNN bk=1x refine=2                                   499.67         3.76       503.43       0.9870          1.0011            1.0000        87.74
GPU-kNN bk=2x refine=0                                   790.23         3.44       793.67       0.9973          1.0002            1.0000        87.74
GPU-kNN bk=2x refine=1                                   851.69         3.25       854.94       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=2x refine=2                                   939.27         4.08       943.36       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=0                                 1_172.74         3.79     1_176.54       0.9986          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=1                                 1_380.71         3.49     1_384.20       0.9999          1.0000            1.0000        87.74
GPU-kNN bk=3x refine=2                                 1_531.87         3.48     1_535.35       0.9999          1.0000            1.0000        87.74
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
GPU-Exhaustive (ground truth)                             52.95    12_742.94    12_795.89       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   3_499.14     1_648.32     5_147.46       1.0000          1.0000            1.0000       302.64
GPU-kNN bk=1x refine=0                                   857.38         4.53       861.92       0.9810          1.0016            1.0000       118.26
GPU-kNN bk=1x refine=1                                   821.58         3.08       824.66       0.9864          1.0011            1.0000       118.26
GPU-kNN bk=1x refine=2                                   980.53         2.81       983.34       0.9868          1.0011            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_205.26         2.78     1_208.04       0.9972          1.0002            1.0000       118.26
GPU-kNN bk=2x refine=1                                 1_993.03         3.23     1_996.26       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_751.04         2.76     2_753.80       0.9992          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 2_011.27         2.89     2_014.16       0.9986          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_728.89         2.69     3_731.58       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_476.23         2.44     5_478.68       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                             51.16    34_932.37    34_983.52       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   5_631.76     2_751.03     8_382.80       0.9999          1.0000            1.0000       473.70
GPU-kNN bk=1x refine=0                                 1_064.11         8.63     1_072.74       0.9721          1.0025            1.0000       175.48
GPU-kNN bk=1x refine=1                                 1_147.30         6.21     1_153.51       0.9791          1.0018            1.0000       175.48
GPU-kNN bk=1x refine=2                                 1_347.58         8.66     1_356.24       0.9797          1.0017            1.0000       175.48
GPU-kNN bk=2x refine=0                                 1_759.18         6.67     1_765.85       0.9963          1.0003            1.0000       175.48
GPU-kNN bk=2x refine=1                                 2_476.16         6.31     2_482.47       0.9985          1.0001            1.0000       175.48
GPU-kNN bk=2x refine=2                                 3_209.27         6.41     3_215.68       0.9986          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=0                                 2_517.13         6.22     2_523.36       0.9984          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=1                                 4_178.96         6.46     4_185.42       0.9998          1.0000            1.0000       175.48
GPU-kNN bk=3x refine=2                                 5_825.93         5.95     5_831.88       0.9998          1.0000            1.0000       175.48
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
GPU-Exhaustive (ground truth)                            145.51    50_712.28    50_857.79       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   7_457.68     4_116.68    11_574.36       0.9999          1.0000            1.0000       597.77
GPU-kNN bk=1x refine=0                                 1_578.65         8.25     1_586.90       0.9714          1.0025            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_125.20         7.98     2_133.18       0.9785          1.0018            1.0000       236.51
GPU-kNN bk=1x refine=2                                 2_666.59         8.65     2_675.24       0.9791          1.0018            1.0000       236.51
GPU-kNN bk=2x refine=0                                 2_650.37         6.93     2_657.31       0.9962          1.0003            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_534.02         8.10     5_542.11       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_368.26         7.93     8_376.19       0.9986          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 4_338.84         7.95     4_346.79       0.9984          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                10_753.88         7.57    10_761.45       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                17_192.12         7.69    17_199.81       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            123.13   139_673.56   139_796.68       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                  12_413.87     6_600.94    19_014.81       0.9998          1.0000            1.0000       883.38
GPU-kNN bk=1x refine=0                                 2_100.38        13.66     2_114.04       0.9604          1.0037            1.0000       350.95
GPU-kNN bk=1x refine=1                                 2_647.05        15.49     2_662.55       0.9691          1.0027            1.0000       350.95
GPU-kNN bk=1x refine=2                                 3_191.35        13.27     3_204.61       0.9700          1.0027            1.0000       350.95
GPU-kNN bk=2x refine=0                                 3_742.96        13.22     3_756.18       0.9951          1.0004            1.0000       350.95
GPU-kNN bk=2x refine=1                                 6_311.87        13.06     6_324.93       0.9977          1.0002            1.0000       350.95
GPU-kNN bk=2x refine=2                                 8_926.65        13.08     8_939.73       0.9978          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=0                                 5_332.00        13.09     5_345.08       0.9981          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=1                                11_184.70        13.54    11_198.24       0.9996          1.0000            1.0000       350.95
GPU-kNN bk=3x refine=2                                17_106.21        12.98    17_119.19       0.9997          1.0000            1.0000       350.95
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
GPU-Exhaustive (ground truth)                            310.44   205_492.29   205_802.72       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  16_426.35     9_664.37    26_090.72       0.9998          1.0000            1.0000      1213.53
GPU-kNN bk=1x refine=0                                 3_099.09        18.11     3_117.20       0.9599          1.0037            1.0000       473.02
GPU-kNN bk=1x refine=1                                 4_893.83        14.85     4_908.67       0.9687          1.0028            1.0000       473.02
GPU-kNN bk=1x refine=2                                 6_757.92        12.37     6_770.29       0.9696          1.0027            1.0000       473.02
GPU-kNN bk=2x refine=0                                 5_708.78        11.94     5_720.72       0.9951          1.0004            1.0000       473.02
GPU-kNN bk=2x refine=1                                14_391.94        12.27    14_404.21       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=2x refine=2                                23_032.27        12.79    23_045.06       0.9978          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=0                                 9_294.07        11.96     9_306.04       0.9981          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=1                                29_027.51        13.25    29_040.76       0.9996          1.0000            1.0000       473.02
GPU-kNN bk=3x refine=2                                48_889.58        13.02    48_902.60       0.9997          1.0000            1.0000       473.02
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
GPU-Exhaustive (ground truth)                            286.41   842_212.56   842_498.97       1.0000          1.0000            1.0000       305.18
CPU-NNDescent (k=15)                                  33_445.95    20_521.98    53_967.94       0.9996          1.0000            1.0000      2238.42
GPU-kNN bk=1x refine=0                                 6_001.85        65.46     6_067.31       0.9393          1.0060            1.0000       877.38
GPU-kNN bk=1x refine=1                                 7_875.23        48.39     7_923.62       0.9504          1.0047            1.0000       877.38
GPU-kNN bk=1x refine=2                                10_025.80        42.16    10_067.96       0.9520          1.0045            1.0000       877.38
GPU-kNN bk=2x refine=0                                10_675.73        37.81    10_713.54       0.9932          1.0005            1.0000       877.38
GPU-kNN bk=2x refine=1                                20_992.90        36.41    21_029.31       0.9963          1.0003            1.0000       877.38
GPU-kNN bk=2x refine=2                                30_807.15        42.94    30_850.08       0.9965          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=0                                14_906.20        38.82    14_945.01       0.9977          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=1                                38_337.88        39.82    38_377.71       0.9994          1.0000            1.0000       877.38
GPU-kNN bk=3x refine=2                                61_548.63        35.27    61_583.91       0.9994          1.0000            1.0000       877.38
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
NNDescent-GPU (unbatched)                 -         714.63       0.9934        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         582.61       0.9934        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1265.67       0.9986        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1256.23       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1487.82       0.9986        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1802.86       0.9984        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1034.11       0.9931        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         870.35       0.9931        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1751.24       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1708.90       0.9983        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1914.87       0.9983        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        2307.46       0.9983        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1346.01       0.9903        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1237.37       0.9903        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2486.84       0.9980        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2454.48       0.9977        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2475.80       0.9978        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2813.39       0.9978        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
