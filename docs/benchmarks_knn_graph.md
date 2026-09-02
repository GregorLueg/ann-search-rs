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
Exhaustive (query)                                        11.21       677.14       688.35       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.21     6_813.14     6_824.35       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_803.47        47.40     2_850.87       0.9989          1.0002            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (self)                        2_803.47       453.90     3_257.37       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (extract)                     2_803.47         4.74     2_808.21       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_498.65        46.26     2_544.91       0.9988          1.0003            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (self)                        2_498.65       433.76     2_932.41       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (extract)                     2_498.65         3.97     2_502.62       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_482.61        64.38     2_546.99       0.9995          1.0001            1.0000       131.68
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_482.61        82.35     2_564.96       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_482.61        46.47     2_529.08       0.9988          1.0003            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (self)                    2_482.61       433.99     2_916.60       0.9998          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_482.61         3.82     2_486.43       0.9997          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_599.43        56.92     2_656.35       0.9989          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_599.43       550.31     3_149.74       0.9991          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_599.43         4.05     2_603.48       0.9221          1.0197            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_630.55        59.43     2_689.98       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_630.55       560.89     3_191.44       0.9994          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_630.55         3.35     2_633.89       0.9225          1.0496            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_609.08        63.15     2_672.22       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (self)                    2_609.08       603.48     3_212.56       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_609.08         4.13     2_613.20       0.9252          1.0974            1.0000       131.68
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
Exhaustive (query)                                        11.61       649.83       661.44       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.61     6_695.13     6_706.74       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_801.67        63.97     1_865.64       0.9989          1.0001            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (self)                        1_801.67       583.64     2_385.32       1.0000          1.0000            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (extract)                     1_801.67         5.72     1_807.40       1.0000          1.0000            1.0000       119.05
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_545.84        64.96     1_610.80       0.9989          1.0001            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (self)                        1_545.84       580.75     2_126.59       1.0000          1.0000            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (extract)                     1_545.84         3.83     1_549.67       1.0000          1.0000            1.0000       131.67
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_580.40        85.66     1_666.06       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_580.40       106.49     1_686.89       0.9996          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_580.40        59.75     1_640.14       0.9989          1.0001            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (self)                    1_580.40       576.22     2_156.62       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_580.40         3.96     1_584.36       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_791.75        65.11     1_856.86       0.9998          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_791.75       620.84     2_412.59       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_791.75         3.42     1_795.18       0.9455          1.0062            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_789.68        67.25     1_856.93       0.9998          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_789.68       631.70     2_421.38       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_789.68         3.59     1_793.27       0.9575          1.0055            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_748.30        67.03     1_815.33       0.9999          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (self)                    1_748.30       633.84     2_382.14       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_748.30         3.89     1_752.18       0.9874          1.0017            1.0000       132.16
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
Exhaustive (query)                                        48.68     1_272.15     1_320.83       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         48.68    12_537.43    12_586.11       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_523.02       115.71     2_638.73       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (self)                        2_523.02     1_100.34     3_623.36       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (extract)                     2_523.02         5.14     2_528.16       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_458.81       117.61     2_576.42       0.9999          1.0001            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (self)                        2_458.81     1_107.61     3_566.42       1.0000          1.0000            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (extract)                     2_458.81         3.91     2_462.72       1.0000          1.0000            1.0000       245.14
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_729.16       153.79     2_882.95       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_729.16       195.16     2_924.32       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_729.16       114.03     2_843.18       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (self)                    2_729.16     1_095.23     3_824.39       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_729.16         3.88     2_733.04       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         3_022.34       118.05     3_140.39       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (self)                 3_022.34     1_139.72     4_162.06       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (extract)              3_022.34         3.41     3_025.74       0.9970          1.0005            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_997.81       116.21     3_114.02       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_997.81     1_119.43     4_117.24       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_997.81         3.57     3_001.38       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_986.75       113.89     3_100.65       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (self)                    2_986.75     1_103.80     4_090.55       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_986.75         3.90     2_990.65       1.0000          1.0000            1.0000       273.50
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
GPU-Exhaustive (ground truth)                             22.66     8_932.22     8_954.89       1.0000          1.0000            1.0000        30.52
CPU-NNDescent (k=15)                                   2_766.67     1_114.58     3_881.26       1.0000          1.0000            1.0000       224.36
GPU-kNN bk=1x refine=0                                   562.03         4.81       566.84       0.9813          1.0016            1.0000        87.74
GPU-kNN bk=1x refine=1                                   495.10         4.22       499.32       0.9866          1.0011            1.0000        87.74
GPU-kNN bk=1x refine=2                                   480.52         4.59       485.11       0.9870          1.0011            1.0000        87.74
GPU-kNN bk=2x refine=0                                   789.91         5.56       795.47       0.9973          1.0002            1.0000        87.74
GPU-kNN bk=2x refine=1                                   867.98         4.78       872.76       0.9991          1.0001            1.0000        87.74
GPU-kNN bk=2x refine=2                                   935.59         4.62       940.21       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=0                                 1_175.77         4.50     1_180.27       0.9986          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=1                                 1_368.97         4.82     1_373.79       0.9999          1.0000            1.0000        87.74
GPU-kNN bk=3x refine=2                                 1_556.52         4.60     1_561.13       0.9999          1.0000            1.0000        87.74
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
GPU-Exhaustive (ground truth)                             54.02    12_753.08    12_807.11       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   3_481.95     1_590.75     5_072.70       1.0000          1.0000            1.0000       302.64
GPU-kNN bk=1x refine=0                                   822.25         5.81       828.06       0.9810          1.0016            1.0000       118.26
GPU-kNN bk=1x refine=1                                   891.97         4.64       896.61       0.9864          1.0011            1.0000       118.26
GPU-kNN bk=1x refine=2                                 1_027.40         4.35     1_031.75       0.9867          1.0011            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_211.51         4.63     1_216.14       0.9972          1.0002            1.0000       118.26
GPU-kNN bk=2x refine=1                                 1_995.92         4.13     2_000.04       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_751.39         4.39     2_755.78       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 1_983.98         4.17     1_988.15       0.9986          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_716.48         4.15     3_720.63       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_457.37         4.03     5_461.40       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                             50.54    34_945.02    34_995.56       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   5_635.03     2_784.00     8_419.03       0.9999          1.0000            1.0000       473.70
GPU-kNN bk=1x refine=0                                 1_102.38        11.84     1_114.22       0.9721          1.0025            1.0000       175.48
GPU-kNN bk=1x refine=1                                 1_169.26        10.40     1_179.66       0.9791          1.0018            1.0000       175.48
GPU-kNN bk=1x refine=2                                 1_297.79         9.43     1_307.22       0.9797          1.0017            1.0000       175.48
GPU-kNN bk=2x refine=0                                 1_729.99         8.47     1_738.45       0.9963          1.0003            1.0000       175.48
GPU-kNN bk=2x refine=1                                 2_456.95         8.50     2_465.45       0.9985          1.0001            1.0000       175.48
GPU-kNN bk=2x refine=2                                 3_223.17         7.97     3_231.14       0.9986          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=0                                 2_489.11         7.92     2_497.03       0.9984          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=1                                 4_159.67         8.11     4_167.78       0.9998          1.0000            1.0000       175.48
GPU-kNN bk=3x refine=2                                 5_848.90         8.40     5_857.30       0.9998          1.0000            1.0000       175.48
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
GPU-Exhaustive (ground truth)                            137.73    50_685.75    50_823.47       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   7_404.75     4_048.00    11_452.75       0.9999          1.0000            1.0000       597.77
GPU-kNN bk=1x refine=0                                 1_582.16        12.96     1_595.12       0.9714          1.0025            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_070.65        10.17     2_080.82       0.9785          1.0018            1.0000       236.51
GPU-kNN bk=1x refine=2                                 2_663.81         9.53     2_673.34       0.9791          1.0018            1.0000       236.51
GPU-kNN bk=2x refine=0                                 2_648.16        10.84     2_659.00       0.9962          1.0003            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_468.99        10.59     5_479.57       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_285.29        10.43     8_295.72       0.9986          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 4_353.52        15.84     4_369.36       0.9983          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                10_765.09        10.07    10_775.16       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                17_181.97        10.35    17_192.32       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            121.08   139_647.26   139_768.34       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                  12_603.28     6_636.41    19_239.69       0.9998          1.0000            1.0000       883.38
GPU-kNN bk=1x refine=0                                 2_107.88        20.36     2_128.25       0.9605          1.0037            1.0000       350.95
GPU-kNN bk=1x refine=1                                 2_718.31        20.76     2_739.07       0.9691          1.0027            1.0000       350.95
GPU-kNN bk=1x refine=2                                 3_207.88        19.08     3_226.96       0.9700          1.0027            1.0000       350.95
GPU-kNN bk=2x refine=0                                 3_775.45        17.42     3_792.87       0.9951          1.0004            1.0000       350.95
GPU-kNN bk=2x refine=1                                 6_337.40        16.66     6_354.06       0.9978          1.0002            1.0000       350.95
GPU-kNN bk=2x refine=2                                 8_979.26        15.88     8_995.15       0.9979          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=0                                 5_387.52        16.55     5_404.07       0.9981          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=1                                11_184.70        16.45    11_201.14       0.9996          1.0000            1.0000       350.95
GPU-kNN bk=3x refine=2                                17_054.91        16.08    17_070.99       0.9997          1.0000            1.0000       350.95
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
GPU-Exhaustive (ground truth)                            296.84   205_476.36   205_773.20       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  16_631.50     9_697.29    26_328.79       0.9998          1.0000            1.0000      1213.53
GPU-kNN bk=1x refine=0                                 3_091.27        24.91     3_116.18       0.9599          1.0037            1.0000       473.02
GPU-kNN bk=1x refine=1                                 4_909.41        22.85     4_932.27       0.9687          1.0028            1.0000       473.02
GPU-kNN bk=1x refine=2                                 6_727.63        24.15     6_751.78       0.9696          1.0027            1.0000       473.02
GPU-kNN bk=2x refine=0                                 5_703.63        24.28     5_727.90       0.9951          1.0004            1.0000       473.02
GPU-kNN bk=2x refine=1                                14_339.10        23.47    14_362.57       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=2x refine=2                                23_079.35        24.37    23_103.71       0.9978          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=0                                 9_290.23        23.41     9_313.65       0.9981          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=1                                29_043.41        22.00    29_065.41       0.9996          1.0000            1.0000       473.02
GPU-kNN bk=3x refine=2                                48_941.58        21.67    48_963.25       0.9997          1.0000            1.0000       473.02
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
GPU-Exhaustive (ground truth)                            288.09   839_636.58   839_924.67       1.0000          1.0000            1.0000       305.18
CPU-NNDescent (k=15)                                  34_070.21    20_606.86    54_677.07       0.9996          1.0000            1.0000      2238.42
GPU-kNN bk=1x refine=0                                 5_761.08        75.84     5_836.92       0.9393          1.0060            1.0000       877.38
GPU-kNN bk=1x refine=1                                 7_820.32        56.69     7_877.01       0.9505          1.0047            1.0000       877.38
GPU-kNN bk=1x refine=2                                 9_914.75        49.23     9_963.99       0.9520          1.0045            1.0000       877.38
GPU-kNN bk=2x refine=0                                10_372.67        45.43    10_418.10       0.9932          1.0005            1.0000       877.38
GPU-kNN bk=2x refine=1                                20_562.62        51.06    20_613.67       0.9963          1.0003            1.0000       877.38
GPU-kNN bk=2x refine=2                                31_218.76        49.91    31_268.67       0.9965          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=0                                15_946.91        51.27    15_998.19       0.9977          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=1                                38_503.09        52.37    38_555.46       0.9994          1.0000            1.0000       877.38
GPU-kNN bk=3x refine=2                                61_727.25        57.42    61_784.67       0.9994          1.0000            1.0000       877.38
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
NNDescent-GPU (unbatched)                 -         711.14       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         563.29       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1210.79       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1272.04       0.9990        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1471.52       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1899.58       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -         992.15       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         836.55       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1796.52       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1737.41       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1945.14       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        2218.37       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1323.30       0.9909        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1229.59       0.9908        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2486.58       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2441.33       0.9983        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2527.28       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2986.54       0.9985        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
