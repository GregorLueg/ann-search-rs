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
Exhaustive (query)                                        11.63       640.74       652.37       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.63     6_043.67     6_055.29       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_815.93        46.80     2_862.73       0.9989          1.0002            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (self)                        2_815.93       425.46     3_241.39       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (extract)                     2_815.93         3.09     2_819.01       0.9996          1.0000            1.0000       118.81
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_508.66        45.12     2_553.79       0.9987          1.0003            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (self)                        2_508.66       427.69     2_936.35       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (extract)                     2_508.66         2.61     2_511.27       0.9996          1.0000            1.0000       131.18
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_428.74        63.74     2_492.48       0.9995          1.0001            1.0000       131.68
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_428.74        84.31     2_513.05       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_428.74        44.89     2_473.64       0.9988          1.0003            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (self)                    2_428.74       432.19     2_860.93       0.9997          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_428.74         2.40     2_431.15       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_635.96        59.50     2_695.46       0.9989          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_635.96       562.11     3_198.07       0.9991          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_635.96         2.49     2_638.45       0.9220          1.0197            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_618.33        58.50     2_676.83       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_618.33       556.73     3_175.06       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_618.33         2.95     2_621.28       0.9225          1.0496            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_593.61        62.01     2_655.62       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (self)                    2_593.61       596.31     3_189.92       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_593.61         2.40     2_596.01       0.9252          1.0974            1.0000       131.68
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
Exhaustive (query)                                        11.28       627.03       638.32       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.28     6_075.34     6_086.62       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_788.75        65.57     1_854.32       0.9982          1.0002            1.0000       115.55
NNDescent-k:auto-nt4-dp0 (self)                        1_788.75       566.48     2_355.22       0.9994          1.0000            1.0000       115.55
NNDescent-k:auto-nt4-dp0 (extract)                     1_788.75         3.02     1_791.76       0.9994          1.0000            1.0000       115.55
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_553.72        60.70     1_614.42       0.9981          1.0001            1.0000       124.67
NNDescent-k:auto-nt8-dp0 (self)                        1_553.72       583.27     2_137.00       0.9994          1.0000            1.0000       124.67
NNDescent-k:auto-nt8-dp0 (extract)                     1_553.72         2.59     1_556.32       0.9994          1.0000            1.0000       124.67
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_621.97        90.64     1_712.61       0.9988          1.0001            1.0000       134.41
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_621.97       111.85     1_733.82       0.9991          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_621.97        60.24     1_682.20       0.9981          1.0001            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0 (self)                    1_621.97       576.22     2_198.18       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_621.97         2.54     1_624.51       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_839.55        67.12     1_906.68       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_839.55       630.44     2_470.00       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_839.55         2.38     1_841.94       0.9452          1.0061            1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_804.37        65.47     1_869.84       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_804.37       609.52     2_413.89       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_804.37         3.05     1_807.43       0.9558          1.0056            1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_773.36        67.95     1_841.31       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp1 (self)                    1_773.36       648.56     2_421.93       0.9994          1.0000            1.0000       134.41
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_773.36         2.67     1_776.03       0.9858          1.0018            1.0000       134.41
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
Exhaustive (query)                                        49.51     1_223.51     1_273.01       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.51    11_727.45    11_776.96       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_534.04       117.65     2_651.69       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (self)                        2_534.04     1_119.08     3_653.13       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (extract)                     2_534.04         3.18     2_537.22       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_456.42       114.95     2_571.37       0.9999          1.0001            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (self)                        2_456.42     1_109.84     3_566.26       0.9999          1.0000            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (extract)                     2_456.42         2.92     2_459.34       0.9999          1.0000            1.0000       245.14
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_752.43       153.03     2_905.46       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_752.43       194.70     2_947.13       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_752.43       113.03     2_865.46       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (self)                    2_752.43     1_101.82     3_854.25       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_752.43         2.31     2_754.74       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         3_068.19       118.44     3_186.63       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (self)                 3_068.19     1_157.48     4_225.67       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (extract)              3_068.19        11.26     3_079.45       0.9970          1.0005            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          3_010.95       118.46     3_129.41       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (self)                  3_010.95     1_143.54     4_154.49       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (extract)               3_010.95         2.51     3_013.46       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_989.59       115.41     3_105.00       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (self)                    2_989.59     1_135.28     4_124.86       1.0000          1.0000            1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_989.59         2.97     2_992.56       1.0000          1.0000            1.0000       273.50
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
GPU-Exhaustive (ground truth)                             26.62     8_901.84     8_928.46       1.0000          1.0000            1.0000        30.52
CPU-NNDescent (k=15)                                   2_792.59     1_183.32     3_975.90       1.0000          1.0000            1.0000       226.85
GPU-kNN bk=1x refine=0                                   561.09         3.65       564.74       0.9802          1.0017            1.0000        87.74
GPU-kNN bk=1x refine=1                                   482.95         4.19       487.13       0.9857          1.0012            1.0000        87.74
GPU-kNN bk=1x refine=2                                   483.73         2.81       486.54       0.9862          1.0011            1.0000        87.74
GPU-kNN bk=2x refine=0                                   787.25         2.68       789.92       0.9971          1.0002            1.0000        87.74
GPU-kNN bk=2x refine=1                                   845.68         2.62       848.30       0.9990          1.0001            1.0000        87.74
GPU-kNN bk=2x refine=2                                   945.30         2.85       948.15       0.9991          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=0                                 1_133.30         2.66     1_135.95       0.9985          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=1                                 1_405.15         2.76     1_407.90       0.9999          1.0000            1.0000        87.74
GPU-kNN bk=3x refine=2                                 1_622.52         2.75     1_625.27       0.9999          1.0000            1.0000        87.74
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
GPU-Exhaustive (ground truth)                             58.35    12_693.27    12_751.62       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   3_713.28     1_813.04     5_526.32       1.0000          1.0000            1.0000       301.88
GPU-kNN bk=1x refine=0                                   855.43         4.15       859.59       0.9801          1.0017            1.0000       118.26
GPU-kNN bk=1x refine=1                                   853.68         4.25       857.93       0.9857          1.0012            1.0000       118.26
GPU-kNN bk=1x refine=2                                 1_004.17         3.33     1_007.50       0.9860          1.0011            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_272.23         2.94     1_275.18       0.9971          1.0002            1.0000       118.26
GPU-kNN bk=2x refine=1                                 2_007.43         5.53     2_012.97       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_896.88         2.59     2_899.47       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 1_995.03         2.76     1_997.79       0.9985          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_821.16         2.76     3_823.92       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_422.87         2.88     5_425.74       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                             59.69    34_772.65    34_832.34       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   5_734.54     3_116.15     8_850.69       0.9999          1.0000            1.0000       441.69
GPU-kNN bk=1x refine=0                                 1_086.38         9.36     1_095.74       0.9703          1.0027            1.0000       175.48
GPU-kNN bk=1x refine=1                                 1_212.69         7.29     1_219.97       0.9776          1.0019            1.0000       175.48
GPU-kNN bk=1x refine=2                                 1_373.38         7.65     1_381.03       0.9783          1.0019            1.0000       175.48
GPU-kNN bk=2x refine=0                                 1_784.29         5.69     1_789.99       0.9961          1.0003            1.0000       175.48
GPU-kNN bk=2x refine=1                                 2_579.86         5.28     2_585.14       0.9984          1.0001            1.0000       175.48
GPU-kNN bk=2x refine=2                                 3_486.20         5.99     3_492.19       0.9985          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=0                                 2_653.32         5.22     2_658.53       0.9983          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=1                                 4_428.91         5.23     4_434.13       0.9997          1.0000            1.0000       175.48
GPU-kNN bk=3x refine=2                                 6_296.60         5.42     6_302.02       0.9998          1.0000            1.0000       175.48
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
GPU-Exhaustive (ground truth)                            159.22    50_685.54    50_844.76       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   7_475.73     4_092.00    11_567.72       0.9999          1.0000            1.0000       592.26
GPU-kNN bk=1x refine=0                                 1_581.45         9.02     1_590.47       0.9701          1.0027            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_250.80         8.84     2_259.64       0.9776          1.0019            1.0000       236.51
GPU-kNN bk=1x refine=2                                 2_871.77         7.38     2_879.15       0.9782          1.0019            1.0000       236.51
GPU-kNN bk=2x refine=0                                 2_707.30         7.92     2_715.22       0.9960          1.0003            1.0000       236.51
GPU-kNN bk=2x refine=1                                 6_097.90         7.13     6_105.03       0.9984          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_930.62         7.28     8_937.90       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 4_479.02         8.31     4_487.34       0.9983          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                10_464.63         7.50    10_472.13       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                19_694.31         7.30    19_701.62       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            124.49   139_075.23   139_199.72       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                  12_815.99     6_882.59    19_698.58       0.9998          1.0000            1.0000       891.37
GPU-kNN bk=1x refine=0                                 2_129.01        16.41     2_145.43       0.9591          1.0038            1.0000       350.95
GPU-kNN bk=1x refine=1                                 2_676.53        15.32     2_691.86       0.9680          1.0029            1.0000       350.95
GPU-kNN bk=1x refine=2                                 3_226.99        13.86     3_240.86       0.9689          1.0028            1.0000       350.95
GPU-kNN bk=2x refine=0                                 3_756.19        14.09     3_770.29       0.9950          1.0004            1.0000       350.95
GPU-kNN bk=2x refine=1                                 6_468.54        14.30     6_482.84       0.9976          1.0002            1.0000       350.95
GPU-kNN bk=2x refine=2                                 9_198.70        14.00     9_212.70       0.9978          1.0002            1.0000       350.95
GPU-kNN bk=3x refine=0                                 5_352.96        13.40     5_366.36       0.9980          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=1                                11_434.93        13.74    11_448.67       0.9996          1.0000            1.0000       350.95
GPU-kNN bk=3x refine=2                                17_569.91        13.10    17_583.01       0.9996          1.0000            1.0000       350.95
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
GPU-Exhaustive (ground truth)                            314.89   206_115.36   206_430.25       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  17_701.08    10_771.81    28_472.88       0.9998          1.0000            1.0000      1217.51
GPU-kNN bk=1x refine=0                                 3_237.16        18.01     3_255.16       0.9592          1.0038            1.0000       473.02
GPU-kNN bk=1x refine=1                                 4_867.36        16.66     4_884.02       0.9680          1.0028            1.0000       473.02
GPU-kNN bk=1x refine=2                                 6_650.06        14.63     6_664.69       0.9689          1.0027            1.0000       473.02
GPU-kNN bk=2x refine=0                                 5_897.01        15.72     5_912.73       0.9950          1.0004            1.0000       473.02
GPU-kNN bk=2x refine=1                                13_671.37        14.60    13_685.97       0.9976          1.0002            1.0000       473.02
GPU-kNN bk=2x refine=2                                22_755.67        13.94    22_769.61       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=3x refine=0                                 9_342.66        13.45     9_356.11       0.9980          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=1                                27_537.35        15.42    27_552.76       0.9996          1.0000            1.0000       473.02
GPU-kNN bk=3x refine=2                                41_588.24        14.26    41_602.50       0.9996          1.0000            1.0000       473.02
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
GPU-Exhaustive (ground truth)                            288.81   844_459.41   844_748.22       1.0000          1.0000            1.0000       305.18
CPU-NNDescent (k=15)                                  33_854.45    21_299.09    55_153.53       0.9996          1.0000            1.0000      2210.45
GPU-kNN bk=1x refine=0                                 6_019.95        42.14     6_062.09       0.9374          1.0062            1.0003       877.38
GPU-kNN bk=1x refine=1                                 8_167.86        41.44     8_209.29       0.9487          1.0049            1.0000       877.38
GPU-kNN bk=1x refine=2                                10_319.72        42.97    10_362.68       0.9503          1.0047            1.0000       877.38
GPU-kNN bk=2x refine=0                                11_932.88        72.98    12_005.86       0.9930          1.0005            1.0000       877.38
GPU-kNN bk=2x refine=1                                20_996.51        34.61    21_031.12       0.9961          1.0003            1.0000       877.38
GPU-kNN bk=2x refine=2                                30_330.05        43.52    30_373.57       0.9963          1.0003            1.0000       877.38
GPU-kNN bk=3x refine=0                                17_848.44        45.42    17_893.87       0.9977          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=1                                41_338.72        42.63    41_381.35       0.9994          1.0000            1.0000       877.38
GPU-kNN bk=3x refine=2                                61_419.91        39.05    61_458.96       0.9994          1.0000            1.0000       877.38
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
NNDescent-GPU (unbatched)                 -         718.00       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         581.92       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1244.20       0.9986        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1299.14       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1518.33       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1763.19       0.9985        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1051.53       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         855.81       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1864.53       0.9985        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1865.68       0.9982        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1939.90       0.9982        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        2106.60       0.9983        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1327.88       0.9896        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1230.21       0.9896        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2564.79       0.9978        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2362.32       0.9975        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2449.60       0.9977        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2667.15       0.9976        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
