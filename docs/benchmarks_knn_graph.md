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
Exhaustive (query)                                        11.32       649.75       661.07       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.32     6_329.08     6_340.40       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_628.09        42.28     2_670.37       0.9989          1.0002            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (self)                        2_628.09       389.26     3_017.35       0.9998          1.0000            1.0000       101.64
NNDescent-k:auto-nt4-dp0 (extract)                     2_628.09         4.68     2_632.77       0.9997          1.0000            1.0000       101.64
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_239.62        43.02     2_282.64       0.9988          1.0003            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (self)                        2_239.62       392.84     2_632.46       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt8-dp0 (extract)                     2_239.62         4.24     2_243.85       0.9997          1.0000            1.0000       114.01
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_324.71        58.20     2_382.91       0.9996          1.0001            1.0000       139.77
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_324.71        76.50     2_401.21       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_324.71        41.90     2_366.61       0.9988          1.0003            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0 (self)                    2_324.71       401.34     2_726.05       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_324.71         3.69     2_328.40       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_457.77        55.61     2_513.38       0.9990          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_457.77       516.19     2_973.96       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_457.77         3.42     2_461.19       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_421.88        78.25     2_500.13       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_421.88       548.33     2_970.21       0.9994          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_421.88         4.08     2_425.96       0.9998          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_393.82        63.85     2_457.68       0.9992          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp1 (self)                    2_393.82       590.13     2_983.96       0.9994          1.0000            1.0000       139.77
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_393.82         3.74     2_397.56       0.9998          1.0000            1.0000       139.77
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
Exhaustive (query)                                        11.25       653.60       664.85       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.25     6_279.62     6_290.87       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_563.67        55.53     1_619.21       0.9989          1.0001            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (self)                        1_563.67       522.63     2_086.30       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt4-dp0 (extract)                     1_563.67         5.06     1_568.73       1.0000          1.0000            1.0000       101.88
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_390.83        54.71     1_445.53       0.9989          1.0001            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (self)                        1_390.83       508.87     1_899.70       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt8-dp0 (extract)                     1_390.83         3.82     1_394.64       1.0000          1.0000            1.0000       114.50
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_235.93        75.93     1_311.87       0.9994          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_235.93       100.92     1_336.85       0.9997          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_235.93        53.70     1_289.63       0.9989          1.0001            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0 (self)                    1_235.93       516.70     1_752.63       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_235.93         3.71     1_239.64       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_470.37        63.01     1_533.38       0.9998          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_470.37       591.67     2_062.04       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_470.37         3.31     1_473.68       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_443.51        64.87     1_508.38       0.9999          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_443.51       601.86     2_045.37       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_443.51         3.97     1_447.49       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_450.12        62.01     1_512.13       0.9999          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp1 (self)                    1_450.12       603.02     2_053.14       1.0000          1.0000            1.0000       140.73
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_450.12         2.91     1_453.03       1.0000          1.0000            1.0000       140.73
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
Exhaustive (query)                                        52.16     1_243.48     1_295.64       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         52.16    12_009.04    12_061.20       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_108.06       115.40     2_223.46       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (self)                        2_108.06     1_084.40     3_192.46       1.0000          1.0000            1.0000       213.55
NNDescent-k:auto-nt4-dp0 (extract)                     2_108.06         4.59     2_112.65       0.9999          1.0000            1.0000       213.55
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_950.37       112.29     2_062.66       0.9999          1.0001            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (self)                        1_950.37     1_072.20     3_022.57       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt8-dp0 (extract)                     1_950.37         3.82     1_954.19       1.0000          1.0000            1.0000       227.97
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_036.29       148.99     2_185.28       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_036.29       189.63     2_225.91       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_036.29       110.32     2_146.60       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0 (self)                    2_036.29     1_069.59     3_105.88       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_036.29         3.63     2_039.92       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_352.34       116.39     2_468.73       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_352.34     1_133.90     3_486.24       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_352.34         3.84     2_356.18       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_309.68       115.58     2_425.26       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_309.68     1_121.90     3_431.58       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_309.68         3.92     2_313.60       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_298.84       113.24     2_412.09       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp1 (self)                    2_298.84     1_104.01     3_402.85       1.0000          1.0000            1.0000       313.53
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_298.84         3.18     2_302.02       1.0000          1.0000            1.0000       313.53
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
GPU-Exhaustive (ground truth)                             26.11     4_166.80     4_192.92       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   2_135.68     1_037.26     3_172.94       1.0000          1.0000            1.0000       300.95
GPU-kNN bk=1x refine=0                                   647.78         5.27       653.04       0.9814          1.0016            1.0000        59.13
GPU-kNN bk=1x refine=1                                   442.10         5.27       447.37       0.9866          1.0011            1.0000        59.13
GPU-kNN bk=1x refine=2                                   447.99         7.06       455.05       0.9870          1.0011            1.0000        59.13
GPU-kNN bk=2x refine=0                                   814.64         4.27       818.91       0.9989          1.0001            1.0000        59.13
GPU-kNN bk=2x refine=1                                   804.90         4.45       809.35       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=2x refine=2                                   886.70         4.15       890.86       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=0                                 1_173.96         4.22     1_178.18       0.9995          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=1                                 1_270.95         4.27     1_275.22       0.9999          1.0000            1.0000        59.13
GPU-kNN bk=3x refine=2                                 1_458.94         4.23     1_463.17       0.9999          1.0000            1.0000        59.13
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
GPU-Exhaustive (ground truth)                             55.03     6_966.78     7_021.81       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   2_761.48     1_554.12     4_315.60       1.0000          1.0000            1.0000       339.49
GPU-kNN bk=1x refine=0                                   808.58         6.74       815.32       0.9810          1.0016            1.0000        89.65
GPU-kNN bk=1x refine=1                                   852.44         5.06       857.51       0.9864          1.0011            1.0000        89.65
GPU-kNN bk=1x refine=2                                 1_043.89         4.80     1_048.69       0.9868          1.0011            1.0000        89.65
GPU-kNN bk=2x refine=0                                 1_138.92         4.13     1_143.06       0.9989          1.0001            1.0000        89.65
GPU-kNN bk=2x refine=1                                 1_893.13         4.14     1_897.27       0.9994          1.0000            1.0000        89.65
GPU-kNN bk=2x refine=2                                 2_678.51         4.03     2_682.55       0.9995          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=0                                 1_842.18         4.39     1_846.57       0.9995          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=1                                 3_600.73         5.36     3_606.09       0.9999          1.0000            1.0000        89.65
GPU-kNN bk=3x refine=2                                 5_402.54         4.06     5_406.60       0.9999          1.0000            1.0000        89.65
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
GPU-Exhaustive (ground truth)                             56.29    15_329.41    15_385.70       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   4_572.38     2_497.96     7_070.34       1.0000          1.0000            1.0000       649.98
GPU-kNN bk=1x refine=0                                   997.69         9.71     1_007.40       0.9721          1.0025            1.0000       118.26
GPU-kNN bk=1x refine=1                                 1_066.09         9.16     1_075.25       0.9791          1.0018            1.0000       118.26
GPU-kNN bk=1x refine=2                                 1_258.33         8.80     1_267.13       0.9797          1.0017            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_455.86         9.20     1_465.06       0.9983          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=1                                 2_269.33         9.66     2_278.99       0.9990          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 3_082.60         9.47     3_092.07       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 2_252.17         9.22     2_261.39       0.9994          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=1                                 4_091.98        10.99     4_102.97       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_967.73         9.15     5_976.88       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                            142.17    26_455.97    26_598.13       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                   5_734.52     3_888.29     9_622.82       1.0000          1.0000            1.0000       900.05
GPU-kNN bk=1x refine=0                                 1_474.32         9.98     1_484.30       0.9714          1.0025            1.0000       179.29
GPU-kNN bk=1x refine=1                                 2_093.76        10.20     2_103.97       0.9785          1.0018            1.0000       179.29
GPU-kNN bk=1x refine=2                                 2_766.07         8.64     2_774.72       0.9791          1.0018            1.0000       179.29
GPU-kNN bk=2x refine=0                                 2_197.79         9.45     2_207.24       0.9982          1.0001            1.0000       179.29
GPU-kNN bk=2x refine=1                                 5_009.02         9.15     5_018.16       0.9990          1.0001            1.0000       179.29
GPU-kNN bk=2x refine=2                                 7_883.69         8.72     7_892.41       0.9990          1.0001            1.0000       179.29
GPU-kNN bk=3x refine=0                                 3_703.74         8.39     3_712.13       0.9993          1.0000            1.0000       179.29
GPU-kNN bk=3x refine=1                                 9_875.07         9.05     9_884.12       0.9998          1.0000            1.0000       179.29
GPU-kNN bk=3x refine=2                                16_545.44        11.49    16_556.93       0.9999          1.0000            1.0000       179.29
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
GPU-Exhaustive (ground truth)                            115.86    59_096.66    59_212.53       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  10_460.65     6_132.04    16_592.69       0.9999          1.0000            1.0000      1696.23
GPU-kNN bk=1x refine=0                                 2_028.80        17.95     2_046.75       0.9605          1.0037            1.0000       236.51
GPU-kNN bk=1x refine=1                                 2_476.40        17.32     2_493.72       0.9691          1.0027            1.0000       236.51
GPU-kNN bk=1x refine=2                                 3_189.42        21.55     3_210.96       0.9700          1.0026            1.0000       236.51
GPU-kNN bk=2x refine=0                                 3_124.60        19.77     3_144.36       0.9974          1.0002            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_849.14        17.42     5_866.56       0.9984          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_711.77        16.59     8_728.37       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 5_055.39        18.03     5_073.42       0.9992          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                11_290.73        19.50    11_310.23       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                17_580.87        18.80    17_599.67       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            310.31   103_123.24   103_433.55       1.0000          1.0000            1.0000       488.28
CPU-NNDescent (k=15)                                  13_695.83     9_410.86    23_106.69       0.9999          1.0000            1.0000      1940.39
GPU-kNN bk=1x refine=0                                 3_130.41        20.26     3_150.66       0.9599          1.0037            1.0000       358.58
GPU-kNN bk=1x refine=1                                 4_903.61        16.42     4_920.03       0.9687          1.0028            1.0000       358.58
GPU-kNN bk=1x refine=2                                 6_717.26        15.80     6_733.06       0.9696          1.0027            1.0000       358.58
GPU-kNN bk=2x refine=0                                 4_598.75        17.55     4_616.29       0.9974          1.0002            1.0000       358.58
GPU-kNN bk=2x refine=1                                12_802.39        16.29    12_818.68       0.9984          1.0001            1.0000       358.58
GPU-kNN bk=2x refine=2                                20_198.51        22.78    20_221.28       0.9985          1.0001            1.0000       358.58
GPU-kNN bk=3x refine=0                                 8_246.18        18.20     8_264.38       0.9992          1.0001            1.0000       358.58
GPU-kNN bk=3x refine=1                                27_067.15        17.51    27_084.67       0.9998          1.0000            1.0000       358.58
GPU-kNN bk=3x refine=2                                44_167.68        16.88    44_184.56       0.9998          1.0000            1.0000       358.58
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
GPU-Exhaustive (ground truth)                            294.53   345_158.44   345_452.96       1.0000          1.0000            1.0000       610.35
CPU-NNDescent (k=15)                                  32_632.77    19_549.34    52_182.12       0.9999          1.0000            1.0000      3890.93
GPU-kNN bk=1x refine=0                                 5_324.56        62.38     5_386.94       0.9393          1.0060            1.0000       591.28
GPU-kNN bk=1x refine=1                                 7_257.84        47.89     7_305.73       0.9504          1.0047            1.0000       591.28
GPU-kNN bk=1x refine=2                                 9_521.80        49.37     9_571.17       0.9520          1.0045            1.0000       591.28
GPU-kNN bk=2x refine=0                                 8_780.00        53.20     8_833.20       0.9958          1.0003            1.0000       591.28
GPU-kNN bk=2x refine=1                                19_207.86        55.55    19_263.41       0.9973          1.0002            1.0000       591.28
GPU-kNN bk=2x refine=2                                27_701.22        51.49    27_752.71       0.9974          1.0002            1.0000       591.28
GPU-kNN bk=3x refine=0                                13_482.45        52.02    13_534.47       0.9989          1.0001            1.0000       591.28
GPU-kNN bk=3x refine=1                                35_673.29        60.19    35_733.48       0.9996          1.0000            1.0000       591.28
GPU-kNN bk=3x refine=2                                62_587.79        58.50    62_646.29       0.9996          1.0000            1.0000       591.28
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
NNDescent-GPU (unbatched)                 -         563.51       0.9949        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         467.53       0.9948        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1070.82       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1059.74       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1035.48       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1170.80       0.9992        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -         822.92       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         732.96       0.9947        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1560.31       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1502.61       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1547.10       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1625.14       0.9993        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1123.13       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1024.22       0.9919        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2175.50       0.9987        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2067.47       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2014.07       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2097.92       0.9987        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.11.0 (commit v0.10.1-85-ge785e27), run on 2026-10-12.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
