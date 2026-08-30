## Self-kNN graph benchmarks

Every index in this crate can produce a full self-kNN graph by querying itself,
but a handful build one as a by-product of construction and can hand it back
without a search pass at all. That is the cheap path, and it is what downstream
single-cell work (BBKNN, MNN, UMAP, Leiden) actually wants. This page collects
those paths and the searched ones they compete against.

```bash
# CPU NN-Descent: build, self-beam and raw extract in one table
cargo run --example gridsearch_nndescent --release

# GPU NN-Descent / CAGRA: extract vs self-beam across build_k and refinement
cargo run --example knn_comparison_cagra --features gpu --release

# Clustered GPU NN-Descent: cluster-count sweep for datasets past the binding limit
cargo run --example gridsearch_clustered_nndescent --features gpu --release
```

## Table of Contents

- [The three paths](#the-three-paths)
- [CPU NN-Descent](#cpu-nn-descent)
- [GPU NN-Descent and CAGRA](#gpu-nn-descent-and-cagra)
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

The `(extract)` row adds the trivial self-edge back before scoring. The
search-based rows and the ground truth both count a point as its own nearest
neighbour at distance zero; a kNN graph does not store that edge, so without the
fix-up the extract row would lose a fixed `1/k` against everything else.

**Tunable parameters:** see
[the standard benchmarks](benchmarks_standard.md#nndescent). The one that
matters most here is the graph degree `k`, which is the ceiling on what
`(extract)` can return.

<details>
<summary><b>CPU NN-Descent - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.34       656.60       667.94       1.0000          1.0000        18.31
Exhaustive (self)                                         11.34     6_760.95     6_772.29       1.0000          1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_760.87        47.20     2_808.06       0.9989          1.0002       118.81
NNDescent-k:auto-nt4-dp0 (self)                        2_760.87       425.96     3_186.83       0.9997          1.0000       118.81
NNDescent-k:auto-nt4-dp0 (extract)                     2_760.87         3.13     2_763.99       0.9996          1.0000       118.81
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_503.37        54.43     2_557.80       0.9987          1.0003       131.18
NNDescent-k:auto-nt8-dp0 (self)                        2_503.37       415.55     2_918.92       0.9997          1.0000       131.18
NNDescent-k:auto-nt8-dp0 (extract)                     2_503.37         2.94     2_506.31       0.9996          1.0000       131.18
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_471.50        64.70     2_536.20       0.9995          1.0001       131.68
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_471.50        85.33     2_556.82       0.9996          1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_471.50        45.70     2_517.20       0.9988          1.0003       131.68
NNDescent-k:auto-nt:auto-dp0 (self)                    2_471.50       423.61     2_895.11       0.9997          1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_471.50         2.99     2_474.49       0.9996          1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_629.10        56.43     2_685.53       0.9989          1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_629.10       537.83     3_166.93       0.9991          1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_629.10         2.32     2_631.42       0.9220          1.0197       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_610.18        58.92     2_669.10       0.9992          1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_610.18       575.72     3_185.90       0.9993          1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_610.18         2.59     2_612.77       0.9225          1.0496       131.68
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_561.44        62.04     2_623.47       0.9992          1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (self)                    2_561.44       613.81     3_175.24       0.9993          1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_561.44         2.54     2_563.98       0.9252          1.0974       131.68
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.16       660.63       671.80       1.0000          1.0000        18.31
Exhaustive (self)                                         11.16     6_614.10     6_625.27       1.0000          1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_757.73        64.10     1_821.83       0.9982          1.0002       115.55
NNDescent-k:auto-nt4-dp0 (self)                        1_757.73       548.56     2_306.29       0.9994          1.0000       115.55
NNDescent-k:auto-nt4-dp0 (extract)                     1_757.73         3.05     1_760.78       0.9994          1.0000       115.55
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_524.36        64.12     1_588.48       0.9981          1.0001       124.67
NNDescent-k:auto-nt8-dp0 (self)                        1_524.36       585.85     2_110.21       0.9994          1.0000       124.67
NNDescent-k:auto-nt8-dp0 (extract)                     1_524.36         2.84     1_527.20       0.9994          1.0000       124.67
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_566.65        82.79     1_649.44       0.9988          1.0001       134.41
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_566.65       107.30     1_673.95       0.9991          1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_566.65        58.51     1_625.16       0.9981          1.0001       134.41
NNDescent-k:auto-nt:auto-dp0 (self)                    1_566.65       551.11     2_117.76       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_566.65         2.57     1_569.22       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_808.82        68.42     1_877.24       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_808.82       628.02     2_436.84       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_808.82         2.38     1_811.20       0.9452          1.0061       134.41
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_762.83        64.25     1_827.08       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_762.83       628.26     2_391.10       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_762.83         2.84     1_765.68       0.9558          1.0056       134.41
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_736.75        65.87     1_802.62       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp1 (self)                    1_736.75       635.78     2_372.53       0.9994          1.0000       134.41
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_736.75         2.96     1_739.71       0.9858          1.0018       134.41
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>CPU NN-Descent - Euclidean (NN embeddings; 128 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.44     1_211.34     1_259.78       1.0000          1.0000        73.24
Exhaustive (self)                                         48.44    12_395.04    12_443.48       1.0000          1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_511.43       115.04     2_626.47       1.0000          1.0000       230.71
NNDescent-k:auto-nt4-dp0 (self)                        2_511.43     1_083.88     3_595.31       0.9999          1.0000       230.71
NNDescent-k:auto-nt4-dp0 (extract)                     2_511.43         2.92     2_514.34       0.9999          1.0000       230.71
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_460.08       122.94     2_583.02       0.9999          1.0001       245.14
NNDescent-k:auto-nt8-dp0 (self)                        2_460.08     1_087.17     3_547.25       0.9999          1.0000       245.14
NNDescent-k:auto-nt8-dp0 (extract)                     2_460.08         2.60     2_462.68       0.9999          1.0000       245.14
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_743.04       153.79     2_896.83       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_743.04       193.15     2_936.19       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_743.04       117.40     2_860.43       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (self)                    2_743.04     1_087.03     3_830.07       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_743.04         2.62     2_745.66       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         3_031.80       124.01     3_155.80       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (self)                 3_031.80     1_151.96     4_183.76       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.25 (extract)              3_031.80         2.75     3_034.55       0.9970          1.0005       273.50
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_968.51       115.56     3_084.07       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_968.51     1_136.48     4_104.99       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_968.51         2.77     2_971.28       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_943.75       115.36     3_059.11       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (self)                    2_943.75     1_105.41     4_049.16       1.0000          1.0000       273.50
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_943.75         2.56     2_946.31       1.0000          1.0000       273.50
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### GPU NN-Descent and CAGRA

The same algorithm with the local join on the GPU, followed by CAGRA's
rank-based detour pruning into a navigational graph. Both the extract and the
self-beam path come out of it. The sweep varies `build_k` (the internal degree
before pruning, as a multiple of `k`) and `refine_knn` (2-hop refinement sweeps
after convergence), with a CPU NN-Descent row and a GPU exhaustive ground truth
for reference.

Dimensions are kept deliberately low here to mimic single-cell embeddings.

**Tunable parameters:**

- *`build_k`*: Internal NN-Descent degree before CAGRA pruning, defaults to
  `1.5 * k`. More edges for the prune to choose from, at build-time cost.
- *`refine_knn`*: 2-hop refinement sweeps after convergence. Each sweep
  evaluates all neighbours-of-neighbours and merges improvements. This is the
  knob aimed at the extracted graph rather than at beam-search recall.
- *`n_trees`*: Random-partition trees for the forest initialisation. Defaults to
  `5 + n^0.25`, capped at 20.
- *`delta`*: Convergence threshold on the improving fraction.

<details>
<summary><b>kNN generation (250k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             25.86     8_904.84     8_930.70       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   2_699.66     1_167.47     3_867.13       1.0000          1.0000       226.85
GPU-NND bk=1x refine=0 (extract)                         675.93         3.78       679.71       0.9135          1.0838       102.04
GPU-NND bk=1x refine=0 (self-beam)                       675.93     1_077.65     1_753.59       0.9901          1.0007       102.04
GPU-NND bk=1x refine=1 (extract)                         570.43         2.98       573.41       0.9191          1.0832       102.04
GPU-NND bk=1x refine=1 (self-beam)                       570.43     1_070.14     1_640.57       0.9904          1.0007       102.04
GPU-NND bk=1x refine=2 (extract)                         559.23         3.72       562.95       0.9195          1.0832       102.04
GPU-NND bk=1x refine=2 (self-beam)                       559.23     1_060.58     1_619.80       0.9905          1.0006       102.04
GPU-NND bk=2x refine=0 (extract)                         901.78         2.64       904.42       0.9304          1.0821       102.04
GPU-NND bk=2x refine=0 (self-beam)                       901.78     1_048.87     1_950.65       0.9956          1.0002       102.04
GPU-NND bk=2x refine=1 (extract)                         960.79         3.10       963.88       0.9324          1.0819       102.04
GPU-NND bk=2x refine=1 (self-beam)                       960.79     1_049.54     2_010.32       0.9960          1.0001       102.04
GPU-NND bk=2x refine=2 (extract)                       1_030.53         2.65     1_033.19       0.9324          1.0819       102.04
GPU-NND bk=2x refine=2 (self-beam)                     1_030.53     1_053.29     2_083.82       0.9960          1.0001       102.04
GPU-NND bk=3x refine=0 (extract)                       1_299.89         2.95     1_302.84       0.9319          1.0820       102.04
GPU-NND bk=3x refine=0 (self-beam)                     1_299.89     1_052.96     2_352.86       0.9961          1.0001       102.04
GPU-NND bk=3x refine=1 (extract)                       1_486.08         2.72     1_488.80       0.9332          1.0819       102.04
GPU-NND bk=3x refine=1 (self-beam)                     1_486.08     1_066.87     2_552.95       0.9964          1.0001       102.04
GPU-NND bk=3x refine=2 (extract)                       1_676.32         2.70     1_679.02       0.9332          1.0819       102.04
GPU-NND bk=3x refine=2 (self-beam)                     1_676.32     1_051.62     2_727.94       0.9964          1.0001       102.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>kNN generation (250k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             56.81    12_744.94    12_801.75       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   3_520.69     1_630.78     5_151.48       1.0000          1.0000       301.88
GPU-NND bk=1x refine=0 (extract)                         921.12         3.46       924.59       0.9134          1.0837       132.56
GPU-NND bk=1x refine=0 (self-beam)                       921.12     1_165.48     2_086.61       0.9901          1.0007       132.56
GPU-NND bk=1x refine=1 (extract)                         989.92         2.80       992.71       0.9190          1.0831       132.56
GPU-NND bk=1x refine=1 (self-beam)                       989.92     1_139.63     2_129.54       0.9904          1.0007       132.56
GPU-NND bk=1x refine=2 (extract)                       1_114.67         2.70     1_117.37       0.9194          1.0831       132.56
GPU-NND bk=1x refine=2 (self-beam)                     1_114.67     1_124.71     2_239.38       0.9904          1.0007       132.56
GPU-NND bk=2x refine=0 (extract)                       1_386.25         2.72     1_388.98       0.9304          1.0820       132.56
GPU-NND bk=2x refine=0 (self-beam)                     1_386.25     1_145.05     2_531.30       0.9956          1.0002       132.56
GPU-NND bk=2x refine=1 (extract)                       2_181.89         2.68     2_184.58       0.9324          1.0818       132.56
GPU-NND bk=2x refine=1 (self-beam)                     2_181.89     1_122.82     3_304.71       0.9960          1.0001       132.56
GPU-NND bk=2x refine=2 (extract)                       2_973.48         2.45     2_975.93       0.9324          1.0818       132.56
GPU-NND bk=2x refine=2 (self-beam)                     2_973.48     1_121.15     4_094.62       0.9960          1.0001       132.56
GPU-NND bk=3x refine=0 (extract)                       2_194.74         2.98     2_197.72       0.9318          1.0819       132.56
GPU-NND bk=3x refine=0 (self-beam)                     2_194.74     1_124.61     3_319.35       0.9961          1.0001       132.56
GPU-NND bk=3x refine=1 (extract)                       4_057.55         3.72     4_061.27       0.9332          1.0818       132.56
GPU-NND bk=3x refine=1 (self-beam)                     4_057.55     1_118.60     5_176.15       0.9964          1.0001       132.56
GPU-NND bk=3x refine=2 (extract)                       5_978.79         3.87     5_982.66       0.9332          1.0818       132.56
GPU-NND bk=3x refine=2 (self-beam)                     5_978.79     1_120.68     7_099.47       0.9964          1.0001       132.56
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             55.54    34_949.36    35_004.90       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   5_651.45     2_792.51     8_443.96       0.9999          1.0000       441.69
GPU-NND bk=1x refine=0 (extract)                       1_189.15         7.21     1_196.36       0.9037          1.0846       204.09
GPU-NND bk=1x refine=0 (self-beam)                     1_189.15     2_173.13     3_362.28       0.9836          1.0012       204.09
GPU-NND bk=1x refine=1 (extract)                       1_219.67         9.00     1_228.67       0.9110          1.0838       204.09
GPU-NND bk=1x refine=1 (self-beam)                     1_219.67     2_174.79     3_394.47       0.9842          1.0011       204.09
GPU-NND bk=1x refine=2 (extract)                       1_386.90         5.80     1_392.70       0.9116          1.0837       204.09
GPU-NND bk=1x refine=2 (self-beam)                     1_386.90     2_159.15     3_546.04       0.9843          1.0011       204.09
GPU-NND bk=2x refine=0 (extract)                       1_826.79         5.28     1_832.07       0.9295          1.0819       204.09
GPU-NND bk=2x refine=0 (self-beam)                     1_826.79     2_130.30     3_957.09       0.9935          1.0003       204.09
GPU-NND bk=2x refine=1 (extract)                       2_643.78         5.04     2_648.82       0.9318          1.0817       204.09
GPU-NND bk=2x refine=1 (self-beam)                     2_643.78     2_133.66     4_777.43       0.9940          1.0002       204.09
GPU-NND bk=2x refine=2 (extract)                       3_458.69         4.95     3_463.64       0.9319          1.0817       204.09
GPU-NND bk=2x refine=2 (self-beam)                     3_458.69     2_134.71     5_593.41       0.9941          1.0002       204.09
GPU-NND bk=3x refine=0 (extract)                       2_748.61         6.69     2_755.30       0.9316          1.0817       204.09
GPU-NND bk=3x refine=0 (self-beam)                     2_748.61     2_139.65     4_888.26       0.9944          1.0002       204.09
GPU-NND bk=3x refine=1 (extract)                       4_554.16         6.37     4_560.53       0.9331          1.0816       204.09
GPU-NND bk=3x refine=1 (self-beam)                     4_554.16     2_138.36     6_692.52       0.9947          1.0002       204.09
GPU-NND bk=3x refine=2 (extract)                       6_374.06         6.52     6_380.57       0.9331          1.0816       204.09
GPU-NND bk=3x refine=2 (self-beam)                     6_374.06     2_136.72     8_510.78       0.9948          1.0002       204.09
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>kNN generation (500k samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            149.53    50_686.11    50_835.64       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                   7_374.44     4_100.32    11_474.76       0.9999          1.0000       592.26
GPU-NND bk=1x refine=0 (extract)                       1_660.60         8.44     1_669.05       0.9034          1.0845       265.12
GPU-NND bk=1x refine=0 (self-beam)                     1_660.60     2_336.88     3_997.48       0.9834          1.0012       265.12
GPU-NND bk=1x refine=1 (extract)                       2_236.66         6.70     2_243.36       0.9109          1.0837       265.12
GPU-NND bk=1x refine=1 (self-beam)                     2_236.66     2_312.31     4_548.98       0.9840          1.0011       265.12
GPU-NND bk=1x refine=2 (extract)                       2_879.12         5.35     2_884.47       0.9115          1.0836       265.12
GPU-NND bk=1x refine=2 (self-beam)                     2_879.12     2_293.88     5_173.00       0.9841          1.0011       265.12
GPU-NND bk=2x refine=0 (extract)                       2_753.68         5.50     2_759.17       0.9294          1.0818       265.12
GPU-NND bk=2x refine=0 (self-beam)                     2_753.68     2_267.49     5_021.17       0.9935          1.0003       265.12
GPU-NND bk=2x refine=1 (extract)                       5_903.27         5.23     5_908.50       0.9317          1.0816       265.12
GPU-NND bk=2x refine=1 (self-beam)                     5_903.27     2_270.06     8_173.34       0.9940          1.0002       265.12
GPU-NND bk=2x refine=2 (extract)                       9_370.30         5.33     9_375.63       0.9318          1.0816       265.12
GPU-NND bk=2x refine=2 (self-beam)                     9_370.30     2_277.11    11_647.41       0.9940          1.0002       265.12
GPU-NND bk=3x refine=0 (extract)                       4_571.18         7.16     4_578.34       0.9316          1.0816       265.12
GPU-NND bk=3x refine=0 (self-beam)                     4_571.18     2_278.94     6_850.12       0.9943          1.0002       265.12
GPU-NND bk=3x refine=1 (extract)                      11_735.63         5.64    11_741.27       0.9331          1.0815       265.12
GPU-NND bk=3x refine=1 (self-beam)                    11_735.63     2_272.47    14_008.10       0.9947          1.0002       265.12
GPU-NND bk=3x refine=2 (extract)                      19_222.07         5.77    19_227.84       0.9331          1.0815       265.12
GPU-NND bk=3x refine=2 (self-beam)                    19_222.07     2_271.83    21_493.90       0.9947          1.0002       265.12
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            122.78   140_056.58   140_179.37       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  12_561.14     6_767.56    19_328.70       0.9998          1.0000       891.37
GPU-NND bk=1x refine=0 (extract)                       2_334.62        15.23     2_349.84       0.8924          1.0857       408.17
GPU-NND bk=1x refine=0 (self-beam)                     2_334.62     4_427.77     6_762.39       0.9757          1.0019       408.17
GPU-NND bk=1x refine=1 (extract)                       2_808.89        18.83     2_827.72       0.9013          1.0846       408.17
GPU-NND bk=1x refine=1 (self-beam)                     2_808.89     4_412.91     7_221.80       0.9765          1.0018       408.17
GPU-NND bk=1x refine=2 (extract)                       3_311.08        14.25     3_325.33       0.9022          1.0845       408.17
GPU-NND bk=1x refine=2 (self-beam)                     3_311.08     4_411.87     7_722.96       0.9766          1.0018       408.17
GPU-NND bk=2x refine=0 (extract)                       3_947.35        14.86     3_962.21       0.9283          1.0818       408.17
GPU-NND bk=2x refine=0 (self-beam)                     3_947.35     4_431.82     8_379.17       0.9911          1.0004       408.17
GPU-NND bk=2x refine=1 (extract)                       6_606.91        13.34     6_620.25       0.9310          1.0816       408.17
GPU-NND bk=2x refine=1 (self-beam)                     6_606.91     4_389.00    10_995.91       0.9917          1.0003       408.17
GPU-NND bk=2x refine=2 (extract)                       9_317.84        13.42     9_331.26       0.9311          1.0816       408.17
GPU-NND bk=2x refine=2 (self-beam)                     9_317.84     4_412.89    13_730.73       0.9918          1.0003       408.17
GPU-NND bk=3x refine=0 (extract)                       5_864.70        15.70     5_880.39       0.9313          1.0815       408.17
GPU-NND bk=3x refine=0 (self-beam)                     5_864.70     4_437.44    10_302.14       0.9925          1.0003       408.17
GPU-NND bk=3x refine=1 (extract)                      11_967.05        11.08    11_978.12       0.9330          1.0814       408.17
GPU-NND bk=3x refine=1 (self-beam)                    11_967.05     4_392.78    16_359.82       0.9930          1.0002       408.17
GPU-NND bk=3x refine=2 (extract)                      18_173.79        12.83    18_186.62       0.9330          1.0814       408.17
GPU-NND bk=3x refine=2 (self-beam)                    18_173.79     4_395.28    22_569.06       0.9930          1.0002       408.17
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>kNN generation (1m samples; 64 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            302.27   205_177.20   205_479.47       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  16_430.19     9_801.24    26_231.43       0.9998          1.0000      1217.51
GPU-NND bk=1x refine=0 (extract)                       3_286.92        16.01     3_302.93       0.8925          1.0856       530.24
GPU-NND bk=1x refine=0 (self-beam)                     3_286.92     4_755.33     8_042.25       0.9757          1.0018       530.24
GPU-NND bk=1x refine=1 (extract)                       5_004.59        19.36     5_023.95       0.9013          1.0845       530.24
GPU-NND bk=1x refine=1 (self-beam)                     5_004.59     4_727.60     9_732.19       0.9765          1.0018       530.24
GPU-NND bk=1x refine=2 (extract)                       6_839.51        12.42     6_851.93       0.9023          1.0844       530.24
GPU-NND bk=1x refine=2 (self-beam)                     6_839.51     4_707.23    11_546.74       0.9766          1.0017       530.24
GPU-NND bk=2x refine=0 (extract)                       5_849.77        12.78     5_862.55       0.9283          1.0817       530.24
GPU-NND bk=2x refine=0 (self-beam)                     5_849.77     4_674.10    10_523.87       0.9911          1.0004       530.24
GPU-NND bk=2x refine=1 (extract)                      14_438.19        11.32    14_449.51       0.9310          1.0815       530.24
GPU-NND bk=2x refine=1 (self-beam)                    14_438.19     4_664.60    19_102.79       0.9917          1.0003       530.24
GPU-NND bk=2x refine=2 (extract)                      23_576.03         9.66    23_585.68       0.9311          1.0815       530.24
GPU-NND bk=2x refine=2 (self-beam)                    23_576.03     4_661.30    28_237.33       0.9917          1.0003       530.24
GPU-NND bk=3x refine=0 (extract)                       9_666.91        14.91     9_681.82       0.9314          1.0814       530.24
GPU-NND bk=3x refine=0 (self-beam)                     9_666.91     4_705.42    14_372.34       0.9925          1.0002       530.24
GPU-NND bk=3x refine=1 (extract)                      29_959.94        12.89    29_972.83       0.9329          1.0813       530.24
GPU-NND bk=3x refine=1 (self-beam)                    29_959.94     4_668.40    34_628.33       0.9929          1.0002       530.24
GPU-NND bk=3x refine=2 (extract)                      50_193.01        15.46    50_208.47       0.9330          1.0813       530.24
GPU-NND bk=3x refine=2 (self-beam)                    50_193.01     4_695.51    54_888.53       0.9929          1.0002       530.24
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>kNN generation (2.5m samples; 32 dimensions)</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                            289.93   841_727.48   842_017.42       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  34_106.73    20_728.82    54_835.55       0.9996          1.0000      2210.45
GPU-NND bk=1x refine=0 (extract)                       6_179.51        86.79     6_266.30       0.8708          1.0882      1020.43
GPU-NND bk=1x refine=0 (self-beam)                     6_179.51    11_430.20    17_609.72       0.9607          1.0033      1020.43
GPU-NND bk=1x refine=1 (extract)                       8_108.76        47.12     8_155.88       0.8820          1.0867      1020.43
GPU-NND bk=1x refine=1 (self-beam)                     8_108.76    11_488.20    19_596.95       0.9619          1.0031      1020.43
GPU-NND bk=1x refine=2 (extract)                      10_391.03        31.14    10_422.17       0.8836          1.0865      1020.43
GPU-NND bk=1x refine=2 (self-beam)                    10_391.03    11_291.50    21_682.53       0.9622          1.0031      1020.43
GPU-NND bk=2x refine=0 (extract)                      11_102.67        29.73    11_132.40       0.9263          1.0818      1020.43
GPU-NND bk=2x refine=0 (self-beam)                    11_102.67    11_183.66    22_286.33       0.9876          1.0005      1020.43
GPU-NND bk=2x refine=1 (extract)                      21_178.38        32.66    21_211.04       0.9295          1.0815      1020.43
GPU-NND bk=2x refine=1 (self-beam)                    21_178.38    11_082.59    32_260.97       0.9884          1.0005      1020.43
GPU-NND bk=2x refine=2 (extract)                      31_598.05        32.29    31_630.33       0.9297          1.0815      1020.43
GPU-NND bk=2x refine=2 (self-beam)                    31_598.05    11_097.38    42_695.42       0.9885          1.0005      1020.43
GPU-NND bk=3x refine=0 (extract)                      16_166.40        33.92    16_200.33       0.9310          1.0813      1020.43
GPU-NND bk=3x refine=0 (self-beam)                    16_166.40    11_035.93    27_202.34       0.9900          1.0003      1020.43
GPU-NND bk=3x refine=1 (extract)                      39_763.70        30.85    39_794.55       0.9327          1.0812      1020.43
GPU-NND bk=3x refine=1 (self-beam)                    39_763.70    11_046.23    50_809.93       0.9906          1.0003      1020.43
GPU-NND bk=3x refine=2 (extract)                      63_494.74        32.75    63_527.50       0.9328          1.0812      1020.43
GPU-NND bk=3x refine=2 (self-beam)                    63_494.74    11_027.20    74_521.95       0.9906          1.0003      1020.43
-----------------------------------------------------------------------------------------------------------------------------------

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
CAGRA comparison uses, which is why the sizes stop lower on this table.

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
NNDescent-GPU (unbatched)                 -         693.65       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         570.08       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1253.76       0.9986        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1428.52       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1423.50       0.9984        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1807.06       0.9985        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1044.35       0.9929        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         867.38       0.9928        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1797.36       0.9985        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1893.25       0.9982        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        2013.17       0.9983        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        2254.98       0.9983        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1343.09       0.9896        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1        1214.72       0.9896        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2534.19       0.9978        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2460.93       0.9975        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2510.89       0.9977        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2780.98       0.9976        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
