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
Exhaustive (query)                                        11.18       638.95       650.13       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.18     6_314.24     6_325.42       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_754.23        43.30     2_797.52       0.9989          1.0002            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (self)                        2_754.23       410.56     3_164.79       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt4-dp0 (extract)                     2_754.23         3.35     2_757.58       0.9997          1.0000            1.0000       118.81
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_464.70        43.01     2_507.72       0.9988          1.0003            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (self)                        2_464.70       401.02     2_865.72       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt8-dp0 (extract)                     2_464.70         3.73     2_468.43       0.9997          1.0000            1.0000       131.18
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_361.80        60.91     2_422.70       0.9995          1.0001            1.0000       131.68
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_361.80        79.70     2_441.50       0.9996          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_361.80        43.51     2_405.31       0.9988          1.0003            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (self)                    2_361.80       412.95     2_774.75       0.9998          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_361.80         4.24     2_366.04       0.9997          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_588.20        54.84     2_643.04       0.9989          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_588.20       519.34     3_107.54       0.9991          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_588.20         3.20     2_591.40       0.9221          1.0197            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_528.24        58.37     2_586.61       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_528.24       540.02     3_068.27       0.9994          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_528.24         3.62     2_531.86       0.9225          1.0496            1.0000       131.68
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_506.11        63.26     2_569.37       0.9992          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (self)                    2_506.11       606.23     3_112.34       0.9993          1.0000            1.0000       131.68
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_506.11         3.76     2_509.87       0.9252          1.0974            1.0000       131.68
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
Exhaustive (query)                                        11.58       645.36       656.95       1.0000          1.0000            1.0000        18.31
Exhaustive (self)                                         11.58     6_504.01     6_515.59       1.0000          1.0000            1.0000        18.31
NNDescent-k:auto-nt4-s:auto-dp0 (query)                1_770.62        59.12     1_829.74       0.9989          1.0001            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (self)                        1_770.62       557.59     2_328.21       1.0000          1.0000            1.0000       119.05
NNDescent-k:auto-nt4-dp0 (extract)                     1_770.62         4.98     1_775.60       1.0000          1.0000            1.0000       119.05
NNDescent-k:auto-nt8-s:auto-dp0 (query)                1_525.87        56.88     1_582.75       0.9990          1.0001            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (self)                        1_525.87       522.52     2_048.39       1.0000          1.0000            1.0000       131.67
NNDescent-k:auto-nt8-dp0 (extract)                     1_525.87         3.80     1_529.67       1.0000          1.0000            1.0000       131.67
NNDescent-k:auto-nt:auto-s75-dp0 (query)               1_538.97        81.39     1_620.36       0.9994          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s100-dp0 (query)              1_538.97       104.69     1_643.67       0.9997          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            1_538.97        57.26     1_596.23       0.9989          1.0001            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (self)                    1_538.97       551.09     2_090.07       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0 (extract)                 1_538.97         3.60     1_542.57       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         1_764.09        60.66     1_824.75       0.9998          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (self)                 1_764.09       567.09     2_331.18       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.25 (extract)              1_764.09         3.25     1_767.34       0.9455          1.0062            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          1_707.05        62.60     1_769.65       0.9999          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (self)                  1_707.05       582.69     2_289.74       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp0.5 (extract)               1_707.05         3.01     1_710.06       0.9575          1.0055            1.0000       132.16
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            1_693.33        64.71     1_758.04       0.9998          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (self)                    1_693.33       606.02     2_299.36       1.0000          1.0000            1.0000       132.16
NNDescent-k:auto-nt:auto-dp1 (extract)                 1_693.33         3.69     1_697.03       0.9874          1.0017            1.0000       132.16
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
Exhaustive (query)                                        49.44     1_241.92     1_291.36       1.0000          1.0000            1.0000        73.24
Exhaustive (self)                                         49.44    12_525.78    12_575.22       1.0000          1.0000            1.0000        73.24
NNDescent-k:auto-nt4-s:auto-dp0 (query)                2_386.18       112.87     2_499.05       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (self)                        2_386.18     1_050.03     3_436.21       1.0000          1.0000            1.0000       230.71
NNDescent-k:auto-nt4-dp0 (extract)                     2_386.18         5.20     2_391.38       0.9999          1.0000            1.0000       230.71
NNDescent-k:auto-nt8-s:auto-dp0 (query)                2_335.12       109.37     2_444.49       0.9999          1.0001            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (self)                        2_335.12     1_054.92     3_390.05       1.0000          1.0000            1.0000       245.14
NNDescent-k:auto-nt8-dp0 (extract)                     2_335.12         3.44     2_338.57       1.0000          1.0000            1.0000       245.14
NNDescent-k:auto-nt:auto-s75-dp0 (query)               2_493.40       147.36     2_640.76       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-s100-dp0 (query)              2_493.40       187.31     2_680.70       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-s:auto-dp0 (query)            2_493.40       108.60     2_602.00       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0 (self)                    2_493.40     1_053.81     3_547.20       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0 (extract)                 2_493.40         4.04     2_497.43       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-s:auto-dp0.25 (query)         2_770.71       114.14     2_884.84       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0.25 (self)                 2_770.71     1_104.30     3_875.01       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0.25 (extract)              2_770.71         3.36     2_774.06       0.9970          1.0005            1.0000       273.49
NNDescent-k:auto-nt:auto-s:auto-dp0.5 (query)          2_736.21       113.14     2_849.35       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0.5 (self)                  2_736.21     1_074.02     3_810.23       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp0.5 (extract)               2_736.21         3.70     2_739.91       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-s:auto-dp1 (query)            2_693.10       111.90     2_805.00       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp1 (self)                    2_693.10     1_060.98     3_754.09       1.0000          1.0000            1.0000       273.49
NNDescent-k:auto-nt:auto-dp1 (extract)                 2_693.10         3.48     2_696.58       1.0000          1.0000            1.0000       273.49
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
GPU-Exhaustive (ground truth)                             25.86     8_892.51     8_918.37       1.0000          1.0000            1.0000        30.52
CPU-NNDescent (k=15)                                   2_597.18     1_064.06     3_661.24       1.0000          1.0000            1.0000       224.36
GPU-kNN bk=1x refine=0                                   460.01         5.62       465.63       0.9813          1.0016            1.0000        87.74
GPU-kNN bk=1x refine=1                                   352.89         4.57       357.46       0.9866          1.0011            1.0000        87.74
GPU-kNN bk=1x refine=2                                   371.22         5.54       376.76       0.9870          1.0011            1.0000        87.74
GPU-kNN bk=2x refine=0                                   693.71         4.96       698.67       0.9973          1.0002            1.0000        87.74
GPU-kNN bk=2x refine=1                                   729.10         5.34       734.44       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=2x refine=2                                   819.98         5.86       825.85       0.9992          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=0                                 1_010.73         5.20     1_015.92       0.9987          1.0001            1.0000        87.74
GPU-kNN bk=3x refine=1                                 1_180.98         5.07     1_186.04       0.9999          1.0000            1.0000        87.74
GPU-kNN bk=3x refine=2                                 1_404.78         5.27     1_410.04       0.9999          1.0000            1.0000        87.74
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
GPU-Exhaustive (ground truth)                             58.28    11_470.27    11_528.55       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   3_283.53     1_536.80     4_820.33       1.0000          1.0000            1.0000       302.64
GPU-kNN bk=1x refine=0                                   740.29         4.46       744.75       0.9810          1.0016            1.0000       118.26
GPU-kNN bk=1x refine=1                                   734.39         5.39       739.78       0.9864          1.0011            1.0000       118.26
GPU-kNN bk=1x refine=2                                   903.45         4.58       908.03       0.9867          1.0011            1.0000       118.26
GPU-kNN bk=2x refine=0                                 1_115.30         4.75     1_120.05       0.9972          1.0002            1.0000       118.26
GPU-kNN bk=2x refine=1                                 1_829.50         4.27     1_833.78       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=2x refine=2                                 2_639.31         5.40     2_644.71       0.9991          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=0                                 1_882.82         4.55     1_887.37       0.9986          1.0001            1.0000       118.26
GPU-kNN bk=3x refine=1                                 3_621.92         4.71     3_626.64       0.9999          1.0000            1.0000       118.26
GPU-kNN bk=3x refine=2                                 5_358.99         4.57     5_363.56       0.9999          1.0000            1.0000       118.26
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
GPU-Exhaustive (ground truth)                             56.51    34_931.65    34_988.17       1.0000          1.0000            1.0000        61.04
CPU-NNDescent (k=15)                                   5_484.09     2_641.81     8_125.90       0.9999          1.0000            1.0000       443.70
GPU-kNN bk=1x refine=0                                   841.50        12.27       853.77       0.9721          1.0025            1.0000       175.48
GPU-kNN bk=1x refine=1                                   898.60        10.79       909.40       0.9791          1.0018            1.0000       175.48
GPU-kNN bk=1x refine=2                                 1_046.47        10.16     1_056.63       0.9797          1.0017            1.0000       175.48
GPU-kNN bk=2x refine=0                                 1_466.32         9.23     1_475.55       0.9963          1.0003            1.0000       175.48
GPU-kNN bk=2x refine=1                                 2_228.94        10.04     2_238.97       0.9985          1.0001            1.0000       175.48
GPU-kNN bk=2x refine=2                                 2_961.18        10.00     2_971.18       0.9986          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=0                                 2_274.80         9.50     2_284.29       0.9984          1.0001            1.0000       175.48
GPU-kNN bk=3x refine=1                                 3_893.53        11.17     3_904.70       0.9998          1.0000            1.0000       175.48
GPU-kNN bk=3x refine=2                                 5_576.27         8.88     5_585.15       0.9998          1.0000            1.0000       175.48
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
GPU-Exhaustive (ground truth)                            151.57    45_444.05    45_595.62       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                   6_831.63     3_835.72    10_667.35       0.9999          1.0000            1.0000       597.77
GPU-kNN bk=1x refine=0                                 1_368.42        11.65     1_380.07       0.9714          1.0025            1.0000       236.51
GPU-kNN bk=1x refine=1                                 1_872.27        10.33     1_882.59       0.9785          1.0018            1.0000       236.51
GPU-kNN bk=1x refine=2                                 2_453.46        10.70     2_464.16       0.9791          1.0018            1.0000       236.51
GPU-kNN bk=2x refine=0                                 2_463.08        11.22     2_474.30       0.9962          1.0003            1.0000       236.51
GPU-kNN bk=2x refine=1                                 5_276.15        11.05     5_287.20       0.9985          1.0001            1.0000       236.51
GPU-kNN bk=2x refine=2                                 8_118.57        11.37     8_129.94       0.9986          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=0                                 4_146.67        11.77     4_158.44       0.9983          1.0001            1.0000       236.51
GPU-kNN bk=3x refine=1                                10_546.46        11.49    10_557.95       0.9998          1.0000            1.0000       236.51
GPU-kNN bk=3x refine=2                                16_985.89        10.89    16_996.77       0.9998          1.0000            1.0000       236.51
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
GPU-Exhaustive (ground truth)                            123.21   139_502.39   139_625.60       1.0000          1.0000            1.0000       122.07
CPU-NNDescent (k=15)                                  11_763.47     6_287.89    18_051.36       0.9998          1.0000            1.0000       883.39
GPU-kNN bk=1x refine=0                                 1_640.68        23.14     1_663.83       0.9605          1.0037            1.0000       350.95
GPU-kNN bk=1x refine=1                                 2_166.80        23.18     2_189.98       0.9691          1.0027            1.0000       350.95
GPU-kNN bk=1x refine=2                                 2_685.08        21.94     2_707.02       0.9700          1.0026            1.0000       350.95
GPU-kNN bk=2x refine=0                                 3_251.31        22.77     3_274.08       0.9951          1.0004            1.0000       350.95
GPU-kNN bk=2x refine=1                                 5_822.78        21.88     5_844.67       0.9977          1.0002            1.0000       350.95
GPU-kNN bk=2x refine=2                                 8_398.97        23.07     8_422.04       0.9979          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=0                                 4_851.17        21.28     4_872.45       0.9981          1.0001            1.0000       350.95
GPU-kNN bk=3x refine=1                                10_702.72        20.87    10_723.59       0.9996          1.0000            1.0000       350.95
GPU-kNN bk=3x refine=2                                16_559.04        19.82    16_578.86       0.9997          1.0000            1.0000       350.95
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
GPU-Exhaustive (ground truth)                            315.76   184_886.56   185_202.32       1.0000          1.0000            1.0000       244.14
CPU-NNDescent (k=15)                                  15_288.62     9_332.51    24_621.13       0.9998          1.0000            1.0000      1227.53
GPU-kNN bk=1x refine=0                                 2_667.60        23.33     2_690.93       0.9599          1.0037            1.0000       473.02
GPU-kNN bk=1x refine=1                                 4_439.56        21.31     4_460.88       0.9687          1.0028            1.0000       473.02
GPU-kNN bk=1x refine=2                                 6_325.32        16.70     6_342.02       0.9696          1.0027            1.0000       473.02
GPU-kNN bk=2x refine=0                                 5_306.44        15.76     5_322.20       0.9951          1.0004            1.0000       473.02
GPU-kNN bk=2x refine=1                                13_938.51        16.54    13_955.04       0.9977          1.0002            1.0000       473.02
GPU-kNN bk=2x refine=2                                22_681.59        16.40    22_697.99       0.9978          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=0                                 8_872.90        15.88     8_888.78       0.9981          1.0001            1.0000       473.02
GPU-kNN bk=3x refine=1                                28_541.48        20.72    28_562.20       0.9996          1.0000            1.0000       473.02
GPU-kNN bk=3x refine=2                                48_467.91        17.13    48_485.04       0.9997          1.0000            1.0000       473.02
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
GPU-Exhaustive (ground truth)                            292.61   841_993.00   842_285.61       1.0000          1.0000            1.0000       305.18
CPU-NNDescent (k=15)                                  31_901.35    19_110.17    51_011.52       0.9996          1.0000            1.0000      2238.43
GPU-kNN bk=1x refine=0                                 4_647.40        56.50     4_703.90       0.9393          1.0060            1.0000       877.38
GPU-kNN bk=1x refine=1                                 6_745.65        55.46     6_801.11       0.9504          1.0047            1.0000       877.38
GPU-kNN bk=1x refine=2                                 8_913.39        48.16     8_961.55       0.9520          1.0045            1.0000       877.38
GPU-kNN bk=2x refine=0                                 9_178.21        48.37     9_226.57       0.9932          1.0005            1.0000       877.38
GPU-kNN bk=2x refine=1                                19_688.01        50.75    19_738.76       0.9963          1.0003            1.0000       877.38
GPU-kNN bk=2x refine=2                                30_378.40        54.42    30_432.82       0.9965          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=0                                13_679.54        53.66    13_733.20       0.9977          1.0002            1.0000       877.38
GPU-kNN bk=3x refine=1                                37_714.87        49.09    37_763.96       0.9994          1.0000            1.0000       877.38
GPU-kNN bk=3x refine=2                                61_657.80        52.02    61_709.81       0.9994          1.0000            1.0000       877.38
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
NNDescent-GPU (unbatched)                 -         632.27       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         478.83       0.9940        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1035.94       0.9993        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1072.24       0.9990        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1239.67       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        1673.17       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -         900.63       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=1)            1         769.34       0.9938        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=2)            2        1634.62       0.9992        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=4)            4        1578.85       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=8)            8        1754.85       0.9991        3_750_000 / 3_750_000     100.00
NNDescent-GPU (clustered, c=16)          16        2136.52       0.9991        3_750_000 / 3_750_000     100.00
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
NNDescent-GPU (unbatched)                 -        1085.80       0.9909        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=1)            1         998.96       0.9908        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=2)            2        2045.37       0.9986        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=4)            4        2073.30       0.9983        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=8)            8        2146.19       0.9984        7_500_000 / 7_500_000     100.00
NNDescent-GPU (clustered, c=16)          16        2612.78       0.9985        7_500_000 / 7_500_000     100.00
-------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
