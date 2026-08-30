## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown for the GPU-accelerated code. If you wish to run
the version with GPU-accelerated exhaustive and IVF script, please use:

```bash
cargo run --example gridsearch_gpu --features gpu --release
```

For the CAGRA style search, use:

```bash
cargo run --example gridsearch_cagra --features gpu --release
```

If you wish to run the Navigating Spread-out Graph (NSG) version where the
initial kNN is generated on the GPU, you can test this via:

```bash
cargo run --example gridsearch_nsg_gpu --features gpu --release
```

As with the other benchmarks: index build, query against a 10% subsample with
noise added, and full self-kNN generation, plus the in-memory index size (GPU
memory is not reported). Everything here runs on the wgpu backend; other
backends such as CUDA may do better still.

Looking for the self-kNN-graph paths (NN-Descent extract, self-beam, clustered)?
Those live in [the kNN-graph benchmarks](benchmarks_knn_graph.md).

## Table of Contents

- [GPU exhaustive and IVF](#gpu-accelerated-exhaustive-and-ivf-vs-cpu-exhaustive)
- [Comparison on larger data sets against the CPU](#comparison-against-ivf-cpu)
- [CAGRA style index](#cagra-type-querying)
- [CAGRA index on larger data](#larger-data-sets)
- [NSG with GPU-accelerated kNN generation](#navigating-spread-out-graph-nsg-with-gpu-accelerated-knn-generation)

### GPU-accelerated exhaustive and IVF vs CPU exhaustive

<details>
<summary><b>GPU - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.25       642.05       653.30       1.0000          1.0000        18.31
Exhaustive (self)                                         11.25     6_079.40     6_090.65       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    15.91       416.60       432.50       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     15.91     3_248.28     3_264.19       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               210.83       213.54       424.37       0.9972          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               210.83       140.73       351.56       0.9997          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               210.83       228.32       439.14       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     210.83       858.70     1_069.53       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               168.33       198.86       367.19       0.9991          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               168.33       238.03       406.35       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     168.33       744.54       912.86       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               238.48       202.20       440.69       0.9937          1.0004         1.15
IVF-GPU-nl547-np27 (query)                               238.48       142.36       380.84       0.9986          1.0001         1.15
IVF-GPU-nl547-np33 (query)                               238.48       153.55       392.04       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     238.48       691.05       929.53       1.0000          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU - Cosine (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.71       705.20       716.91       1.0000          1.0000        18.88
Exhaustive (self)                                         11.71     6_808.72     6_820.42       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                    15.79       450.58       466.37       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                     15.79     3_122.52     3_138.31       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               159.69       120.24       279.93       0.9980          1.0001         1.15
IVF-GPU-nl273-np16 (query)                               159.69       142.09       301.78       0.9999          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               159.69       160.44       320.13       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     159.69       916.93     1_076.62       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               205.46       199.51       404.97       0.9991          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               205.46       245.41       450.87       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     205.46       779.11       984.57       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               298.54       200.06       498.60       0.9946          1.0003         1.15
IVF-GPU-nl547-np27 (query)                               298.54       140.05       438.59       0.9988          1.0001         1.15
IVF-GPU-nl547-np33 (query)                               298.54       153.72       452.26       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     298.54       718.95     1_017.49       1.0000          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (Correlated)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.18       632.59       643.77       1.0000          1.0000        18.31
Exhaustive (self)                                         11.18     6_361.68     6_372.86       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    14.06       414.88       428.95       0.9989          1.0000        18.31
GPU-Exhaustive (self)                                     14.06     3_242.42     3_256.48       0.9989          1.0000        18.31
IVF-GPU-nl273-np13 (query)                                99.14       232.96       332.10       0.9989          1.0000         1.15
IVF-GPU-nl273-np16 (query)                                99.14       140.18       239.32       0.9989          1.0000         1.15
IVF-GPU-nl273-np23 (query)                                99.14       151.63       250.77       0.9989          1.0000         1.15
IVF-GPU-nl273 (self)                                      99.14       712.84       811.97       0.9989          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               154.32       208.27       362.59       0.9989          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               154.32       155.30       309.62       0.9989          1.0000         1.15
IVF-GPU-nl387 (self)                                     154.32       656.69       811.01       0.9989          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               230.36       197.37       427.74       0.9989          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               230.36       214.38       444.74       0.9989          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               230.36       154.62       384.98       0.9989          1.0000         1.15
IVF-GPU-nl547 (self)                                     230.36       740.29       970.66       0.9989          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.31       641.31       652.62       1.0000          1.0000        18.31
Exhaustive (self)                                         11.31     6_466.46     6_477.77       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    14.38       406.20       420.59       0.9995          1.0000        18.31
GPU-Exhaustive (self)                                     14.38     3_248.67     3_263.06       0.9995          1.0000        18.31
IVF-GPU-nl273-np13 (query)                                98.24        90.88       189.12       0.9995          1.0000         1.15
IVF-GPU-nl273-np16 (query)                                98.24       130.21       228.44       0.9995          1.0000         1.15
IVF-GPU-nl273-np23 (query)                                98.24       227.24       325.47       0.9995          1.0000         1.15
IVF-GPU-nl273 (self)                                      98.24       793.68       891.92       0.9995          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               168.01       203.32       371.33       0.9995          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               168.01       227.63       395.63       0.9995          1.0000         1.15
IVF-GPU-nl387 (self)                                     168.01       696.45       864.46       0.9995          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               237.85       198.07       435.93       0.9995          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               237.85       214.02       451.87       0.9995          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               237.85       235.62       473.48       0.9995          1.0000         1.15
IVF-GPU-nl547 (self)                                     237.85       746.50       984.35       0.9995          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.17     1_226.21     1_275.38       1.0000          1.0000        73.24
Exhaustive (self)                                         49.17    12_415.62    12_464.79       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    57.11       931.65       988.77       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     57.11     7_499.52     7_556.63       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               356.79       281.93       638.72       0.9998          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               356.79       284.11       640.90       0.9999          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               356.79       331.07       687.87       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     356.79     1_729.75     2_086.55       0.9999          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               537.14       193.32       730.46       0.9999          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               537.14       310.19       847.33       0.9999          1.0000         1.15
IVF-GPU-nl387 (self)                                     537.14     1_558.12     2_095.26       0.9999          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               770.66       151.33       922.00       0.9999          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               770.66       195.69       966.35       0.9999          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               770.66       305.17     1_075.84       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                     770.66     1_445.39     2_216.06       0.9999          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Comparison against IVF CPU

The CPU IVF implementation against the GPU one. The GPU pays a fixed setup
cost, so the sample count is raised to 250k and the dimensionality to 64 or 128
for these runs.

#### With 250k samples and 64 dimensions

<details>
<summary><b>CPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        46.07     1_427.12     1_473.19       1.0000          1.0000        61.04
Exhaustive (self)                                         46.07    23_618.36    23_664.43       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                   716.78       405.53     1_122.31       0.9997          1.0001        61.12
IVF-nl353-np18 (query)                                   716.78       418.80     1_135.58       0.9997          1.0001        61.12
IVF-nl353-np26 (query)                                   716.78       603.53     1_320.31       0.9999          1.0000        61.12
IVF-nl353 (self)                                         716.78     9_086.70     9_803.48       0.9999          1.0000        61.12
IVF-nl500-np22 (query)                                 1_343.71       386.49     1_730.20       0.9998          1.0000        61.16
IVF-nl500-np25 (query)                                 1_343.71       436.30     1_780.01       0.9999          1.0000        61.16
IVF-nl500-np31 (query)                                 1_343.71       510.76     1_854.47       0.9999          1.0000        61.16
IVF-nl500 (self)                                       1_343.71     7_765.63     9_109.33       0.9999          1.0000        61.16
IVF-nl707-np26 (query)                                 2_735.53       338.63     3_074.16       0.9999          1.0000        61.21
IVF-nl707-np35 (query)                                 2_735.53       434.17     3_169.70       0.9999          1.0000        61.21
IVF-nl707-np37 (query)                                 2_735.53       457.12     3_192.64       0.9999          1.0000        61.21
IVF-nl707 (self)                                       2_735.53     6_482.45     9_217.97       0.9999          1.0000        61.21
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        49.85     1_399.20     1_449.04       1.0000          1.0000        61.04
Exhaustive (self)                                         49.85    22_686.16    22_736.01       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    54.48       935.67       990.16       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     54.48    12_611.39    12_665.87       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                               310.46       290.92       601.38       0.9997          1.0001         1.91
IVF-GPU-nl353-np18 (query)                               310.46       276.56       587.02       0.9997          1.0001         1.91
IVF-GPU-nl353-np26 (query)                               310.46       339.72       650.18       0.9999          1.0000         1.91
IVF-GPU-nl353 (self)                                     310.46     2_590.35     2_900.82       0.9999          1.0000         1.91
IVF-GPU-nl500-np22 (query)                               407.62       265.57       673.18       0.9998          1.0000         1.91
IVF-GPU-nl500-np25 (query)                               407.62       301.42       709.03       0.9999          1.0000         1.91
IVF-GPU-nl500-np31 (query)                               407.62       311.67       719.28       0.9999          1.0000         1.91
IVF-GPU-nl500 (self)                                     407.62     2_316.99     2_724.61       0.9999          1.0000         1.91
IVF-GPU-nl707-np26 (query)                               664.59       257.24       921.83       0.9999          1.0000         1.91
IVF-GPU-nl707-np35 (query)                               664.59       312.57       977.16       0.9999          1.0000         1.91
IVF-GPU-nl707-np37 (query)                               664.59       311.13       975.72       0.9999          1.0000         1.91
IVF-GPU-nl707 (self)                                     664.59     2_249.36     2_913.95       0.9999          1.0000         1.91
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        90.61     2_049.70     2_140.31       1.0000          1.0000       122.07
Exhaustive (self)                                         90.61    32_540.27    32_630.88       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                   947.09       816.97     1_764.05       0.9999          1.0000       122.25
IVF-nl353-np18 (query)                                   947.09       863.17     1_810.26       0.9999          1.0000       122.25
IVF-nl353-np26 (query)                                   947.09     1_239.63     2_186.72       1.0000          1.0000       122.25
IVF-nl353 (self)                                         947.09    20_420.61    21_367.70       0.9999          1.0000       122.25
IVF-nl500-np22 (query)                                 1_822.64       789.27     2_611.91       0.9999          1.0000       122.32
IVF-nl500-np25 (query)                                 1_822.64       887.86     2_710.50       0.9999          1.0000       122.32
IVF-nl500-np31 (query)                                 1_822.64     1_077.25     2_899.89       0.9999          1.0000       122.32
IVF-nl500 (self)                                       1_822.64    17_481.99    19_304.63       0.9999          1.0000       122.32
IVF-nl707-np26 (query)                                 3_831.83       644.49     4_476.32       0.9999          1.0000       122.42
IVF-nl707-np35 (query)                                 3_831.83       859.93     4_691.76       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 3_831.83       919.83     4_751.66       1.0000          1.0000       122.42
IVF-nl707 (self)                                       3_831.83    15_047.91    18_879.74       0.9999          1.0000       122.42
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        98.14     2_019.89     2_118.03       1.0000          1.0000       122.07
Exhaustive (self)                                         98.14    33_942.23    34_040.37       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   109.18     1_376.42     1_485.60       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    109.18    20_922.99    21_032.18       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                               577.29       347.17       924.46       0.9999          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               577.29       345.35       922.64       0.9999          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               577.29       420.72       998.01       0.9999          1.0000         1.91
IVF-GPU-nl353 (self)                                     577.29     4_001.36     4_578.65       0.9999          1.0000         1.91
IVF-GPU-nl500-np22 (query)                               811.04       331.58     1_142.61       0.9999          1.0000         1.91
IVF-GPU-nl500-np25 (query)                               811.04       368.16     1_179.20       0.9999          1.0000         1.91
IVF-GPU-nl500-np31 (query)                               811.04       393.46     1_204.50       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                     811.04     3_632.54     4_443.58       0.9999          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             1_280.88       324.90     1_605.78       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             1_280.88       366.60     1_647.48       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             1_280.88       375.22     1_656.10       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   1_280.88     3_153.67     4_434.55       0.9999          1.0000         1.91
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Increasing the number of samples

<details>
<summary><b>CPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       131.52     2_870.02     3_001.54       1.0000          1.0000       122.07
Exhaustive (self)                                        131.52    90_587.87    90_719.38       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 1_573.19       799.36     2_372.55       0.9998          1.0000       122.20
IVF-nl500-np25 (query)                                 1_573.19       896.30     2_469.49       0.9999          1.0000       122.20
IVF-nl500-np31 (query)                                 1_573.19     1_106.63     2_679.82       0.9999          1.0000       122.20
IVF-nl500 (self)                                       1_573.19    36_090.04    37_663.23       0.9999          1.0000       122.20
IVF-nl707-np26 (query)                                 2_973.61       683.92     3_657.53       0.9999          1.0000       122.25
IVF-nl707-np35 (query)                                 2_973.61       906.94     3_880.54       0.9999          1.0000       122.25
IVF-nl707-np37 (query)                                 2_973.61       953.97     3_927.58       0.9999          1.0000       122.25
IVF-nl707 (self)                                       2_973.61    31_044.11    34_017.72       0.9999          1.0000       122.25
IVF-nl1000-np31 (query)                                5_857.83       586.67     6_444.50       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                5_857.83       827.95     6_685.78       0.9999          1.0000       122.32
IVF-nl1000-np50 (query)                                5_857.83       937.67     6_795.50       0.9999          1.0000       122.32
IVF-nl1000 (self)                                      5_857.83    27_019.46    32_877.29       0.9999          1.0000       122.32
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       130.57     2_740.77     2_871.34       1.0000          1.0000       122.07
Exhaustive (self)                                        130.57    90_220.47    90_351.04       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   145.50     1_646.68     1_792.18       0.9999          1.0000       122.07
GPU-Exhaustive (self)                                    145.50    50_533.92    50_679.43       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                               597.37       357.31       954.68       0.9999          1.0000         3.82
IVF-GPU-nl500-np25 (query)                               597.37       370.73       968.11       0.9999          1.0000         3.82
IVF-GPU-nl500-np31 (query)                               597.37       409.00     1_006.37       0.9999          1.0000         3.82
IVF-GPU-nl500 (self)                                     597.37     7_999.00     8_596.37       0.9999          1.0000         3.82
IVF-GPU-nl707-np26 (query)                               824.38       327.85     1_152.23       0.9998          1.0000         3.82
IVF-GPU-nl707-np35 (query)                               824.38       386.78     1_211.16       0.9999          1.0000         3.82
IVF-GPU-nl707-np37 (query)                               824.38       380.50     1_204.89       0.9999          1.0000         3.82
IVF-GPU-nl707 (self)                                     824.38     7_126.83     7_951.21       0.9999          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            1_242.81       334.14     1_576.94       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            1_242.81       364.45     1_607.26       0.9999          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            1_242.81       388.55     1_631.36       0.9999          1.0000         3.82
IVF-GPU-nl1000 (self)                                  1_242.81     6_305.91     7_548.71       0.9999          1.0000         3.82
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       254.51     4_113.02     4_367.52       1.0000          1.0000       244.14
Exhaustive (self)                                        254.51   130_569.69   130_824.19       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_117.53     1_583.85     3_701.39       0.9999          1.0000       244.39
IVF-nl500-np25 (query)                                 2_117.53     1_796.86     3_914.39       0.9999          1.0000       244.39
IVF-nl500-np31 (query)                                 2_117.53     2_214.49     4_332.02       0.9999          1.0000       244.39
IVF-nl500 (self)                                       2_117.53    73_623.40    75_740.93       0.9999          1.0000       244.39
IVF-nl707-np26 (query)                                 4_387.00     1_325.66     5_712.66       0.9999          1.0000       244.49
IVF-nl707-np35 (query)                                 4_387.00     1_771.99     6_158.99       0.9999          1.0000       244.49
IVF-nl707-np37 (query)                                 4_387.00     1_878.91     6_265.91       0.9999          1.0000       244.49
IVF-nl707 (self)                                       4_387.00    61_963.59    66_350.59       0.9999          1.0000       244.49
IVF-nl1000-np31 (query)                                9_462.18     1_122.19    10_584.37       0.9999          1.0000       244.64
IVF-nl1000-np44 (query)                                9_462.18     1_576.58    11_038.76       0.9999          1.0000       244.64
IVF-nl1000-np50 (query)                                9_462.18     1_795.52    11_257.70       0.9999          1.0000       244.64
IVF-nl1000 (self)                                      9_462.18    52_010.30    61_472.48       0.9999          1.0000       244.64
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       252.84     4_131.71     4_384.55       1.0000          1.0000       244.14
Exhaustive (self)                                        252.84   131_592.71   131_845.55       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                   283.46     2_697.81     2_981.27       0.9999          1.0000       244.14
GPU-Exhaustive (self)                                    283.46    83_154.55    83_438.02       0.9999          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             1_082.42       471.52     1_553.94       0.9999          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             1_082.42       477.57     1_560.00       0.9999          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             1_082.42       537.14     1_619.57       0.9999          1.0000         3.82
IVF-GPU-nl500 (self)                                   1_082.42    12_352.13    13_434.55       0.9999          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             1_603.32       449.46     2_052.79       0.9999          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             1_603.32       487.56     2_090.89       0.9999          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             1_603.32       485.77     2_089.10       0.9999          1.0000         3.82
IVF-GPU-nl707 (self)                                   1_603.32    10_695.83    12_299.16       0.9999          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            2_489.41       421.37     2_910.78       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            2_489.41       464.60     2_954.00       0.9999          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            2_489.41       496.53     2_985.94       0.9999          1.0000         3.82
IVF-GPU-nl1000 (self)                                  2_489.41     9_383.60    11_873.01       0.9999          1.0000         3.82
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### CAGRA-type querying

A [CAGRA-style index](https://arxiv.org/abs/2308.15136): NN-Descent runs
entirely on the GPU (random init, then a random-partition forest, then local
joins until convergence), and the resulting graph is pruned to degree `k` by
rank-based detour counting into a directed navigational graph. Queries are a GPU
beam search, one workgroup per query, with the query vector in shared memory and
a linear-probing hash table for visited-node deduplication.

**Tunable parameters:**

* `build_k`: Internal NNDescent degree before CAGRA pruning. Defaults to
  `1.5 * k`. Higher values give CAGRA more edges to choose from when building
  the navigational graph, at the cost of build time.
* `refine_knn`: Number of 2-hop refinement sweeps after NNDescent convergence.
  Each sweep evaluates all neighbours-of-neighbours and merges improvements.
  Defaults to 0. Mostly a lever on the extracted graph rather than on beam
  search recall.
* `n_trees`: Number of random partition trees for forest initialisation.
  Defaults to 5 + n^0.25, capped at 20. More trees raise the raw graph quality
  ceiling but increase build time linearly.
* `beam_width`: Number of active candidates maintained during beam search.
  Defaults to 2 * max(k_out, k_graph). Wider beams improve recall at the cost
  of query latency. Auto-scaled when using CagraGpuSearchParams::from_graph().
* `max_beam_iters`: Safety cap on beam search iterations. Defaults to
  3 * beam_width. Most queries terminate naturally well before this limit; it
  only fires for pathological cases where the search keeps discovering better
  candidates.
* `n_entry_points`: Number of seed nodes per query for beam search. Defaults
  to 8. Entry points are sourced from a small Annoy forest (external queries)
  or from the kNN graph's closest neighbours (self-query).

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.30       653.72       665.01       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.30     6_318.69     6_329.99       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    15.24       417.87       433.11       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     15.24     3_247.82     3_263.06       1.0000          1.0000        18.31
CAGRA-auto (query)                                       538.09       108.14       646.23       0.9594          1.0023        61.23
CAGRA-auto (self)                                        538.09       638.24     1_176.32       0.9615          1.0023        61.23
CAGRA-bw16 (query)                                       538.09        80.59       618.68       0.9492          1.0027        61.23
CAGRA-bw16 (self)                                        538.09       302.56       840.65       0.9545          1.0026        61.23
CAGRA-bw30 (query)                                       538.09       107.18       645.26       0.9585          1.0023        61.23
CAGRA-bw30 (self)                                        538.09       598.56     1_136.65       0.9609          1.0024        61.23
CAGRA-bw48 (query)                                       538.09       158.20       696.29       0.9647          1.0020        61.23
CAGRA-bw48 (self)                                        538.09     1_085.32     1_623.41       0.9657          1.0021        61.23
CAGRA-bw64 (query)                                       538.09       235.09       773.18       0.9683          1.0018        61.23
CAGRA-bw64 (self)                                        538.09     1_640.85     2_178.94       0.9687          1.0019        61.23
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Cosine (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.77       687.33       699.10       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                     11.77     6_779.79     6_791.57       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                    16.06       401.87       417.94       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                     16.06     3_119.89     3_135.96       1.0000          1.0000        18.88
CAGRA-auto (query)                                       625.48       174.68       800.17       0.9607          1.0022        61.80
CAGRA-auto (self)                                        625.48       654.00     1_279.48       0.9639          1.0021        61.80
CAGRA-bw16 (query)                                       625.48       138.26       763.74       0.9491          1.0026        61.80
CAGRA-bw16 (self)                                        625.48       316.88       942.36       0.9562          1.0025        61.80
CAGRA-bw30 (query)                                       625.48       168.32       793.81       0.9598          1.0022        61.80
CAGRA-bw30 (self)                                        625.48       599.73     1_225.21       0.9633          1.0022        61.80
CAGRA-bw48 (query)                                       625.48       216.98       842.47       0.9665          1.0019        61.80
CAGRA-bw48 (self)                                        625.48     1_103.98     1_729.47       0.9684          1.0019        61.80
CAGRA-bw64 (query)                                       625.48       275.22       900.70       0.9704          1.0017        61.80
CAGRA-bw64 (self)                                        625.48     1_680.28     2_305.76       0.9716          1.0017        61.80
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Correlated)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.11       663.95       675.07       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.11     6_500.10     6_511.21       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    15.50       415.69       431.19       0.9989          1.0000        18.31
GPU-Exhaustive (self)                                     15.50     3_235.82     3_251.32       0.9989          1.0000        18.31
CAGRA-auto (query)                                       505.60       109.14       614.73       0.9819          1.0009        61.23
CAGRA-auto (self)                                        505.60       629.03     1_134.62       0.9944          1.0002        61.23
CAGRA-bw16 (query)                                       505.60       109.89       615.49       0.9588          1.0021        61.23
CAGRA-bw16 (self)                                        505.60       298.76       804.36       0.9902          1.0003        61.23
CAGRA-bw30 (query)                                       505.60       130.31       635.91       0.9805          1.0009        61.23
CAGRA-bw30 (self)                                        505.60       580.33     1_085.93       0.9941          1.0002        61.23
CAGRA-bw48 (query)                                       505.60       177.49       683.09       0.9896          1.0005        61.23
CAGRA-bw48 (self)                                        505.60     1_063.33     1_568.93       0.9963          1.0001        61.23
CAGRA-bw64 (query)                                       505.60       208.20       713.80       0.9933          1.0003        61.23
CAGRA-bw64 (self)                                        505.60     1_593.68     2_099.28       0.9972          1.0001        61.23
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.13       662.52       673.65       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.13     6_562.37     6_573.50       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    14.86       413.93       428.79       0.9995          1.0000        18.31
GPU-Exhaustive (self)                                     14.86     3_248.22     3_263.08       0.9995          1.0000        18.31
CAGRA-auto (query)                                       462.77       104.93       567.70       0.9885          1.0006        61.23
CAGRA-auto (self)                                        462.77       631.65     1_094.43       0.9959          1.0002        61.23
CAGRA-bw16 (query)                                       462.77        75.71       538.48       0.9725          1.0015        61.23
CAGRA-bw16 (self)                                        462.77       311.59       774.37       0.9918          1.0003        61.23
CAGRA-bw30 (query)                                       462.77       125.60       588.37       0.9876          1.0006        61.23
CAGRA-bw30 (self)                                        462.77       581.14     1_043.92       0.9956          1.0002        61.23
CAGRA-bw48 (query)                                       462.77       174.92       637.69       0.9936          1.0003        61.23
CAGRA-bw48 (self)                                        462.77     1_070.54     1_533.32       0.9976          1.0001        61.23
CAGRA-bw64 (query)                                       462.77       208.47       671.25       0.9959          1.0002        61.23
CAGRA-bw64 (self)                                        462.77     1_609.84     2_072.62       0.9983          1.0001        61.23
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 128D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    48.66     1_227.86     1_276.51       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     48.66    12_482.77    12_531.42       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    54.24       856.98       911.22       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     54.24     7_498.33     7_552.58       1.0000          1.0000        73.24
CAGRA-auto (query)                                     1_526.14       206.97     1_733.10       0.9999          1.0000       116.16
CAGRA-auto (self)                                      1_526.14       687.70     2_213.83       0.9999          1.0000       116.16
CAGRA-bw16 (query)                                     1_526.14       175.32     1_701.46       0.9998          1.0000       116.16
CAGRA-bw16 (self)                                      1_526.14       382.51     1_908.65       0.9998          1.0000       116.16
CAGRA-bw30 (query)                                     1_526.14       209.57     1_735.71       0.9999          1.0000       116.16
CAGRA-bw30 (self)                                      1_526.14       659.76     2_185.90       0.9999          1.0000       116.16
CAGRA-bw48 (query)                                     1_526.14       246.32     1_772.46       0.9999          1.0000       116.16
CAGRA-bw48 (self)                                      1_526.14     1_045.43     2_571.57       0.9999          1.0000       116.16
CAGRA-bw64 (query)                                     1_526.14       291.06     1_817.20       0.9999          1.0000       116.16
CAGRA-bw64 (self)                                      1_526.14     1_487.11     3_013.25       0.9999          1.0000       116.16
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Larger data sets

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    45.00     2_359.80     2_404.81       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     45.00    24_069.53    24_114.54       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    49.57     1_420.22     1_469.80       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     49.57    12_655.62    12_705.19       1.0000          1.0000        61.04
CAGRA-auto (query)                                     1_630.73       310.44     1_941.17       0.9999          1.0000       132.56
CAGRA-auto (self)                                      1_630.73     1_021.93     2_652.66       0.9999          1.0000       132.56
CAGRA-bw16 (query)                                     1_630.73       258.66     1_889.40       0.9998          1.0000       132.56
CAGRA-bw16 (self)                                      1_630.73       540.74     2_171.47       0.9998          1.0000       132.56
CAGRA-bw30 (query)                                     1_630.73       303.71     1_934.44       0.9999          1.0000       132.56
CAGRA-bw30 (self)                                      1_630.73       959.25     2_589.99       0.9999          1.0000       132.56
CAGRA-bw48 (query)                                     1_630.73       367.03     1_997.76       1.0000          1.0000       132.56
CAGRA-bw48 (self)                                      1_630.73     1_615.49     3_246.23       1.0000          1.0000       132.56
CAGRA-bw64 (query)                                     1_630.73       440.58     2_071.31       1.0000          1.0000       132.56
CAGRA-bw64 (self)                                      1_630.73     2_338.80     3_969.53       1.0000          1.0000       132.56
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    87.16     3_431.07     3_518.22       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     87.16    34_554.23    34_641.38       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    99.72     2_244.02     2_343.74       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     99.72    20_655.37    20_755.09       1.0000          1.0000       122.07
CAGRA-auto (query)                                     2_755.75       419.62     3_175.37       0.9999          1.0000       193.60
CAGRA-auto (self)                                      2_755.75     1_167.43     3_923.19       0.9999          1.0000       193.60
CAGRA-bw16 (query)                                     2_755.75       361.76     3_117.51       0.9998          1.0000       193.60
CAGRA-bw16 (self)                                      2_755.75       640.29     3_396.04       0.9998          1.0000       193.60
CAGRA-bw30 (query)                                     2_755.75       406.25     3_162.00       0.9999          1.0000       193.60
CAGRA-bw30 (self)                                      2_755.75     1_099.54     3_855.29       0.9999          1.0000       193.60
CAGRA-bw48 (query)                                     2_755.75       471.99     3_227.74       0.9999          1.0000       193.60
CAGRA-bw48 (self)                                      2_755.75     1_747.88     4_503.63       0.9999          1.0000       193.60
CAGRA-bw64 (query)                                     2_755.75       579.27     3_335.02       0.9999          1.0000       193.60
CAGRA-bw64 (self)                                      2_755.75     2_488.64     5_244.39       0.9999          1.0000       193.60
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                   124.36     9_355.54     9_479.91       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                    124.36    92_175.96    92_300.33       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   135.81     5_237.23     5_373.04       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    135.81    50_592.75    50_728.55       1.0000          1.0000       122.07
CAGRA-auto (query)                                     3_247.86       590.46     3_838.32       0.9999          1.0000       265.12
CAGRA-auto (self)                                      3_247.86     2_052.85     5_300.71       0.9999          1.0000       265.12
CAGRA-bw16 (query)                                     3_247.86       470.94     3_718.80       0.9998          1.0000       265.12
CAGRA-bw16 (self)                                      3_247.86     1_085.98     4_333.85       0.9998          1.0000       265.12
CAGRA-bw30 (query)                                     3_247.86       552.00     3_799.87       0.9999          1.0000       265.12
CAGRA-bw30 (self)                                      3_247.86     1_923.60     5_171.46       0.9999          1.0000       265.12
CAGRA-bw48 (query)                                     3_247.86       660.55     3_908.41       0.9999          1.0000       265.12
CAGRA-bw48 (self)                                      3_247.86     3_198.84     6_446.70       0.9999          1.0000       265.12
CAGRA-bw64 (query)                                     3_247.86       802.53     4_050.39       0.9999          1.0000       265.12
CAGRA-bw64 (self)                                      3_247.86     4_633.91     7_881.77       0.9999          1.0000       265.12
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 128D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                   253.25    13_646.50    13_899.75       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                    253.25   133_062.55   133_315.80       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                   289.00     8_569.26     8_858.26       0.9999          1.0000       244.14
GPU-Exhaustive (self)                                    289.00    83_153.69    83_442.69       0.9999          1.0000       244.14
CAGRA-auto (query)                                     6_023.07       846.16     6_869.23       0.9999          1.0000       387.19
CAGRA-auto (self)                                      6_023.07     2_342.30     8_365.37       0.9998          1.0000       387.19
CAGRA-bw16 (query)                                     6_023.07       729.76     6_752.83       0.9998          1.0000       387.19
CAGRA-bw16 (self)                                      6_023.07     1_269.29     7_292.36       0.9998          1.0000       387.19
CAGRA-bw30 (query)                                     6_023.07       836.71     6_859.78       0.9999          1.0000       387.19
CAGRA-bw30 (self)                                      6_023.07     2_203.45     8_226.52       0.9998          1.0000       387.19
CAGRA-bw48 (query)                                     6_023.07       960.52     6_983.59       0.9999          1.0000       387.19
CAGRA-bw48 (self)                                      6_023.07     3_509.00     9_532.07       0.9999          1.0000       387.19
CAGRA-bw64 (query)                                     6_023.07     1_088.82     7_111.89       0.9999          1.0000       387.19
CAGRA-bw64 (self)                                      6_023.07     4_952.11    10_975.18       0.9999          1.0000       387.19
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Navigating Spread-out Graph (NSG) with GPU-accelerated kNN generation

NSG builds on top of an existing kNN graph. The CPU path uses NN-Descent; the
GPU path swaps in the same NN-Descent that feeds CAGRA. The two columns below
differ only in that initialisation step.

<details>
<summary><b>NSG with CPU initialisation</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        91.59     1_985.68     2_077.27       1.0000          1.0000       122.07
Exhaustive (self)                                         91.59    33_852.29    33_943.88       1.0000          1.0000       122.07
NSG-R24-L50-ef50 (query)                               8_015.74       162.38     8_178.12       0.9983          1.0062       144.96
NSG-R24-L50-efauto (query)                             8_015.74       255.81     8_271.55       0.9994          1.0025       144.96
NSG-R24-L50-ef150 (query)                              8_015.74       345.00     8_360.73       0.9998          1.0008       144.96
NSG-R24-L50 (self)                                     8_015.74     2_582.75    10_598.49       1.0000          1.0000       144.96
NSG-R24-L100-ef50 (query)                              9_694.08       162.12     9_856.20       0.9991          1.0030       145.91
NSG-R24-L100-efauto (query)                            9_694.08       250.71     9_944.79       0.9999          1.0003       145.91
NSG-R24-L100-ef150 (query)                             9_694.08       337.90    10_031.98       1.0000          1.0000       145.91
NSG-R24-L100 (self)                                    9_694.08     2_648.86    12_342.94       1.0000          1.0000       145.91
NSG-R24-L150-ef50 (query)                             11_005.91       163.92    11_169.82       0.9991          1.0030       145.91
NSG-R24-L150-efauto (query)                           11_005.91       250.32    11_256.23       0.9999          1.0003       145.91
NSG-R24-L150-ef150 (query)                            11_005.91       338.36    11_344.27       1.0000          1.0000       145.91
NSG-R24-L150 (self)                                   11_005.91     2_639.18    13_645.09       1.0000          1.0000       145.91
NSG-R32-L50-ef50 (query)                               7_926.94       174.59     8_101.54       0.9984          1.0059       152.59
NSG-R32-L50-efauto (query)                             7_926.94       273.43     8_200.37       0.9994          1.0021       152.59
NSG-R32-L50-ef150 (query)                              7_926.94       371.57     8_298.52       0.9998          1.0006       152.59
NSG-R32-L50 (self)                                     7_926.94     2_846.63    10_773.58       1.0000          1.0000       152.59
NSG-R32-L100-ef50 (query)                             10_007.99       174.29    10_182.28       0.9990          1.0033       152.59
NSG-R32-L100-efauto (query)                           10_007.99       266.41    10_274.40       0.9999          1.0003       152.59
NSG-R32-L100-ef150 (query)                            10_007.99       364.35    10_372.34       1.0000          1.0000       152.59
NSG-R32-L100 (self)                                   10_007.99     3_044.88    13_052.87       1.0000          1.0000       152.59
NSG-R32-L150-ef50 (query)                             10_992.09       175.12    11_167.21       0.9990          1.0033       152.59
NSG-R32-L150-efauto (query)                           10_992.09       268.90    11_260.99       0.9999          1.0003       152.59
NSG-R32-L150-ef150 (query)                            10_992.09       361.12    11_353.20       1.0000          1.0000       152.59
NSG-R32-L150 (self)                                   10_992.09     2_880.35    13_872.44       1.0000          1.0000       152.59
NSG-R48-L50-ef50 (query)                               8_038.94       183.29     8_222.23       0.9985          1.0056       167.85
NSG-R48-L50-efauto (query)                             8_038.94       278.27     8_317.21       0.9994          1.0021       167.85
NSG-R48-L50-ef150 (query)                              8_038.94       369.25     8_408.19       0.9998          1.0006       167.85
NSG-R48-L50 (self)                                     8_038.94     2_938.84    10_977.79       1.0000          1.0000       167.85
NSG-R48-L100-ef50 (query)                              9_707.00       179.39     9_886.39       0.9990          1.0033       167.85
NSG-R48-L100-efauto (query)                            9_707.00       277.59     9_984.58       0.9999          1.0003       167.85
NSG-R48-L100-ef150 (query)                             9_707.00       374.76    10_081.76       1.0000          1.0000       167.85
NSG-R48-L100 (self)                                    9_707.00     2_984.92    12_691.91       1.0000          1.0000       167.85
NSG-R48-L150-ef50 (query)                             11_100.29       181.01    11_281.30       0.9990          1.0033       167.85
NSG-R48-L150-efauto (query)                           11_100.29       281.07    11_381.36       0.9999          1.0003       167.85
NSG-R48-L150-ef150 (query)                            11_100.29       382.89    11_483.18       1.0000          1.0000       167.85
NSG-R48-L150 (self)                                   11_100.29     3_008.89    14_109.18       1.0000          1.0000       167.85
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>NSG with GPU initialisation</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        88.40     2_012.73     2_101.13       1.0000          1.0000       122.07
Exhaustive (self)                                         88.40    33_771.20    33_859.61       1.0000          1.0000       122.07
NSG-GPU-R24-L50-ef50 (query)                           5_469.76       162.46     5_632.22       0.9980          1.0072       144.96
NSG-GPU-R24-L50-efauto (query)                         5_469.76       251.56     5_721.32       0.9994          1.0025       144.96
NSG-GPU-R24-L50-ef150 (query)                          5_469.76       341.08     5_810.84       0.9998          1.0008       144.96
NSG-GPU-R24-L50 (self)                                 5_469.76     2_605.81     8_075.57       1.0000          1.0000       144.96
NSG-GPU-R24-L100-ef50 (query)                          7_208.07       162.19     7_370.26       0.9987          1.0039       144.96
NSG-GPU-R24-L100-efauto (query)                        7_208.07       253.23     7_461.30       0.9998          1.0004       144.96
NSG-GPU-R24-L100-ef150 (query)                         7_208.07       337.59     7_545.67       0.9999          1.0001       144.96
NSG-GPU-R24-L100 (self)                                7_208.07     2_638.43     9_846.51       1.0000          1.0000       144.96
NSG-GPU-R24-L150-ef50 (query)                          8_658.46       162.93     8_821.39       0.9987          1.0039       144.96
NSG-GPU-R24-L150-efauto (query)                        8_658.46       251.81     8_910.27       0.9998          1.0004       144.96
NSG-GPU-R24-L150-ef150 (query)                         8_658.46       340.49     8_998.95       0.9999          1.0001       144.96
NSG-GPU-R24-L150 (self)                                8_658.46     2_648.14    11_306.60       1.0000          1.0000       144.96
NSG-GPU-R32-L50-ef50 (query)                           5_508.32       177.57     5_685.89       0.9983          1.0063       152.59
NSG-GPU-R32-L50-efauto (query)                         5_508.32       270.17     5_778.50       0.9994          1.0021       152.59
NSG-GPU-R32-L50-ef150 (query)                          5_508.32       361.79     5_870.11       0.9998          1.0006       152.59
NSG-GPU-R32-L50 (self)                                 5_508.32     2_865.29     8_373.62       1.0000          1.0000       152.59
NSG-GPU-R32-L100-ef50 (query)                          7_224.50       175.83     7_400.32       0.9989          1.0035       152.59
NSG-GPU-R32-L100-efauto (query)                        7_224.50       269.93     7_494.43       0.9998          1.0004       152.59
NSG-GPU-R32-L100-ef150 (query)                         7_224.50       362.38     7_586.88       0.9999          1.0001       152.59
NSG-GPU-R32-L100 (self)                                7_224.50     2_889.84    10_114.34       1.0000          1.0000       152.59
NSG-GPU-R32-L150-ef50 (query)                          8_554.00       176.85     8_730.85       0.9989          1.0035       152.59
NSG-GPU-R32-L150-efauto (query)                        8_554.00       278.79     8_832.79       0.9998          1.0004       152.59
NSG-GPU-R32-L150-ef150 (query)                         8_554.00       376.71     8_930.71       0.9999          1.0001       152.59
NSG-GPU-R32-L150 (self)                                8_554.00     2_933.01    11_487.01       1.0000          1.0000       152.59
NSG-GPU-R48-L50-ef50 (query)                           5_539.26       181.23     5_720.50       0.9984          1.0060       167.85
NSG-GPU-R48-L50-efauto (query)                         5_539.26       279.66     5_818.92       0.9994          1.0021       167.85
NSG-GPU-R48-L50-ef150 (query)                          5_539.26       377.29     5_916.55       0.9998          1.0006       167.85
NSG-GPU-R48-L50 (self)                                 5_539.26     2_991.05     8_530.31       1.0000          1.0000       167.85
NSG-GPU-R48-L100-ef50 (query)                          7_275.23       182.39     7_457.62       0.9989          1.0035       167.85
NSG-GPU-R48-L100-efauto (query)                        7_275.23       279.65     7_554.88       0.9998          1.0004       167.85
NSG-GPU-R48-L100-ef150 (query)                         7_275.23       375.69     7_650.92       0.9999          1.0001       167.85
NSG-GPU-R48-L100 (self)                                7_275.23     3_067.17    10_342.40       1.0000          1.0000       167.85
NSG-GPU-R48-L150-ef50 (query)                          8_602.07       183.63     8_785.70       0.9989          1.0035       167.85
NSG-GPU-R48-L150-efauto (query)                        8_602.07       279.93     8_882.00       0.9998          1.0004       167.85
NSG-GPU-R48-L150-ef150 (query)                         8_602.07       380.10     8_982.17       0.9999          1.0001       167.85
NSG-GPU-R48-L150 (self)                                8_602.07     3_039.40    11_641.47       1.0000          1.0000       167.85
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>NSG with CPU initialisation (more samples)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       254.12     4_069.13     4_323.26       1.0000          1.0000       244.14
Exhaustive (self)                                        254.12   132_924.77   133_178.89       1.0000          1.0000       244.14
NSG-R24-L50-ef50 (query)                              18_114.14       180.46    18_294.60       0.9958          1.0208       289.92
NSG-R24-L50-efauto (query)                            18_114.14       277.62    18_391.75       0.9986          1.0066       289.92
NSG-R24-L50-ef150 (query)                             18_114.14       369.34    18_483.48       0.9992          1.0036       289.92
NSG-R24-L50 (self)                                    18_114.14     5_712.43    23_826.57       0.9999          1.0000       289.92
NSG-R24-L100-ef50 (query)                             21_994.79       187.48    22_182.27       0.9960          1.0195       289.92
NSG-R24-L100-efauto (query)                           21_994.79       285.87    22_280.66       0.9988          1.0056       289.92
NSG-R24-L100-ef150 (query)                            21_994.79       369.45    22_364.24       0.9995          1.0024       289.92
NSG-R24-L100 (self)                                   21_994.79     5_785.22    27_780.01       0.9999          1.0000       289.92
NSG-R24-L150-ef50 (query)                             25_401.88       182.82    25_584.70       0.9981          1.0085       289.92
NSG-R24-L150-efauto (query)                           25_401.88       278.68    25_680.57       0.9992          1.0035       289.92
NSG-R24-L150-ef150 (query)                            25_401.88       372.85    25_774.74       0.9994          1.0029       289.92
NSG-R24-L150 (self)                                   25_401.88     5_817.97    31_219.86       0.9999          1.0000       289.92
NSG-R32-L50-ef50 (query)                              17_979.32       195.85    18_175.16       0.9964          1.0177       305.18
NSG-R32-L50-efauto (query)                            17_979.32       297.97    18_277.28       0.9986          1.0066       305.18
NSG-R32-L50-ef150 (query)                             17_979.32       396.38    18_375.70       0.9992          1.0036       305.18
NSG-R32-L50 (self)                                    17_979.32     6_389.30    24_368.62       0.9999          1.0000       305.18
NSG-R32-L100-ef50 (query)                             21_934.69       202.06    22_136.75       0.9963          1.0175       305.18
NSG-R32-L100-efauto (query)                           21_934.69       308.62    22_243.31       0.9988          1.0056       305.18
NSG-R32-L100-ef150 (query)                            21_934.69       406.94    22_341.64       0.9995          1.0024       305.18
NSG-R32-L100 (self)                                   21_934.69     6_330.26    28_264.95       0.9999          1.0000       305.18
NSG-R32-L150-ef50 (query)                             25_270.08       196.15    25_466.23       0.9984          1.0068       305.18
NSG-R32-L150-efauto (query)                           25_270.08       296.13    25_566.20       0.9992          1.0035       305.18
NSG-R32-L150-ef150 (query)                            25_270.08       395.27    25_665.35       0.9994          1.0029       305.18
NSG-R32-L150 (self)                                   25_270.08     6_313.45    31_583.53       0.9999          1.0000       305.18
NSG-R48-L50-ef50 (query)                              18_144.93       200.44    18_345.37       0.9964          1.0177       335.69
NSG-R48-L50-efauto (query)                            18_144.93       306.74    18_451.67       0.9986          1.0066       335.69
NSG-R48-L50-ef150 (query)                             18_144.93       407.15    18_552.08       0.9992          1.0036       335.69
NSG-R48-L50 (self)                                    18_144.93     6_547.46    24_692.39       0.9999          1.0000       335.69
NSG-R48-L100-ef50 (query)                             21_987.51       202.02    22_189.53       0.9963          1.0175       335.69
NSG-R48-L100-efauto (query)                           21_987.51       308.02    22_295.52       0.9988          1.0056       335.69
NSG-R48-L100-ef150 (query)                            21_987.51       408.52    22_396.03       0.9995          1.0024       335.69
NSG-R48-L100 (self)                                   21_987.51     6_626.64    28_614.15       0.9999          1.0000       335.69
NSG-R48-L150-ef50 (query)                             25_520.71       206.53    25_727.24       0.9985          1.0065       335.69
NSG-R48-L150-efauto (query)                           25_520.71       314.52    25_835.23       0.9992          1.0035       335.69
NSG-R48-L150-ef150 (query)                            25_520.71       415.93    25_936.64       0.9994          1.0029       335.69
NSG-R48-L150 (self)                                   25_520.71     6_642.63    32_163.34       0.9999          1.0000       335.69
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>NSG with GPU initialisation (more samples)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       254.83     4_234.78     4_489.61       1.0000          1.0000       244.14
Exhaustive (self)                                        254.83   135_670.14   135_924.97       1.0000          1.0000       244.14
NSG-GPU-R24-L50-ef50 (query)                          11_784.19       180.67    11_964.86       0.9953          1.0234       289.92
NSG-GPU-R24-L50-efauto (query)                        11_784.19       278.02    12_062.20       0.9986          1.0069       289.92
NSG-GPU-R24-L50-ef150 (query)                         11_784.19       374.90    12_159.09       0.9994          1.0028       289.92
NSG-GPU-R24-L50 (self)                                11_784.19     5_744.16    17_528.35       0.9999          1.0000       289.92
NSG-GPU-R24-L100-ef50 (query)                         15_927.97       186.30    16_114.27       0.9956          1.0216       289.92
NSG-GPU-R24-L100-efauto (query)                       15_927.97       281.51    16_209.49       0.9988          1.0059       289.92
NSG-GPU-R24-L100-ef150 (query)                        15_927.97       372.15    16_300.12       0.9996          1.0016       289.92
NSG-GPU-R24-L100 (self)                               15_927.97     5_805.34    21_733.32       0.9999          1.0000       289.92
NSG-GPU-R24-L150-ef50 (query)                         19_048.32       184.13    19_232.45       0.9981          1.0088       289.92
NSG-GPU-R24-L150-efauto (query)                       19_048.32       279.30    19_327.62       0.9992          1.0035       289.92
NSG-GPU-R24-L150-ef150 (query)                        19_048.32       371.66    19_419.98       0.9994          1.0029       289.92
NSG-GPU-R24-L150 (self)                               19_048.32     5_783.66    24_831.98       0.9999          1.0000       289.92
NSG-GPU-R32-L50-ef50 (query)                          11_849.57       200.80    12_050.37       0.9959          1.0204       305.18
NSG-GPU-R32-L50-efauto (query)                        11_849.57       300.62    12_150.19       0.9986          1.0066       305.18
NSG-GPU-R32-L50-ef150 (query)                         11_849.57       402.09    12_251.66       0.9994          1.0028       305.18
NSG-GPU-R32-L50 (self)                                11_849.57     6_322.79    18_172.36       0.9999          1.0000       305.18
NSG-GPU-R32-L100-ef50 (query)                         15_735.18       199.05    15_934.23       0.9958          1.0203       305.18
NSG-GPU-R32-L100-efauto (query)                       15_735.18       302.04    16_037.22       0.9988          1.0056       305.18
NSG-GPU-R32-L100-ef150 (query)                        15_735.18       403.25    16_138.43       0.9996          1.0016       305.18
NSG-GPU-R32-L100 (self)                               15_735.18     6_525.44    22_260.61       0.9999          1.0000       305.18
NSG-GPU-R32-L150-ef50 (query)                         20_321.51       202.11    20_523.61       0.9984          1.0075       305.18
NSG-GPU-R32-L150-efauto (query)                       20_321.51       300.59    20_622.10       0.9992          1.0035       305.18
NSG-GPU-R32-L150-ef150 (query)                        20_321.51       399.83    20_721.33       0.9994          1.0029       305.18
NSG-GPU-R32-L150 (self)                               20_321.51     6_348.86    26_670.37       0.9999          1.0000       305.18
NSG-GPU-R48-L50-ef50 (query)                          11_865.00       212.28    12_077.28       0.9959          1.0204       335.69
NSG-GPU-R48-L50-efauto (query)                        11_865.00       310.12    12_175.12       0.9986          1.0066       335.69
NSG-GPU-R48-L50-ef150 (query)                         11_865.00       420.83    12_285.83       0.9994          1.0028       335.69
NSG-GPU-R48-L50 (self)                                11_865.00     6_522.50    18_387.51       0.9999          1.0000       335.69
NSG-GPU-R48-L100-ef50 (query)                         15_751.88       204.48    15_956.36       0.9958          1.0203       335.69
NSG-GPU-R48-L100-efauto (query)                       15_751.88       312.24    16_064.12       0.9988          1.0056       335.69
NSG-GPU-R48-L100-ef150 (query)                        15_751.88       412.43    16_164.31       0.9996          1.0016       335.69
NSG-GPU-R48-L100 (self)                               15_751.88     6_575.05    22_326.93       0.9999          1.0000       335.69
NSG-GPU-R48-L150-ef50 (query)                         19_165.83       204.05    19_369.87       0.9984          1.0075       335.69
NSG-GPU-R48-L150-efauto (query)                       19_165.83       310.55    19_476.37       0.9992          1.0035       335.69
NSG-GPU-R48-L150-ef150 (query)                        19_165.83       411.51    19_577.33       0.9994          1.0029       335.69
NSG-GPU-R48-L150 (self)                               19_165.83     6_566.35    25_732.18       0.9999          1.0000       335.69
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
