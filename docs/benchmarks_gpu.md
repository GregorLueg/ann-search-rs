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

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided (however, GPU memory is not
reported). To note also, every benchmark here is run on the wgpu backend.
Other backends like cuda might provide even more speed benefits.

## Table of Contents

- [GPU exhaustive and IVF](#gpu-accelerated-exhaustive-and-ivf-vs-cpu-exhaustive)
- [Comparison on larger data sets against the CPU](#comparison-against-ivf-cpu)
- [CAGRA style index](#cagra-type-querying)
- [CAGRA index on larger data](#larger-data-sets)
- [CAGRA for kNN generation](#two-tier-knn-generation)

### GPU-accelerated exhaustive and IVF vs CPU exhaustive

The GPU acceleration is particularly notable for the exhaustive index. The
IVF-GPU reaches very fast speeds here, but not much faster actually than the
IVF-CPU version (or exhaustive GPU index). The advantages for the IVF-GPU index
become more apparent in larger data sets (more to that below). Also to note is
that the data is kept on the GPU for easier access and less frequent transfer
between CPU and GPU, hence, the apparent reduced memory footprint. The data
lives on the GPU for this version. (Be aware of your VRAM limits!).

<details>
<summary><b>GPU - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.06     1_550.79     1_553.84       1.0000          1.0000        18.31
Exhaustive (self)                                          3.06    15_466.23    15_469.29       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     4.98       640.70       645.68       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      4.98     5_405.24     5_410.21       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               636.25       286.08       922.32       0.9873          1.0010         1.15
IVF-GPU-nl273-np16 (query)                               636.25       331.66       967.91       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               636.25       410.46     1_046.71       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     636.25     1_588.11     2_224.35       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_229.11       306.40     1_535.50       0.9925          1.0006         1.15
IVF-GPU-nl387-np27 (query)                             1_229.11       388.51     1_617.61       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_229.11     1_380.51     2_609.61       0.9998          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_378.43       290.18     2_668.61       0.9892          1.0008         1.15
IVF-GPU-nl547-np27 (query)                             2_378.43       327.48     2_705.91       0.9971          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             2_378.43       350.05     2_728.48       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_378.43     1_329.38     3_707.80       0.9998          1.0000         1.15
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
Exhaustive (query)                                         3.89     1_463.19     1_467.07       1.0000          1.0000        18.88
Exhaustive (self)                                          3.89    15_454.71    15_458.60       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     5.75       658.12       663.87       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      5.75     5_609.68     5_615.43       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               594.29       292.09       886.38       0.9881          1.0009         1.15
IVF-GPU-nl273-np16 (query)                               594.29       357.69       951.98       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               594.29       411.28     1_005.57       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     594.29     1_571.89     2_166.18       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_158.81       303.13     1_461.94       0.9929          1.0005         1.15
IVF-GPU-nl387-np27 (query)                             1_158.81       404.14     1_562.95       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_158.81     1_412.93     2_571.74       0.9997          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_270.61       288.63     2_559.24       0.9901          1.0007         1.15
IVF-GPU-nl547-np27 (query)                             2_270.61       333.42     2_604.03       0.9974          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             2_270.61       342.32     2_612.92       0.9998          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_270.61     1_331.35     3_601.96       0.9998          1.0000         1.15
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
Exhaustive (query)                                         3.10     1_581.99     1_585.09       1.0000          1.0000        18.31
Exhaustive (self)                                          3.10    16_124.89    16_127.98       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     4.91       639.39       644.30       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      4.91     5_403.91     5_408.82       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               638.22       280.17       918.39       0.9999          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               638.22       231.41       869.63       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               638.22       371.29     1_009.51       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     638.22     1_457.62     2_095.84       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_248.90       300.37     1_549.26       0.9999          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_248.90       382.40     1_631.30       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_248.90     1_292.40     2_541.30       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_436.45       294.76     2_731.21       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_436.45       312.49     2_748.95       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_436.45       340.80     2_777.25       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_436.45     1_248.68     3_685.14       1.0000          1.0000         1.15
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
Exhaustive (query)                                         2.98     1_569.92     1_572.91       1.0000          1.0000        18.31
Exhaustive (self)                                          2.98    16_326.74    16_329.72       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.20       639.10       644.30       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.20     5_405.56     5_410.76       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               637.12       278.38       915.50       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               637.12       328.08       965.20       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               637.12       411.08     1_048.20       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     637.12     1_466.07     2_103.19       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_244.61       296.46     1_541.07       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_244.61       384.96     1_629.57       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_244.61     1_274.22     2_518.82       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_447.01       153.64     2_600.65       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_447.01       284.15     2_731.16       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_447.01       334.01     2_781.03       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_447.01     1_230.69     3_677.70       1.0000          1.0000         1.15
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
Exhaustive (query)                                        14.25     6_141.96     6_156.21       1.0000          1.0000        73.24
Exhaustive (self)                                         14.25    63_856.73    63_870.98       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    23.12     1_345.65     1_368.77       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     23.12    12_413.71    12_436.83       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               555.80       435.71       991.50       0.9998          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               555.80       508.49     1_064.29       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               555.80       633.90     1_189.69       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     555.80     3_916.11     4_471.90       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_009.33       460.50     1_469.83       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_009.33       574.72     1_584.05       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_009.33     3_357.62     4_366.96       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_058.26       324.71     2_382.97       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_058.26       454.75     2_513.02       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_058.26       542.77     2_601.04       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_058.26     3_124.59     5_182.86       1.0000          1.0000         1.15
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### Comparison against IVF CPU

In this case, the IVF CPU implementation is being compared against the GPU
version. GPU acceleration shines with larger data sets and larger dimensions,
hence, the number of samples was increased to 250_000 and dimensions to 64 or
128 for these benchmarks.

#### With 250k samples and 64 dimensions

<details>
<summary><b>CPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.67     5_008.99     5_019.67       1.0000          1.0000        61.04
Exhaustive (self)                                         10.67    84_548.91    84_559.58       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                 1_838.98       336.16     2_175.14       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                 1_838.98       358.90     2_197.88       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                 1_838.98       544.70     2_383.68       1.0000          1.0000        61.12
IVF-nl353 (self)                                       1_838.98     7_641.02     9_479.99       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 3_672.77       317.52     3_990.30       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 3_672.77       358.98     4_031.76       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 3_672.77       446.55     4_119.33       1.0000          1.0000        61.16
IVF-nl500 (self)                                       3_672.77     6_318.50     9_991.28       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 7_043.11       279.04     7_322.14       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 7_043.11       364.69     7_407.80       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 7_043.11       386.59     7_429.70       1.0000          1.0000        61.21
IVF-nl707 (self)                                       7_043.11     5_039.12    12_082.23       1.0000          1.0000        61.21
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
Exhaustive (query)                                        10.82     5_472.88     5_483.70       1.0000          1.0000        61.04
Exhaustive (self)                                         10.82    89_802.75    89_813.57       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.09     1_418.55     1_436.64       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.09    21_507.95    21_526.05       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                             1_810.73       458.35     2_269.08       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_810.73       501.66     2_312.39       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_810.73       609.42     2_420.15       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_810.73     5_205.79     7_016.52       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             3_714.83       489.93     4_204.76       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             3_714.83       536.57     4_251.41       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             3_714.83       563.42     4_278.25       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   3_714.83     4_646.95     8_361.79       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             7_210.55       418.41     7_628.96       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             7_210.55       519.35     7_729.89       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             7_210.55       540.80     7_751.35       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   7_210.55     4_032.31    11_242.85       1.0000          1.0000         1.91
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

The results here are more favourable of the GPU acceleration. We go from ~90
seconds with exhaustive search on CPU to ~20 seconds on GPU for full kNN
generation; with the IVF variants, we can go from 10 seconds for the CPU based
version to ~7 seconds on the GPU one, a smaller effect than on for the
exhaustive search.

---

<details>
<summary><b>CPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        24.29    10_993.33    11_017.62       1.0000          1.0000       122.07
Exhaustive (self)                                         24.29   185_296.41   185_320.69       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                 1_006.43       699.32     1_705.75       1.0000          1.0000       122.25
IVF-nl353-np18 (query)                                 1_006.43       739.62     1_746.05       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                 1_006.43     1_056.11     2_062.54       1.0000          1.0000       122.25
IVF-nl353 (self)                                       1_006.43    16_909.87    17_916.30       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 1_707.34       653.69     2_361.03       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 1_707.34       739.17     2_446.51       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 1_707.34       921.42     2_628.75       1.0000          1.0000       122.32
IVF-nl500 (self)                                       1_707.34    14_746.18    16_453.52       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 3_483.43       565.45     4_048.87       0.9999          1.0000       122.42
IVF-nl707-np35 (query)                                 3_483.43       749.57     4_233.00       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 3_483.43       796.61     4_280.04       1.0000          1.0000       122.42
IVF-nl707 (self)                                       3_483.43    12_632.46    16_115.89       1.0000          1.0000       122.42
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
Exhaustive (query)                                        24.16    11_356.73    11_380.88       1.0000          1.0000       122.07
Exhaustive (self)                                         24.16   196_336.19   196_360.35       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    38.12     2_186.40     2_224.52       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     38.12    34_449.05    34_487.17       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                               974.57       592.65     1_567.22       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               974.57       631.20     1_605.77       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               974.57       802.02     1_776.59       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                     974.57     9_095.35    10_069.93       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             1_737.38       633.18     2_370.56       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             1_737.38       640.61     2_377.99       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             1_737.38       785.22     2_522.60       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   1_737.38     7_990.92     9_728.29       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             3_423.68       578.90     4_002.58       0.9999          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             3_423.68       631.37     4_055.05       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             3_423.68       660.60     4_084.28       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   3_423.68     6_898.98    10_322.65       1.0000          1.0000         1.91
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

The exhaustive kNN search on the CPU takes ~200 seconds (3+ minutes). Leveraging
the GPU, we cut this down to 30 seconds, a 4x speedup. The IVF CPU as a highly
optimised version takes 15 seconds, we can cut this down to 10 seconds. In
this case, the acceleration is more modest (similar as before) – the exhaustiv
search benefits from the large volume of data.

#### Increasing the number of samples

Results are becoming more pronounced with more samples and showing the
advantage of the GPU acceleration.

<details>
<summary><b>CPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.58    11_872.88    11_893.46       1.0000          1.0000       122.07
Exhaustive (self)                                         20.58   372_973.35   372_993.93       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 3_770.66       639.21     4_409.87       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 3_770.66       713.23     4_483.89       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 3_770.66       879.51     4_650.16       1.0000          1.0000       122.20
IVF-nl500 (self)                                       3_770.66    27_712.14    31_482.79       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 7_219.21       561.41     7_780.62       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 7_219.21       729.61     7_948.83       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 7_219.21       770.07     7_989.28       1.0000          1.0000       122.25
IVF-nl707 (self)                                       7_219.21    24_406.35    31_625.56       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                               13_784.38       487.86    14_272.24       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                               13_784.38       670.51    14_454.89       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                               13_784.38       753.78    14_538.16       1.0000          1.0000       122.32
IVF-nl1000 (self)                                     13_784.38    21_598.01    35_382.39       1.0000          1.0000       122.32
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
Exhaustive (query)                                        21.13    12_686.35    12_707.48       1.0000          1.0000       122.07
Exhaustive (self)                                         21.13   394_148.78   394_169.91       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    32.53     2_700.69     2_733.22       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     32.53    85_670.72    85_703.26       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             4_284.82       639.96     4_924.78       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             4_284.82       649.19     4_934.01       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             4_284.82       741.41     5_026.23       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   4_284.82    16_042.02    20_326.84       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             8_477.41       595.11     9_072.52       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             8_477.41       668.22     9_145.62       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             8_477.41       667.01     9_144.42       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   8_477.41    14_174.83    22_652.24       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                           15_701.81       542.27    16_244.07       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                           15_701.81       624.01    16_325.82       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                           15_701.81       665.15    16_366.96       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                 15_701.81    12_132.47    27_834.28       1.0000          1.0000         3.82
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
Exhaustive (query)                                        49.26    25_622.69    25_671.95       1.0000          1.0000       244.14
Exhaustive (self)                                         49.26   867_410.44   867_459.70       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_043.08     1_330.90     3_373.98       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 2_043.08     1_486.57     3_529.65       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 2_043.08     1_822.21     3_865.28       1.0000          1.0000       244.39
IVF-nl500 (self)                                       2_043.08    59_345.77    61_388.85       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 3_664.79     1_152.32     4_817.12       0.9999          1.0000       244.49
IVF-nl707-np35 (query)                                 3_664.79     1_516.05     5_180.84       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 3_664.79     1_592.53     5_257.32       1.0000          1.0000       244.49
IVF-nl707 (self)                                       3_664.79    51_991.29    55_656.09       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                                7_791.93       994.33     8_786.25       0.9998          1.0000       244.64
IVF-nl1000-np44 (query)                                7_791.93     1_374.73     9_166.65       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                                7_791.93     1_541.10     9_333.03       1.0000          1.0000       244.64
IVF-nl1000 (self)                                      7_791.93    44_353.35    52_145.27       1.0000          1.0000       244.64
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
Exhaustive (query)                                        47.40    25_887.97    25_935.38       1.0000          1.0000       244.14
Exhaustive (self)                                         47.40   879_240.32   879_287.72       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    77.20     4_346.42     4_423.62       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     77.20   137_383.97   137_461.17       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             2_002.56       990.79     2_993.35       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_002.56       973.50     2_976.06       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_002.56     1_187.55     3_190.12       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_002.56    29_215.29    31_217.85       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             3_766.05       892.42     4_658.47       0.9999          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             3_766.05       993.09     4_759.14       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             3_766.05     1_017.70     4_783.75       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   3_766.05    25_618.63    29_384.68       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            8_055.35       798.24     8_853.59       0.9998          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            8_055.35       978.06     9_033.41       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            8_055.35     1_026.44     9_081.79       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  8_055.35    22_085.02    30_140.38       1.0000          1.0000         3.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

The overall trends hold true. The exhaustive search becomes much faster on the
GPU, the IVF-based version gets a decent 2x bonus here. In this case, the
dimensionality starts being large enough that the GPU has enough data to
churn through and the difference with CPU versions becomes more apparent.

### CAGRA-type querying

The crate also offers a [CAGRA-style index](https://arxiv.org/abs/2308.15136),
combining GPU-accelerated NNDescent graph construction with CAGRA navigational
graph optimisation and beam search. The index is built in four phases:

1. **Random graph initialisation**: each node gets `build_k` random neighbours
   with computed distances, providing a baseline graph even before the forest
   runs.
2. **GPU forest initialisation**: a shallow random partition forest (default 20
   trees) groups nearby points into leaves. All-pairs distances within each
   leaf are computed on the GPU and merged into the graph via a proposal
   buffer. Leaf sizes are dynamically capped to fit within the GPU's shared
   memory budget (32 KB), so this scales correctly to high dimensions.
3. **GPU NNDescent iterations**: the standard local join loop runs entirely on
   the GPU. Each iteration builds reverse edges, evaluates (new, new) and
   (new, old) candidate pairs in shared memory, and merges proposals into the
   sorted graph. Convergence is checked by downloading a single `u32` counter
   per iteration. Typically converges in 4-6 iterations.
4. **CAGRA graph optimisation**: the NNDescent graph (at degree `build_k`) is
   pruned to degree `k` using rank-based detour counting, reverse edge
   construction, and forward/reverse merge. This produces a directed
   navigational graph with improved long-range reachability for beam search.

Querying uses a GPU beam search kernel: one workgroup per query, with the query
vector in shared memory, a sorted candidate queue, and a linear-probing hash
table for visited-node deduplication. Beam width and iteration limits are
scaled automatically based on `k` and the graph degree via
`CagraGpuSearchParams::from_graph()`. For small individual queries, a CPU
path is used that doesn't have the overhead of the GPU kernel launches.
Generally speaking, this index does not perform too well on very well separated
data. However, it does perform well on low-rank data.

#### Parameter guidance

The two key build parameters are `build_k` (internal NNDescent degree before
CAGRA pruning) and `refine_knn` (number of 2-hop refinement sweeps after
NNDescent convergence).

**Key parameters:**

* `build_k`: Internal NNDescent degree before CAGRA pruning. Defaults to 2 * k.
  Higher values give CAGRA more edges to select from when building the
  navigational graph, at the cost of build time. 3 * k shows diminishing returns.
* `refine_knn`: Number of 2-hop refinement sweeps after NNDescent convergence.
  Each sweep evaluates all neighbours-of-neighbours and merges improvements.
  Defaults to 0. Marginal benefit for beam search recall; primarily improves
  extract graph quality up to the forest ceiling.
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

Generally speaking CAGRA allows for very fast querying; however, the generation
of the index takes a bit more time compared to IVF for example. Also, it "fails"
in very well clustered data. It works better in data sets

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                     3.07     1_559.79     1_562.85       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.07    16_963.05    16_966.11       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.22       640.62       645.84       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.22     5_401.99     5_407.20       1.0000          1.0000        18.31
CAGRA-auto (query)                                       607.29       225.47       832.76       0.9412          1.0034        86.98
CAGRA-auto (self)                                        607.29       709.75     1_317.04       0.9375          1.0049        86.98
CAGRA-bw16 (query)                                       607.29       139.09       746.38       0.9195          1.0046        86.98
CAGRA-bw16 (self)                                        607.29       328.42       935.72       0.9146          1.0068        86.98
CAGRA-bw30 (query)                                       607.29       165.46       772.76       0.9391          1.0035        86.98
CAGRA-bw30 (self)                                        607.29       652.44     1_259.74       0.9354          1.0050        86.98
CAGRA-bw48 (query)                                       607.29       227.71       835.00       0.9559          1.0025        86.98
CAGRA-bw48 (self)                                        607.29     1_235.37     1_842.66       0.9528          1.0037        86.98
CAGRA-bw64 (query)                                       607.29       301.92       909.21       0.9651          1.0020        86.98
CAGRA-bw64 (self)                                        607.29     1_887.31     2_494.61       0.9625          1.0030        86.98
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
CPU-Exhaustive (query)                                     4.39     1_785.21     1_789.60       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      4.39    18_547.21    18_551.60       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     7.20       672.58       679.78       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      7.20     5_606.81     5_614.01       1.0000          1.0000        18.88
CAGRA-auto (query)                                       631.50       223.25       854.75       0.9423          1.0034        87.55
CAGRA-auto (self)                                        631.50       719.26     1_350.76       0.9393          1.0046        87.55
CAGRA-bw16 (query)                                       631.50       174.73       806.23       0.9206          1.0047        87.55
CAGRA-bw16 (self)                                        631.50       337.93       969.43       0.9169          1.0065        87.55
CAGRA-bw30 (query)                                       631.50       188.93       820.43       0.9402          1.0036        87.55
CAGRA-bw30 (self)                                        631.50       668.69     1_300.19       0.9372          1.0048        87.55
CAGRA-bw48 (query)                                       631.50       241.73       873.23       0.9564          1.0026        87.55
CAGRA-bw48 (self)                                        631.50     1_246.08     1_877.58       0.9544          1.0035        87.55
CAGRA-bw64 (query)                                       631.50       308.73       940.22       0.9656          1.0020        87.55
CAGRA-bw64 (self)                                        631.50     1_938.05     2_569.55       0.9638          1.0028        87.55
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
CPU-Exhaustive (query)                                     3.82     1_737.96     1_741.79       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.82    17_100.28    17_104.10       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.61       643.55       649.16       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.61     5_405.89     5_411.50       1.0000          1.0000        18.31
CAGRA-auto (query)                                       543.04       203.15       746.20       0.9951          1.0002        86.98
CAGRA-auto (self)                                        543.04       645.70     1_188.75       0.9969          1.0002        86.98
CAGRA-bw16 (query)                                       543.04       112.23       655.27       0.9867          1.0007        86.98
CAGRA-bw16 (self)                                        543.04       316.88       859.93       0.9924          1.0005        86.98
CAGRA-bw30 (query)                                       543.04       125.05       668.09       0.9946          1.0003        86.98
CAGRA-bw30 (self)                                        543.04       596.00     1_139.04       0.9966          1.0002        86.98
CAGRA-bw48 (query)                                       543.04       206.35       749.40       0.9977          1.0001        86.98
CAGRA-bw48 (self)                                        543.04     1_058.35     1_601.39       0.9985          1.0001        86.98
CAGRA-bw64 (query)                                       543.04       266.87       809.91       0.9986          1.0001        86.98
CAGRA-bw64 (self)                                        543.04     1_582.11     2_125.15       0.9991          1.0001        86.98
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
CPU-Exhaustive (query)                                     3.19     1_615.15     1_618.33       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.19    17_002.44    17_005.63       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.90       645.96       651.86       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.90     5_399.04     5_404.93       1.0000          1.0000        18.31
CAGRA-auto (query)                                       540.78       112.87       653.65       0.9985          1.0001        86.98
CAGRA-auto (self)                                        540.78       633.27     1_174.06       0.9986          1.0001        86.98
CAGRA-bw16 (query)                                       540.78       162.63       703.41       0.9952          1.0002        86.98
CAGRA-bw16 (self)                                        540.78       315.50       856.29       0.9956          1.0003        86.98
CAGRA-bw30 (query)                                       540.78       171.87       712.65       0.9983          1.0001        86.98
CAGRA-bw30 (self)                                        540.78       581.99     1_122.77       0.9984          1.0001        86.98
CAGRA-bw48 (query)                                       540.78       154.64       695.42       0.9994          1.0000        86.98
CAGRA-bw48 (self)                                        540.78     1_040.53     1_581.32       0.9994          1.0001        86.98
CAGRA-bw64 (query)                                       540.78       236.43       777.21       0.9997          1.0000        86.98
CAGRA-bw64 (self)                                        540.78     1_563.64     2_104.43       0.9997          1.0000        86.98
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
CPU-Exhaustive (query)                                    14.98     6_719.52     6_734.50       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     14.98    70_657.79    70_672.77       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    24.20     1_358.90     1_383.10       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     24.20    12_428.77    12_452.98       1.0000          1.0000        73.24
CAGRA-auto (query)                                     3_128.78       278.28     3_407.05       0.9941          1.0003       141.91
CAGRA-auto (self)                                      3_128.78       795.12     3_923.90       0.9932          1.0005       141.91
CAGRA-bw16 (query)                                     3_128.78       230.19     3_358.97       0.9876          1.0006       141.91
CAGRA-bw16 (self)                                      3_128.78       412.13     3_540.90       0.9855          1.0009       141.91
CAGRA-bw30 (query)                                     3_128.78       256.33     3_385.10       0.9937          1.0003       141.91
CAGRA-bw30 (self)                                      3_128.78       742.88     3_871.65       0.9927          1.0005       141.91
CAGRA-bw48 (query)                                     3_128.78       310.60     3_439.37       0.9970          1.0002       141.91
CAGRA-bw48 (self)                                      3_128.78     1_289.39     4_418.17       0.9964          1.0003       141.91
CAGRA-bw64 (query)                                     3_128.78       376.76     3_505.54       0.9983          1.0001       141.91
CAGRA-bw64 (self)                                      3_128.78     1_888.82     5_017.60       0.9978          1.0002       141.91
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Larger data sets

Let's test CAGRA similar to IVF GPU on larger data sets.

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D (Exhaustive vs CAGRA beam search)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.70     9_087.86     9_099.55       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     11.70    92_992.88    93_004.57       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.86     2_315.22     2_334.07       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.86    21_318.29    21_337.14       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_450.23       435.10     2_885.33       0.9950          1.0002       175.48
CAGRA-auto (self)                                      2_450.23     1_201.86     3_652.09       0.9939          1.0004       175.48
CAGRA-bw16 (query)                                     2_450.23       343.62     2_793.85       0.9885          1.0005       175.48
CAGRA-bw16 (self)                                      2_450.23       597.10     3_047.33       0.9865          1.0008       175.48
CAGRA-bw30 (query)                                     2_450.23       371.62     2_821.85       0.9945          1.0003       175.48
CAGRA-bw30 (self)                                      2_450.23     1_099.16     3_549.40       0.9934          1.0004       175.48
CAGRA-bw48 (query)                                     2_450.23       459.29     2_909.52       0.9975          1.0001       175.48
CAGRA-bw48 (self)                                      2_450.23     2_025.10     4_475.33       0.9969          1.0002       175.48
CAGRA-bw64 (query)                                     2_450.23       580.42     3_030.65       0.9985          1.0001       175.48
CAGRA-bw64 (self)                                      2_450.23     3_018.10     5_468.33       0.9981          1.0001       175.48
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
CPU-Exhaustive (query)                                    24.75    21_545.60    21_570.36       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     24.75   203_144.70   203_169.45       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    41.62     3_636.89     3_678.51       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     41.62    34_539.38    34_581.00       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_810.45       607.05     7_417.50       0.9925          1.0004       236.51
CAGRA-auto (self)                                      6_810.45     1_384.59     8_195.03       0.9910          1.0006       236.51
CAGRA-bw16 (query)                                     6_810.45       547.04     7_357.48       0.9849          1.0008       236.51
CAGRA-bw16 (self)                                      6_810.45       726.15     7_536.60       0.9817          1.0011       236.51
CAGRA-bw30 (query)                                     6_810.45       590.89     7_401.33       0.9919          1.0004       236.51
CAGRA-bw30 (self)                                      6_810.45     1_295.44     8_105.88       0.9902          1.0006       236.51
CAGRA-bw48 (query)                                     6_810.45       687.80     7_498.24       0.9958          1.0002       236.51
CAGRA-bw48 (self)                                      6_810.45     2_238.45     9_048.90       0.9950          1.0003       236.51
CAGRA-bw64 (query)                                     6_810.45       760.03     7_570.47       0.9975          1.0001       236.51
CAGRA-bw64 (self)                                      6_810.45     3_318.98    10_129.43       0.9969          1.0002       236.51
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
CPU-Exhaustive (query)                                    21.18    44_344.25    44_365.43       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     21.18   417_682.70   417_703.88       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.48     8_711.46     8_747.94       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.48    85_625.35    85_661.83       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_253.53       708.48     6_962.01       0.9911          1.0005       350.95
CAGRA-auto (self)                                      6_253.53     2_450.97     8_704.50       0.9894          1.0007       350.95
CAGRA-bw16 (query)                                     6_253.53       684.62     6_938.15       0.9826          1.0009       350.95
CAGRA-bw16 (self)                                      6_253.53     1_201.69     7_455.22       0.9796          1.0013       350.95
CAGRA-bw30 (query)                                     6_253.53       742.53     6_996.06       0.9904          1.0006       350.95
CAGRA-bw30 (self)                                      6_253.53     2_260.80     8_514.33       0.9886          1.0008       350.95
CAGRA-bw48 (query)                                     6_253.53       932.54     7_186.07       0.9949          1.0003       350.95
CAGRA-bw48 (self)                                      6_253.53     4_050.69    10_304.22       0.9939          1.0004       350.95
CAGRA-bw64 (query)                                     6_253.53     1_120.70     7_374.23       0.9968          1.0002       350.95
CAGRA-bw64 (self)                                      6_253.53     6_073.92    12_327.45       0.9961          1.0003       350.95
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
CPU-Exhaustive (query)                                    47.43    86_520.46    86_567.90       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     47.43   859_506.75   859_554.18       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    70.58    14_046.74    14_117.32       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     70.58   138_452.27   138_522.85       1.0000          1.0000       244.14
CAGRA-auto (query)                                    13_937.44     1_273.08    15_210.52       0.9876          1.0007       473.02
CAGRA-auto (self)                                     13_937.44     2_845.84    16_783.27       0.9851          1.0010       473.02
CAGRA-bw16 (query)                                    13_937.44     1_117.26    15_054.70       0.9776          1.0013       473.02
CAGRA-bw16 (self)                                     13_937.44     1_484.24    15_421.68       0.9732          1.0018       473.02
CAGRA-bw30 (query)                                    13_937.44     1_220.69    15_158.13       0.9868          1.0008       473.02
CAGRA-bw30 (self)                                     13_937.44     2_624.14    16_561.58       0.9841          1.0011       473.02
CAGRA-bw48 (query)                                    13_937.44     1_454.11    15_391.54       0.9925          1.0004       473.02
CAGRA-bw48 (self)                                     13_937.44     4_618.72    18_556.15       0.9910          1.0006       473.02
CAGRA-bw64 (query)                                    13_937.44     1_682.26    15_619.70       0.9950          1.0003       473.02
CAGRA-bw64 (self)                                     13_937.44     6_846.82    20_784.26       0.9939          1.0004       473.02
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Two-tier kNN generation

For downstream tasks that require a full kNN graph (e.g. BBKNN, MNN, UMAP,
Leiden clustering), the index offers three paths with different speed/accuracy
trade-offs:

| Method | Mechanism | Typical recall | Use case |
|--------|-----------|---------------|----------|
| **Extract** | Direct reshape of the NNDescent graph. No search performed. | ~0.9 | Fast, however, lowever precision. |
| **Self-beam** | GPU beam search over the CAGRA navigational graph for every vector in the index. | 0.99 | Production kNN graphs for all types of applications. |

Below are examples of kNN generation. The dimensions are specifically kept
quite low to mimic single cell situations. This is where the CAGRA-style part
is quite performant and can be used to quickly generate kNN graphs from the
data... To run these, you can use:

```bash
cargo run --example knn_comparison_cagra --features gpu --release
```

The application idea here is to use these for large single cell data sets in
which the kNN can be further accelerated.

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 32 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                              8.63    15_058.99    15_067.62       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   5_135.65     1_176.75     6_312.40       1.0000          1.0000       279.88
GPU-NND bk=1x refine=0 (extract)                         856.38        42.90       899.28       0.8935          1.0888       144.96
GPU-NND bk=1x refine=0 (self-beam)                       856.38     1_107.45     1_963.83       0.9955          1.0004       144.96
GPU-NND bk=1x refine=1 (extract)                         882.86        42.04       924.90       0.9227          1.0848       144.96
GPU-NND bk=1x refine=1 (self-beam)                       882.86     1_106.33     1_989.19       0.9960          1.0003       144.96
GPU-NND bk=1x refine=2 (extract)                         947.10        42.20       989.30       0.9242          1.0847       144.96
GPU-NND bk=1x refine=2 (self-beam)                       947.10     1_102.97     2_050.07       0.9961          1.0003       144.96
GPU-NND bk=2x refine=0 (extract)                       1_264.38        42.92     1_307.30       0.9267          1.0844       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_264.38     1_107.97     2_372.35       0.9980          1.0001       144.96
GPU-NND bk=2x refine=1 (extract)                       1_543.94        41.83     1_585.78       0.9329          1.0837       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_543.94     1_105.87     2_649.81       0.9984          1.0001       144.96
GPU-NND bk=2x refine=2 (extract)                       1_857.92        41.52     1_899.44       0.9330          1.0837       144.96
GPU-NND bk=2x refine=2 (self-beam)                     1_857.92     1_111.74     2_969.66       0.9985          1.0001       144.96
GPU-NND bk=3x refine=0 (extract)                       2_215.86        42.33     2_258.19       0.9304          1.0840       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_215.86     1_107.52     3_323.38       0.9983          1.0001       144.96
GPU-NND bk=3x refine=1 (extract)                       2_936.04        43.69     2_979.73       0.9333          1.0837       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_936.04     1_105.35     4_041.39       0.9986          1.0001       144.96
GPU-NND bk=3x refine=2 (extract)                       3_612.31        42.70     3_655.01       0.9333          1.0837       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_612.31     1_107.84     4_720.15       0.9985          1.0001       144.96
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             19.52    21_554.08    21_573.60       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   6_609.26     1_888.30     8_497.56       1.0000          1.0000       377.92
GPU-NND bk=1x refine=0 (extract)                       1_237.68        43.31     1_280.99       0.8651          1.0910       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_237.68     1_298.93     2_536.61       0.9881          1.0010       175.48
GPU-NND bk=1x refine=1 (extract)                       1_692.22        42.90     1_735.12       0.9104          1.0848       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_692.22     1_227.16     2_919.38       0.9896          1.0008       175.48
GPU-NND bk=1x refine=2 (extract)                       2_153.49        43.24     2_196.73       0.9139          1.0844       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_153.49     1_219.60     3_373.09       0.9898          1.0008       175.48
GPU-NND bk=2x refine=0 (extract)                       2_055.55        42.03     2_097.58       0.9189          1.0838       175.48
GPU-NND bk=2x refine=0 (self-beam)                     2_055.55     1_223.35     3_278.90       0.9941          1.0003       175.48
GPU-NND bk=2x refine=1 (extract)                       3_461.93        42.41     3_504.33       0.9318          1.0825       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_461.93     1_229.77     4_691.69       0.9954          1.0002       175.48
GPU-NND bk=2x refine=2 (extract)                       4_863.21        42.20     4_905.41       0.9320          1.0824       175.48
GPU-NND bk=2x refine=2 (self-beam)                     4_863.21     1_226.51     6_089.72       0.9955          1.0002       175.48
GPU-NND bk=3x refine=0 (extract)                       4_700.89        42.83     4_743.72       0.9280          1.0828       175.48
GPU-NND bk=3x refine=0 (self-beam)                     4_700.89     1_230.74     5_931.63       0.9952          1.0002       175.48
GPU-NND bk=3x refine=1 (extract)                       7_075.83        41.77     7_117.60       0.9332          1.0823       175.48
GPU-NND bk=3x refine=1 (self-beam)                     7_075.83     1_228.89     8_304.71       0.9959          1.0002       175.48
GPU-NND bk=3x refine=2 (extract)                       9_520.58        41.77     9_562.35       0.9332          1.0823       175.48
GPU-NND bk=3x refine=2 (self-beam)                     9_520.58     1_228.99    10_749.58       0.9958          1.0002       175.48
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 32 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             17.49    59_596.89    59_614.38       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  11_486.31     2_966.89    14_453.19       1.0000          1.0000       627.78
GPU-NND bk=1x refine=0 (extract)                       1_632.75        87.63     1_720.38       0.8738          1.0914       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_632.75     2_275.27     3_908.01       0.9917          1.0007       289.92
GPU-NND bk=1x refine=1 (extract)                       1_873.60        84.99     1_958.59       0.9148          1.0854       289.92
GPU-NND bk=1x refine=1 (self-beam)                     1_873.60     2_226.85     4_100.46       0.9928          1.0006       289.92
GPU-NND bk=1x refine=2 (extract)                       2_260.81        83.01     2_343.82       0.9178          1.0850       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_260.81     2_212.08     4_472.89       0.9930          1.0006       289.92
GPU-NND bk=2x refine=0 (extract)                       2_486.63        84.11     2_570.75       0.9229          1.0845       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_486.63     2_227.08     4_713.71       0.9966          1.0002       289.92
GPU-NND bk=2x refine=1 (extract)                       3_559.67        83.34     3_643.02       0.9324          1.0834       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_559.67     2_225.32     5_784.99       0.9974          1.0001       289.92
GPU-NND bk=2x refine=2 (extract)                       4_673.60        83.75     4_757.34       0.9326          1.0834       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_673.60     2_227.15     6_900.75       0.9974          1.0001       289.92
GPU-NND bk=3x refine=0 (extract)                       4_626.95        84.83     4_711.78       0.9294          1.0838       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_626.95     2_237.93     6_864.87       0.9973          1.0001       289.92
GPU-NND bk=3x refine=1 (extract)                       6_628.02        84.29     6_712.31       0.9332          1.0834       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_628.02     2_227.61     8_855.63       0.9976          1.0001       289.92
GPU-NND bk=3x refine=2 (extract)                       8_623.23        83.33     8_706.57       0.9333          1.0834       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_623.23     2_229.57    10_852.80       0.9976          1.0001       289.92
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             36.75    85_707.32    85_744.07       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  14_805.16     4_277.21    19_082.37       1.0000          1.0000       793.86
GPU-NND bk=1x refine=0 (extract)                       2_519.43        84.53     2_603.97       0.8321          1.0957       350.95
GPU-NND bk=1x refine=0 (self-beam)                     2_519.43     2_459.26     4_978.70       0.9786          1.0019       350.95
GPU-NND bk=1x refine=1 (extract)                       3_892.65        82.57     3_975.21       0.8940          1.0864       350.95
GPU-NND bk=1x refine=1 (self-beam)                     3_892.65     2_459.90     6_352.55       0.9813          1.0016       350.95
GPU-NND bk=1x refine=2 (extract)                       5_364.39        83.52     5_447.90       0.9005          1.0856       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_364.39     2_446.56     7_810.94       0.9819          1.0015       350.95
GPU-NND bk=2x refine=0 (extract)                       4_263.67        82.47     4_346.15       0.9101          1.0846       350.95
GPU-NND bk=2x refine=0 (self-beam)                     4_263.67     2_468.08     6_731.75       0.9901          1.0006       350.95
GPU-NND bk=2x refine=1 (extract)                       9_292.04        85.68     9_377.71       0.9301          1.0823       350.95
GPU-NND bk=2x refine=1 (self-beam)                     9_292.04     2_454.98    11_747.01       0.9924          1.0004       350.95
GPU-NND bk=2x refine=2 (extract)                      14_455.81        82.54    14_538.34       0.9308          1.0823       350.95
GPU-NND bk=2x refine=2 (self-beam)                    14_455.81     2_464.85    16_920.66       0.9927          1.0004       350.95
GPU-NND bk=3x refine=0 (extract)                       9_919.64        82.04    10_001.68       0.9254          1.0828       350.95
GPU-NND bk=3x refine=0 (self-beam)                     9_919.64     2_461.25    12_380.89       0.9924          1.0004       350.95
GPU-NND bk=3x refine=1 (extract)                      18_828.17        82.26    18_910.43       0.9329          1.0821       350.95
GPU-NND bk=3x refine=1 (self-beam)                    18_828.17     2_478.67    21_306.84       0.9936          1.0003       350.95
GPU-NND bk=3x refine=2 (extract)                      27_600.67        84.01    27_684.69       0.9330          1.0821       350.95
GPU-NND bk=3x refine=2 (self-beam)                    27_600.67     2_463.26    30_063.94       0.9937          1.0003       350.95
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 32 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             33.93   238_007.37   238_041.29       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  24_857.71     6_253.09    31_110.80       1.0000          1.0000      1143.55
GPU-NND bk=1x refine=0 (extract)                       3_434.63       171.42     3_606.05       0.8562          1.0938       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_434.63     4_627.25     8_061.88       0.9870          1.0011       579.83
GPU-NND bk=1x refine=1 (extract)                       4_371.34       173.03     4_544.37       0.9061          1.0862       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_371.34     4_495.31     8_866.64       0.9888          1.0009       579.83
GPU-NND bk=1x refine=2 (extract)                       5_389.55       170.46     5_560.01       0.9107          1.0856       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_389.55     4_487.82     9_877.37       0.9892          1.0009       579.83
GPU-NND bk=2x refine=0 (extract)                       5_314.34       171.38     5_485.73       0.9186          1.0847       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_314.34     4_534.73     9_849.07       0.9947          1.0003       579.83
GPU-NND bk=2x refine=1 (extract)                       8_519.99       167.45     8_687.44       0.9317          1.0832       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_519.99     4_522.79    13_042.78       0.9960          1.0002       579.83
GPU-NND bk=2x refine=2 (extract)                      11_837.19       167.09    12_004.28       0.9321          1.0832       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_837.19     4_514.22    16_351.41       0.9961          1.0002       579.83
GPU-NND bk=3x refine=0 (extract)                       9_433.87       168.57     9_602.43       0.9280          1.0836       579.83
GPU-NND bk=3x refine=0 (self-beam)                     9_433.87     4_539.54    13_973.41       0.9960          1.0002       579.83
GPU-NND bk=3x refine=1 (extract)                      15_510.86       169.02    15_679.88       0.9332          1.0831       579.83
GPU-NND bk=3x refine=1 (self-beam)                    15_510.86     4_524.80    20_035.66       0.9966          1.0001       579.83
GPU-NND bk=3x refine=2 (extract)                      21_633.71       165.78    21_799.49       0.9332          1.0831       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_633.71     4_525.83    26_159.55       0.9966          1.0001       579.83
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             68.96   341_237.55   341_306.50       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  33_135.45     9_839.08    42_974.53       0.9999          1.0000      1659.75
GPU-NND bk=1x refine=0 (extract)                       5_078.19       173.18     5_251.37       0.8034          1.1002       701.90
GPU-NND bk=1x refine=0 (self-beam)                     5_078.19     5_034.32    10_112.51       0.9687          1.0028       701.90
GPU-NND bk=1x refine=1 (extract)                       9_097.46       173.19     9_270.65       0.8773          1.0884       701.90
GPU-NND bk=1x refine=1 (self-beam)                     9_097.46     4_959.52    14_056.97       0.9728          1.0024       701.90
GPU-NND bk=1x refine=2 (extract)                      13_024.07       170.04    13_194.11       0.8871          1.0871       701.90
GPU-NND bk=1x refine=2 (self-beam)                    13_024.07     4_968.84    17_992.91       0.9738          1.0022       701.90
GPU-NND bk=2x refine=0 (extract)                       8_943.15       170.22     9_113.37       0.9020          1.0853       701.90
GPU-NND bk=2x refine=0 (self-beam)                     8_943.15     5_017.59    13_960.74       0.9861          1.0009       701.90
GPU-NND bk=2x refine=1 (extract)                      21_231.31       166.03    21_397.34       0.9281          1.0824       701.90
GPU-NND bk=2x refine=1 (self-beam)                    21_231.31     4_988.86    26_220.17       0.9896          1.0005       701.90
GPU-NND bk=2x refine=2 (extract)                      33_800.43       167.03    33_967.47       0.9293          1.0822       701.90
GPU-NND bk=2x refine=2 (self-beam)                    33_800.43     5_014.46    38_814.90       0.9900          1.0005       701.90
GPU-NND bk=3x refine=0 (extract)                      21_259.42       168.19    21_427.61       0.9230          1.0829       701.90
GPU-NND bk=3x refine=0 (self-beam)                    21_259.42     5_019.03    26_278.45       0.9899          1.0005       701.90
GPU-NND bk=3x refine=1 (extract)                      43_292.76       170.00    43_462.76       0.9326          1.0819       701.90
GPU-NND bk=3x refine=1 (self-beam)                    43_292.76     5_004.84    48_297.61       0.9916          1.0004       701.90
GPU-NND bk=3x refine=2 (extract)                      65_982.34       167.35    66_149.69       0.9328          1.0819       701.90
GPU-NND bk=3x refine=2 (self-beam)                    65_982.34     5_003.22    70_985.57       0.9917          1.0003       701.90
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Let's do one large data set with 2.5m samples at 32 dimensions and see what
happens ... ?

<details>
<summary><b>Generation of a kNN graph with CAGRA (2.5m samples; 32 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             84.88 1_477_861.10 1_477_945.98       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  69_453.70    19_653.87    89_107.57       0.9999          1.0000      3254.84
GPU-NND bk=1x refine=0 (extract)                       7_350.75       432.04     7_782.80       0.7873          1.1049      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_350.75    11_677.91    19_028.66       0.9671          1.0032      1449.59
GPU-NND bk=1x refine=1 (extract)                      10_660.12       449.44    11_109.56       0.8696          1.0903      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    10_660.12    11_617.64    22_277.75       0.9724          1.0026      1449.59
GPU-NND bk=1x refine=2 (extract)                      13_998.16       454.30    14_452.47       0.8825          1.0884      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    13_998.16    11_639.03    25_637.19       0.9738          1.0024      1449.59
GPU-NND bk=2x refine=0 (extract)                      12_998.71       430.99    13_429.70       0.9035          1.0858      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    12_998.71    11_660.17    24_658.89       0.9885          1.0008      1449.59
GPU-NND bk=2x refine=1 (extract)                      23_972.22       423.67    24_395.89       0.9285          1.0828      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    23_972.22    11_620.45    35_592.68       0.9918          1.0004      1449.59
GPU-NND bk=2x refine=2 (extract)                      35_389.65       423.83    35_813.48       0.9297          1.0827      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    35_389.65    11_628.06    47_017.71       0.9922          1.0004      1449.59
GPU-NND bk=3x refine=0 (extract)                      24_600.50       427.76    25_028.26       0.9237          1.0833      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    24_600.50    11_655.18    36_255.68       0.9921          1.0004      1449.59
GPU-NND bk=3x refine=1 (extract)                      45_937.54       425.93    46_363.47       0.9327          1.0824      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    45_937.54    11_674.17    57_611.71       0.9936          1.0003      1449.59
GPU-NND bk=3x refine=2 (extract)                      67_244.68       429.45    67_674.13       0.9329          1.0824      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    67_244.68    11_677.16    78_921.83       0.9937          1.0003      1449.59
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Especially on larger data sets, we can accelerate the queries substantially
and get up to 2x to 3x speed increases to generate the full kNN graph with
Recall@k of ≥0.99. If you are okay with a graph that has Recall ≥0.9 you
can do that in <10 seconds on a million samples or ~30 seconds on 2.5 million
samples (with n_dim = 32 dim). Also, the data is very contrived here... On real
data, NNDescent will have to do quite a few iterations. The Annoy
initialisations are already very good, so the CPU version basically has no
need for refining the kNN graph. On real data, the GPU outperforms more
substantially.

## Conclusions

GPU acceleration in the setting of the `wgpu` backend only starts making sense
with large indices and large dimensionality (assuming you can hold the data
in VRAM or unified memory for Apple Silicon). With smaller dimensionalities and
less samples, the overhead of launching the GPU kernels does not give
substantial performance benefits over the highly optimised CPU code. Exhaustive
searches over larger data sets however become more viable with GPU acceleration
and it here where some of the biggest gains can be observed. To note, these
implemetations are not designed (and cannot) compete with what is possible
on data centre GPUs with cuBLAS under the hood! They serve as an acceleration
in specific situations and were designed to enable fast kNN generation for
1m to 10m sample situations with lower dimensions (think single cell).

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
