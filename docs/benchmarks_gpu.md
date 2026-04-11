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
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.36     1_652.33     1_655.69       1.0000     0.000000        18.31
Exhaustive (self)                                          3.36    17_203.07    17_206.43       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.29       643.18       648.47       1.0000     0.000005        18.31
GPU-Exhaustive (self)                                      5.29     5_449.94     5_455.23       1.0000     0.000005        18.31
IVF-GPU-nl273-np13 (query)                               422.18       303.16       725.34       0.9942     0.017444         1.15
IVF-GPU-nl273-np16 (query)                               422.18       355.20       777.38       0.9987     0.002855         1.15
IVF-GPU-nl273-np23 (query)                               422.18       428.10       850.28       1.0000     0.000005         1.15
IVF-GPU-nl273 (self)                                     422.18     1_620.54     2_042.71       1.0000     0.000005         1.15
IVF-GPU-nl387-np19 (query)                               835.20       306.48     1_141.68       0.9976     0.012096         1.15
IVF-GPU-nl387-np27 (query)                               835.20       401.93     1_237.12       1.0000     0.000005         1.15
IVF-GPU-nl387 (self)                                     835.20     1_362.07     2_197.26       1.0000     0.000005         1.15
IVF-GPU-nl547-np23 (query)                             1_617.27       307.43     1_924.70       0.9958     0.015835         1.15
IVF-GPU-nl547-np27 (query)                             1_617.27       337.09     1_954.36       0.9991     0.002102         1.15
IVF-GPU-nl547-np33 (query)                             1_617.27       349.04     1_966.31       0.9999     0.000146         1.15
IVF-GPU-nl547 (self)                                   1_617.27     1_369.40     2_986.67       0.9999     0.000153         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.15     1_651.51     1_655.67       1.0000     0.000000        18.88
Exhaustive (self)                                          4.15    17_170.15    17_174.31       1.0000     0.000000        18.88
GPU-Exhaustive (query)                                     6.43       664.50       670.93       1.0000     0.000000        18.88
GPU-Exhaustive (self)                                      6.43     5_623.34     5_629.78       1.0000     0.000000        18.88
IVF-GPU-nl273-np13 (query)                               383.64       300.04       683.68       0.9946     0.000011         1.15
IVF-GPU-nl273-np16 (query)                               383.64       346.69       730.34       0.9989     0.000002         1.15
IVF-GPU-nl273-np23 (query)                               383.64       414.95       798.59       1.0000     0.000000         1.15
IVF-GPU-nl273 (self)                                     383.64     1_601.62     1_985.26       1.0000     0.000000         1.15
IVF-GPU-nl387-np19 (query)                               754.78       290.71     1_045.49       0.9979     0.000007         1.15
IVF-GPU-nl387-np27 (query)                               754.78       414.36     1_169.13       1.0000     0.000000         1.15
IVF-GPU-nl387 (self)                                     754.78     1_387.30     2_142.08       1.0000     0.000000         1.15
IVF-GPU-nl547-np23 (query)                             1_449.63       303.81     1_753.44       0.9957     0.000012         1.15
IVF-GPU-nl547-np27 (query)                             1_449.63       322.16     1_771.79       0.9991     0.000002         1.15
IVF-GPU-nl547-np33 (query)                             1_449.63       353.78     1_803.40       0.9999     0.000000         1.15
IVF-GPU-nl547 (self)                                   1_449.63     1_445.76     2_895.38       0.9999     0.000000         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.40     1_673.45     1_676.85       1.0000     0.000000        18.31
Exhaustive (self)                                          3.40    17_488.19    17_491.60       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.44       644.01       649.46       1.0000     0.000000        18.31
GPU-Exhaustive (self)                                      5.44     5_415.57     5_421.01       1.0000     0.000000        18.31
IVF-GPU-nl273-np13 (query)                               401.17       288.43       689.60       1.0000     0.000008         1.15
IVF-GPU-nl273-np16 (query)                               401.17       240.54       641.71       1.0000     0.000000         1.15
IVF-GPU-nl273-np23 (query)                               401.17       237.13       638.30       1.0000     0.000000         1.15
IVF-GPU-nl273 (self)                                     401.17     1_539.62     1_940.79       1.0000     0.000000         1.15
IVF-GPU-nl387-np19 (query)                               782.35       293.20     1_075.55       1.0000     0.000000         1.15
IVF-GPU-nl387-np27 (query)                               782.35       385.09     1_167.44       1.0000     0.000000         1.15
IVF-GPU-nl387 (self)                                     782.35     1_306.27     2_088.62       1.0000     0.000000         1.15
IVF-GPU-nl547-np23 (query)                             1_509.25       160.40     1_669.65       1.0000     0.000000         1.15
IVF-GPU-nl547-np27 (query)                             1_509.25       298.49     1_807.74       1.0000     0.000000         1.15
IVF-GPU-nl547-np33 (query)                             1_509.25       343.85     1_853.10       1.0000     0.000000         1.15
IVF-GPU-nl547 (self)                                   1_509.25     1_240.48     2_749.73       1.0000     0.000000         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.23     1_679.13     1_682.36       1.0000     0.000000        18.31
Exhaustive (self)                                          3.23    17_515.02    17_518.25       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.68       649.27       654.95       1.0000     0.000001        18.31
GPU-Exhaustive (self)                                      5.68     5_418.49     5_424.16       1.0000     0.000001        18.31
IVF-GPU-nl273-np13 (query)                               414.65       282.35       697.00       1.0000     0.000004         1.15
IVF-GPU-nl273-np16 (query)                               414.65       333.57       748.22       1.0000     0.000001         1.15
IVF-GPU-nl273-np23 (query)                               414.65       419.09       833.75       1.0000     0.000001         1.15
IVF-GPU-nl273 (self)                                     414.65     1_480.47     1_895.12       1.0000     0.000001         1.15
IVF-GPU-nl387-np19 (query)                               796.77       283.76     1_080.53       1.0000     0.000001         1.15
IVF-GPU-nl387-np27 (query)                               796.77       403.13     1_199.89       1.0000     0.000001         1.15
IVF-GPU-nl387 (self)                                     796.77     1_253.93     2_050.70       1.0000     0.000001         1.15
IVF-GPU-nl547-np23 (query)                             1_561.80       151.01     1_712.82       1.0000     0.000001         1.15
IVF-GPU-nl547-np27 (query)                             1_561.80       291.49     1_853.30       1.0000     0.000001         1.15
IVF-GPU-nl547-np33 (query)                             1_561.80       351.64     1_913.44       1.0000     0.000001         1.15
IVF-GPU-nl547 (self)                                   1_561.80     1_239.76     2_801.57       1.0000     0.000001         1.15
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.50     6_632.52     6_647.02       1.0000     0.000000        73.24
Exhaustive (self)                                         14.50    70_457.69    70_472.19       1.0000     0.000000        73.24
GPU-Exhaustive (query)                                    23.85     1_359.04     1_382.89       1.0000     0.000027        73.24
GPU-Exhaustive (self)                                     23.85    12_456.84    12_480.68       1.0000     0.000027        73.24
IVF-GPU-nl273-np13 (query)                               487.88       452.12       940.00       0.9996     0.006405         1.15
IVF-GPU-nl273-np16 (query)                               487.88       513.50     1_001.38       1.0000     0.000161         1.15
IVF-GPU-nl273-np23 (query)                               487.88       638.90     1_126.78       1.0000     0.000027         1.15
IVF-GPU-nl273 (self)                                     487.88     3_906.78     4_394.66       1.0000     0.000027         1.15
IVF-GPU-nl387-np19 (query)                               843.36       482.44     1_325.80       1.0000     0.000027         1.15
IVF-GPU-nl387-np27 (query)                               843.36       592.27     1_435.63       1.0000     0.000027         1.15
IVF-GPU-nl387 (self)                                     843.36     3_377.40     4_220.76       1.0000     0.000027         1.15
IVF-GPU-nl547-np23 (query)                             1_762.35       324.26     2_086.61       0.9998     0.002357         1.15
IVF-GPU-nl547-np27 (query)                             1_762.35       450.35     2_212.70       1.0000     0.000211         1.15
IVF-GPU-nl547-np33 (query)                             1_762.35       544.41     2_306.75       1.0000     0.000027         1.15
IVF-GPU-nl547 (self)                                   1_762.35     3_124.10     4_886.44       1.0000     0.000027         1.15
--------------------------------------------------------------------------------------------------------------------------------

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
================================================================================================================================
Benchmark: 250k samples, 64D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.43     5_402.26     5_413.69       1.0000     0.000000        61.04
Exhaustive (self)                                         11.43    91_939.76    91_951.19       1.0000     0.000000        61.04
IVF-nl353-np17 (query)                                 1_164.75       346.70     1_511.45       1.0000     0.000000        61.12
IVF-nl353-np18 (query)                                 1_164.75       358.10     1_522.85       1.0000     0.000000        61.12
IVF-nl353-np26 (query)                                 1_164.75       513.95     1_678.70       1.0000     0.000000        61.12
IVF-nl353 (self)                                       1_164.75     7_686.21     8_850.96       1.0000     0.000000        61.12
IVF-nl500-np22 (query)                                 2_334.55       318.35     2_652.90       1.0000     0.000016        61.16
IVF-nl500-np25 (query)                                 2_334.55       355.39     2_689.94       1.0000     0.000000        61.16
IVF-nl500-np31 (query)                                 2_334.55       445.70     2_780.25       1.0000     0.000000        61.16
IVF-nl500 (self)                                       2_334.55     6_158.45     8_493.00       1.0000     0.000000        61.16
IVF-nl707-np26 (query)                                 4_566.56       296.63     4_863.20       1.0000     0.000025        61.21
IVF-nl707-np35 (query)                                 4_566.56       438.13     5_004.69       1.0000     0.000000        61.21
IVF-nl707-np37 (query)                                 4_566.56       483.20     5_049.76       1.0000     0.000000        61.21
IVF-nl707 (self)                                       4_566.56     5_965.23    10_531.80       1.0000     0.000000        61.21
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.86     5_510.16     5_522.02       1.0000     0.000000        61.04
Exhaustive (self)                                         11.86    91_370.27    91_382.13       1.0000     0.000000        61.04
GPU-Exhaustive (query)                                    18.29     1_417.33     1_435.62       1.0000     0.000003        61.04
GPU-Exhaustive (self)                                     18.29    21_521.52    21_539.81       1.0000     0.000003        61.04
IVF-GPU-nl353-np17 (query)                             1_238.33       479.81     1_718.14       1.0000     0.000003         1.91
IVF-GPU-nl353-np18 (query)                             1_238.33       507.60     1_745.93       1.0000     0.000003         1.91
IVF-GPU-nl353-np26 (query)                             1_238.33       615.15     1_853.47       1.0000     0.000003         1.91
IVF-GPU-nl353 (self)                                   1_238.33     5_107.35     6_345.68       1.0000     0.000003         1.91
IVF-GPU-nl500-np22 (query)                             2_382.70       493.10     2_875.80       1.0000     0.000019         1.91
IVF-GPU-nl500-np25 (query)                             2_382.70       546.43     2_929.13       1.0000     0.000003         1.91
IVF-GPU-nl500-np31 (query)                             2_382.70       497.67     2_880.38       1.0000     0.000003         1.91
IVF-GPU-nl500 (self)                                   2_382.70     4_478.29     6_861.00       1.0000     0.000003         1.91
IVF-GPU-nl707-np26 (query)                             4_445.46       427.89     4_873.35       1.0000     0.000028         1.91
IVF-GPU-nl707-np35 (query)                             4_445.46       511.75     4_957.22       1.0000     0.000003         1.91
IVF-GPU-nl707-np37 (query)                             4_445.46       535.74     4_981.21       1.0000     0.000003         1.91
IVF-GPU-nl707 (self)                                   4_445.46     4_031.69     8_477.15       1.0000     0.000003         1.91
--------------------------------------------------------------------------------------------------------------------------------

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
================================================================================================================================
Benchmark: 250k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        24.99    11_692.10    11_717.09       1.0000     0.000000       122.07
Exhaustive (self)                                         24.99   190_994.00   191_018.98       1.0000     0.000000       122.07
IVF-nl353-np17 (query)                                   772.89       712.62     1_485.51       0.9999     0.001614       122.25
IVF-nl353-np18 (query)                                   772.89       749.00     1_521.88       0.9999     0.000721       122.25
IVF-nl353-np26 (query)                                   772.89     1_063.55     1_836.43       1.0000     0.000000       122.25
IVF-nl353 (self)                                         772.89    17_467.58    18_240.46       1.0000     0.000000       122.25
IVF-nl500-np22 (query)                                 1_551.70       648.33     2_200.03       1.0000     0.000569       122.32
IVF-nl500-np25 (query)                                 1_551.70       726.23     2_277.93       1.0000     0.000000       122.32
IVF-nl500-np31 (query)                                 1_551.70       890.03     2_441.73       1.0000     0.000000       122.32
IVF-nl500 (self)                                       1_551.70    14_424.42    15_976.12       1.0000     0.000000       122.32
IVF-nl707-np26 (query)                                 2_980.74       560.45     3_541.19       0.9999     0.001561       122.42
IVF-nl707-np35 (query)                                 2_980.74       742.69     3_723.42       1.0000     0.000000       122.42
IVF-nl707-np37 (query)                                 2_980.74       787.19     3_767.93       1.0000     0.000000       122.42
IVF-nl707 (self)                                       2_980.74    12_570.20    15_550.94       1.0000     0.000000       122.42
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        24.46    11_328.60    11_353.06       1.0000     0.000000       122.07
Exhaustive (self)                                         24.46   196_063.77   196_088.24       1.0000     0.000000       122.07
GPU-Exhaustive (query)                                    39.38     2_214.43     2_253.81       1.0000     0.000025       122.07
GPU-Exhaustive (self)                                     39.38    34_761.31    34_800.69       1.0000     0.000025       122.07
IVF-GPU-nl353-np17 (query)                               792.79       612.42     1_405.22       0.9999     0.001639         1.91
IVF-GPU-nl353-np18 (query)                               792.79       656.81     1_449.60       0.9999     0.000746         1.91
IVF-GPU-nl353-np26 (query)                               792.79       808.19     1_600.99       1.0000     0.000025         1.91
IVF-GPU-nl353 (self)                                     792.79     9_235.49    10_028.28       1.0000     0.000025         1.91
IVF-GPU-nl500-np22 (query)                             1_556.25       621.46     2_177.72       1.0000     0.000594         1.91
IVF-GPU-nl500-np25 (query)                             1_556.25       633.62     2_189.88       1.0000     0.000025         1.91
IVF-GPU-nl500-np31 (query)                             1_556.25       730.37     2_286.62       1.0000     0.000025         1.91
IVF-GPU-nl500 (self)                                   1_556.25     7_863.07     9_419.32       1.0000     0.000025         1.91
IVF-GPU-nl707-np26 (query)                             3_056.30       589.32     3_645.62       0.9999     0.001586         1.91
IVF-GPU-nl707-np35 (query)                             3_056.30       662.13     3_718.43       1.0000     0.000025         1.91
IVF-GPU-nl707-np37 (query)                             3_056.30       658.22     3_714.52       1.0000     0.000025         1.91
IVF-GPU-nl707 (self)                                   3_056.30     6_959.98    10_016.28       1.0000     0.000025         1.91
--------------------------------------------------------------------------------------------------------------------------------

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
================================================================================================================================
Benchmark: 500k samples, 64D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.79    11_839.81    11_860.59       1.0000     0.000000       122.07
Exhaustive (self)                                         20.79   377_113.03   377_133.82       1.0000     0.000000       122.07
IVF-nl500-np22 (query)                                 2_491.85       654.88     3_146.73       1.0000     0.000000       122.20
IVF-nl500-np25 (query)                                 2_491.85       743.10     3_234.95       1.0000     0.000000       122.20
IVF-nl500-np31 (query)                                 2_491.85       946.80     3_438.65       1.0000     0.000000       122.20
IVF-nl500 (self)                                       2_491.85    30_030.22    32_522.06       1.0000     0.000000       122.20
IVF-nl707-np26 (query)                                 4_686.76       573.24     5_260.00       1.0000     0.000014       122.25
IVF-nl707-np35 (query)                                 4_686.76       771.34     5_458.11       1.0000     0.000000       122.25
IVF-nl707-np37 (query)                                 4_686.76       823.20     5_509.96       1.0000     0.000000       122.25
IVF-nl707 (self)                                       4_686.76    26_194.15    30_880.91       1.0000     0.000000       122.25
IVF-nl1000-np31 (query)                                8_868.05       499.59     9_367.64       1.0000     0.000033       122.32
IVF-nl1000-np44 (query)                                8_868.05       699.71     9_567.76       1.0000     0.000000       122.32
IVF-nl1000-np50 (query)                                8_868.05       788.58     9_656.63       1.0000     0.000000       122.32
IVF-nl1000 (self)                                      8_868.05    21_920.02    30_788.08       1.0000     0.000000       122.32
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.93    12_867.41    12_888.33       1.0000     0.000000       122.07
Exhaustive (self)                                         20.93   402_508.34   402_529.27       1.0000     0.000000       122.07
GPU-Exhaustive (query)                                    34.74     2_700.87     2_735.62       1.0000     0.000003       122.07
GPU-Exhaustive (self)                                     34.74    85_665.73    85_700.47       1.0000     0.000003       122.07
IVF-GPU-nl500-np22 (query)                             2_760.09       635.81     3_395.90       1.0000     0.000003         3.82
IVF-GPU-nl500-np25 (query)                             2_760.09       674.72     3_434.82       1.0000     0.000003         3.82
IVF-GPU-nl500-np31 (query)                             2_760.09       755.19     3_515.29       1.0000     0.000003         3.82
IVF-GPU-nl500 (self)                                   2_760.09    16_629.32    19_389.42       1.0000     0.000003         3.82
IVF-GPU-nl707-np26 (query)                             5_564.83       455.97     6_020.81       1.0000     0.000017         3.82
IVF-GPU-nl707-np35 (query)                             5_564.83       655.10     6_219.93       1.0000     0.000003         3.82
IVF-GPU-nl707-np37 (query)                             5_564.83       675.62     6_240.45       1.0000     0.000003         3.82
IVF-GPU-nl707 (self)                                   5_564.83    14_695.11    20_259.94       1.0000     0.000003         3.82
IVF-GPU-nl1000-np31 (query)                           10_260.31       552.15    10_812.46       1.0000     0.000036         3.82
IVF-GPU-nl1000-np44 (query)                           10_260.31       636.66    10_896.97       1.0000     0.000003         3.82
IVF-GPU-nl1000-np50 (query)                           10_260.31       679.72    10_940.04       1.0000     0.000003         3.82
IVF-GPU-nl1000 (self)                                 10_260.31    12_362.91    22_623.22       1.0000     0.000003         3.82
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 128D
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        47.99    25_189.87    25_237.86       1.0000     0.000000       244.14
Exhaustive (self)                                         47.99   848_833.28   848_881.27       1.0000     0.000000       244.14
IVF-nl500-np22 (query)                                 1_670.88     1_386.20     3_057.08       0.9999     0.001213       244.39
IVF-nl500-np25 (query)                                 1_670.88     1_533.65     3_204.53       1.0000     0.000335       244.39
IVF-nl500-np31 (query)                                 1_670.88     1_844.07     3_514.95       1.0000     0.000000       244.39
IVF-nl500 (self)                                       1_670.88    60_800.89    62_471.77       1.0000     0.000000       244.39
IVF-nl707-np26 (query)                                 3_332.92     1_141.56     4_474.48       0.9997     0.003019       244.49
IVF-nl707-np35 (query)                                 3_332.92     1_519.73     4_852.65       1.0000     0.000010       244.49
IVF-nl707-np37 (query)                                 3_332.92     1_601.83     4_934.75       1.0000     0.000000       244.49
IVF-nl707 (self)                                       3_332.92    52_398.99    55_731.91       1.0000     0.000017       244.49
IVF-nl1000-np31 (query)                                7_178.44       985.71     8_164.15       0.9996     0.004000       244.64
IVF-nl1000-np44 (query)                                7_178.44     1_392.61     8_571.05       1.0000     0.000000       244.64
IVF-nl1000-np50 (query)                                7_178.44     1_571.59     8_750.02       1.0000     0.000000       244.64
IVF-nl1000 (self)                                      7_178.44    47_381.70    54_560.13       1.0000     0.000000       244.64
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        48.55    26_301.52    26_350.07       1.0000     0.000000       244.14
Exhaustive (self)                                         48.55   870_622.75   870_671.30       1.0000     0.000000       244.14
GPU-Exhaustive (query)                                    77.29     4_352.30     4_429.59       1.0000     0.000023       244.14
GPU-Exhaustive (self)                                     77.29   138_504.49   138_581.78       1.0000     0.000023       244.14
IVF-GPU-nl500-np22 (query)                             1_724.95       996.66     2_721.61       0.9999     0.001236         3.82
IVF-GPU-nl500-np25 (query)                             1_724.95     1_026.71     2_751.65       1.0000     0.000357         3.82
IVF-GPU-nl500-np31 (query)                             1_724.95     1_215.22     2_940.17       1.0000     0.000023         3.82
IVF-GPU-nl500 (self)                                   1_724.95    30_110.81    31_835.75       1.0000     0.000023         3.82
IVF-GPU-nl707-np26 (query)                             3_348.97       911.89     4_260.85       0.9997     0.003042         3.82
IVF-GPU-nl707-np35 (query)                             3_348.97     1_004.10     4_353.07       1.0000     0.000033         3.82
IVF-GPU-nl707-np37 (query)                             3_348.97     1_052.09     4_401.06       1.0000     0.000023         3.82
IVF-GPU-nl707 (self)                                   3_348.97    25_999.14    29_348.10       1.0000     0.000040         3.82
IVF-GPU-nl1000-np31 (query)                            7_123.60       844.49     7_968.09       0.9996     0.004023         3.82
IVF-GPU-nl1000-np44 (query)                            7_123.60       989.45     8_113.06       1.0000     0.000023         3.82
IVF-GPU-nl1000-np50 (query)                            7_123.60     1_010.69     8_134.29       1.0000     0.000023         3.82
IVF-GPU-nl1000 (self)                                  7_123.60    22_269.70    29_393.31       1.0000     0.000023         3.82
--------------------------------------------------------------------------------------------------------------------------------

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
================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                     3.02     1_541.14     1_544.16       1.0000     0.000000        18.31
CPU-Exhaustive (self)                                      3.02    16_601.27    16_604.29       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     4.98       645.60       650.58       1.0000     0.000005        18.31
GPU-Exhaustive (self)                                      4.98     5_442.09     5_447.07       1.0000     0.000005        18.31
CAGRA-auto (query)                                       576.50       129.12       705.62       0.9476     0.214308        86.98
CAGRA-auto (self)                                        576.50       775.90     1_352.40       0.9440     0.320535        86.98
CAGRA-bw16 (query)                                       576.50        89.69       666.19       0.9278     0.286963        86.98
CAGRA-bw16 (self)                                        576.50       363.07       939.57       0.9222     0.444823        86.98
CAGRA-bw30 (query)                                       576.50       124.09       700.58       0.9456     0.221683        86.98
CAGRA-bw30 (self)                                        576.50       710.19     1_286.69       0.9416     0.331667        86.98
CAGRA-bw48 (query)                                       576.50       199.33       775.82       0.9610     0.161387        86.98
CAGRA-bw48 (self)                                        576.50     1_371.99     1_948.49       0.9581     0.245559        86.98
CAGRA-bw64 (query)                                       576.50       262.14       838.64       0.9694     0.126339        86.98
CAGRA-bw64 (self)                                        576.50     2_124.83     2_701.32       0.9673     0.197956        86.98
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                     3.91     1_614.34     1_618.25       1.0000     0.000000        18.88
CPU-Exhaustive (self)                                      3.91    17_373.42    17_377.34       1.0000     0.000000        18.88
GPU-Exhaustive (query)                                     6.68       700.32       707.00       1.0000     0.000000        18.88
GPU-Exhaustive (self)                                      6.68     5_638.14     5_644.82       1.0000     0.000000        18.88
CAGRA-auto (query)                                       687.76       201.73       889.49       0.9492     0.000145        87.55
CAGRA-auto (self)                                        687.76       781.71     1_469.47       0.9455     0.000208        87.55
CAGRA-bw16 (query)                                       687.76       136.00       823.76       0.9293     0.000197        87.55
CAGRA-bw16 (self)                                        687.76       359.18     1_046.94       0.9240     0.000292        87.55
CAGRA-bw30 (query)                                       687.76       197.86       885.62       0.9472     0.000150        87.55
CAGRA-bw30 (self)                                        687.76       718.31     1_406.07       0.9432     0.000217        87.55
CAGRA-bw48 (query)                                       687.76       209.33       897.09       0.9625     0.000107        87.55
CAGRA-bw48 (self)                                        687.76     1_377.90     2_065.66       0.9594     0.000157        87.55
CAGRA-bw64 (query)                                       687.76       320.44     1_008.20       0.9709     0.000084        87.55
CAGRA-bw64 (self)                                        687.76     2_128.88     2_816.64       0.9685     0.000124        87.55
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                     3.52     1_607.14     1_610.66       1.0000     0.000000        18.31
CPU-Exhaustive (self)                                      3.52    17_028.63    17_032.15       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     4.99       642.20       647.18       1.0000     0.000000        18.31
GPU-Exhaustive (self)                                      4.99     5_443.32     5_448.31       1.0000     0.000000        18.31
CAGRA-auto (query)                                       561.42       122.70       684.12       0.9961     0.000450        86.98
CAGRA-auto (self)                                        561.42       716.39     1_277.82       0.9974     0.000584        86.98
CAGRA-bw16 (query)                                       561.42        86.26       647.69       0.9886     0.001205        86.98
CAGRA-bw16 (self)                                        561.42       337.83       899.25       0.9933     0.001201        86.98
CAGRA-bw30 (query)                                       561.42       119.36       680.78       0.9956     0.000497        86.98
CAGRA-bw30 (self)                                        561.42       658.97     1_220.40       0.9971     0.000636        86.98
CAGRA-bw48 (query)                                       561.42       220.25       781.68       0.9982     0.000216        86.98
CAGRA-bw48 (self)                                        561.42     1_218.82     1_780.24       0.9988     0.000327        86.98
CAGRA-bw64 (query)                                       561.42       236.47       797.90       0.9991     0.000119        86.98
CAGRA-bw64 (self)                                        561.42     1_836.24     2_397.66       0.9994     0.000230        86.98
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 32D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                     3.13     1_575.51     1_578.64       1.0000     0.000000        18.31
CPU-Exhaustive (self)                                      3.13    16_795.86    16_799.00       1.0000     0.000000        18.31
GPU-Exhaustive (query)                                     5.02       643.68       648.71       1.0000     0.000001        18.31
GPU-Exhaustive (self)                                      5.02     5_452.35     5_457.37       1.0000     0.000001        18.31
CAGRA-auto (query)                                       542.70       121.71       664.41       0.9989     0.000661        86.98
CAGRA-auto (self)                                        542.70       708.50     1_251.20       0.9988     0.001067        86.98
CAGRA-bw16 (query)                                       542.70        85.97       628.67       0.9960     0.001713        86.98
CAGRA-bw16 (self)                                        542.70       338.01       880.71       0.9962     0.002514        86.98
CAGRA-bw30 (query)                                       542.70       118.40       661.10       0.9987     0.000729        86.98
CAGRA-bw30 (self)                                        542.70       653.94     1_196.64       0.9987     0.001163        86.98
CAGRA-bw48 (query)                                       542.70       171.28       713.98       0.9995     0.000268        86.98
CAGRA-bw48 (self)                                        542.70     1_195.59     1_738.30       0.9995     0.000535        86.98
CAGRA-bw64 (query)                                       542.70       233.30       776.01       0.9998     0.000163        86.98
CAGRA-bw64 (self)                                        542.70     1_789.94     2_332.65       0.9998     0.000367        86.98
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k samples, 128D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    14.52     6_153.01     6_167.53       1.0000     0.000000        73.24
CPU-Exhaustive (self)                                     14.52    64_868.28    64_882.80       1.0000     0.000000        73.24
GPU-Exhaustive (query)                                    23.54     1_365.96     1_389.50       1.0000     0.000027        73.24
GPU-Exhaustive (self)                                     23.54    12_534.86    12_558.39       1.0000     0.000027        73.24
CAGRA-auto (query)                                     3_416.24       346.39     3_762.63       0.9762     0.292025       141.91
CAGRA-auto (self)                                      3_416.24     1_212.07     4_628.31       0.9732     0.448007       141.91
CAGRA-bw16 (query)                                     3_416.24       218.88     3_635.12       0.9625     0.432355       141.91
CAGRA-bw16 (self)                                      3_416.24       623.21     4_039.45       0.9572     0.693148       141.91
CAGRA-bw30 (query)                                     3_416.24       261.85     3_678.09       0.9748     0.308142       141.91
CAGRA-bw30 (self)                                      3_416.24     1_126.47     4_542.71       0.9716     0.470749       141.91
CAGRA-bw48 (query)                                     3_416.24       384.38     3_800.62       0.9846     0.189855       141.91
CAGRA-bw48 (self)                                      3_416.24     1_949.34     5_365.58       0.9823     0.306493       141.91
CAGRA-bw64 (query)                                     3_416.24       475.71     3_891.95       0.9894     0.133484       141.91
CAGRA-bw64 (self)                                      3_416.24     2_843.04     6_259.28       0.9875     0.230996       141.91
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Larger data sets

Let's test CAGRA similar to IVF GPU on larger data sets.

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 64D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    11.05     8_412.64     8_423.69       1.0000     0.000000        61.04
CPU-Exhaustive (self)                                     11.05    88_609.30    88_620.35       1.0000     0.000000        61.04
GPU-Exhaustive (query)                                    18.32     2_324.22     2_342.54       1.0000     0.000003        61.04
GPU-Exhaustive (self)                                     18.32    21_516.03    21_534.35       1.0000     0.000003        61.04
CAGRA-auto (query)                                     2_476.98       444.75     2_921.74       0.9952     0.008347       175.48
CAGRA-auto (self)                                      2_476.98     1_490.41     3_967.40       0.9943     0.013961       175.48
CAGRA-bw16 (query)                                     2_476.98       321.16     2_798.15       0.9891     0.016937       175.48
CAGRA-bw16 (self)                                      2_476.98       731.56     3_208.54       0.9871     0.027719       175.48
CAGRA-bw30 (query)                                     2_476.98       400.64     2_877.63       0.9948     0.009205       175.48
CAGRA-bw30 (self)                                      2_476.98     1_375.18     3_852.16       0.9937     0.015071       175.48
CAGRA-bw48 (query)                                     2_476.98       506.37     2_983.35       0.9975     0.004873       175.48
CAGRA-bw48 (self)                                      2_476.98     2_467.84     4_944.82       0.9970     0.008203       175.48
CAGRA-bw64 (query)                                     2_476.98       629.38     3_106.37       0.9985     0.003086       175.48
CAGRA-bw64 (self)                                      2_476.98     3_657.82     6_134.81       0.9983     0.005510       175.48
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 128D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    25.92    18_311.78    18_337.70       1.0000     0.000000       122.07
CPU-Exhaustive (self)                                     25.92   192_609.53   192_635.45       1.0000     0.000000       122.07
GPU-Exhaustive (query)                                    40.81     3_647.44     3_688.24       1.0000     0.000025       122.07
GPU-Exhaustive (self)                                     40.81    34_755.94    34_796.75       1.0000     0.000025       122.07
CAGRA-auto (query)                                     7_995.69       685.88     8_681.57       0.9716     0.295700       236.51
CAGRA-auto (self)                                      7_995.69     2_050.12    10_045.80       0.9653     0.506590       236.51
CAGRA-bw16 (query)                                     7_995.69       600.25     8_595.93       0.9565     0.438954       236.51
CAGRA-bw16 (self)                                      7_995.69     1_060.74     9_056.42       0.9472     0.769512       236.51
CAGRA-bw30 (query)                                     7_995.69       649.92     8_645.61       0.9701     0.311454       236.51
CAGRA-bw30 (self)                                      7_995.69     1_914.19     9_909.88       0.9635     0.531677       236.51
CAGRA-bw48 (query)                                     7_995.69       785.97     8_781.65       0.9805     0.205189       236.51
CAGRA-bw48 (self)                                      7_995.69     3_317.33    11_313.02       0.9761     0.357363       236.51
CAGRA-bw64 (query)                                     7_995.69       917.76     8_913.44       0.9858     0.147225       236.51
CAGRA-bw64 (self)                                      7_995.69     4_817.12    12_812.81       0.9825     0.270789       236.51
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 64D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    21.09    40_928.91    40_950.00       1.0000     0.000000       122.07
CPU-Exhaustive (self)                                     21.09   382_879.47   382_900.56       1.0000     0.000000       122.07
GPU-Exhaustive (query)                                    34.84     8_716.94     8_751.79       1.0000     0.000003       122.07
GPU-Exhaustive (self)                                     34.84    85_706.59    85_741.43       1.0000     0.000003       122.07
CAGRA-auto (query)                                     6_253.51       827.08     7_080.58       0.9907     0.016228       350.95
CAGRA-auto (self)                                      6_253.51     3_030.78     9_284.29       0.9889     0.023488       350.95
CAGRA-bw16 (query)                                     6_253.51       683.91     6_937.42       0.9816     0.028925       350.95
CAGRA-bw16 (self)                                      6_253.51     1_480.87     7_734.38       0.9788     0.041649       350.95
CAGRA-bw30 (query)                                     6_253.51       773.47     7_026.97       0.9899     0.017437       350.95
CAGRA-bw30 (self)                                      6_253.51     2_785.33     9_038.84       0.9879     0.025263       350.95
CAGRA-bw48 (query)                                     6_253.51       977.05     7_230.56       0.9946     0.009440       350.95
CAGRA-bw48 (self)                                      6_253.51     5_026.02    11_279.52       0.9936     0.014587       350.95
CAGRA-bw64 (query)                                     6_253.51     1_227.82     7_481.33       0.9966     0.006044       350.95
CAGRA-bw64 (self)                                      6_253.51     7_469.97    13_723.48       0.9959     0.009908       350.95
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 128 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 128D (Exhaustive vs CAGRA beam search)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
CPU-Exhaustive (query)                                    47.04    86_774.50    86_821.55       1.0000     0.000000       244.14
CPU-Exhaustive (self)                                     47.04   847_727.31   847_774.35       1.0000     0.000000       244.14
GPU-Exhaustive (query)                                    75.99    14_063.30    14_139.28       1.0000     0.000023       244.14
GPU-Exhaustive (self)                                     75.99   138_481.49   138_557.47       1.0000     0.000023       244.14
CAGRA-auto (query)                                    18_965.89     1_337.45    20_303.35       0.9488     0.554959       473.02
CAGRA-auto (self)                                     18_965.89     4_187.96    23_153.85       0.9398     0.852382       473.02
CAGRA-bw16 (query)                                    18_965.89     1_117.76    20_083.66       0.9297     0.757044       473.02
CAGRA-bw16 (self)                                     18_965.89     2_164.46    21_130.35       0.9176     1.192372       473.02
CAGRA-bw30 (query)                                    18_965.89     1_293.89    20_259.78       0.9468     0.577489       473.02
CAGRA-bw30 (self)                                     18_965.89     3_897.96    22_863.85       0.9374     0.887430       473.02
CAGRA-bw48 (query)                                    18_965.89     1_572.19    20_538.08       0.9617     0.414419       473.02
CAGRA-bw48 (self)                                     18_965.89     6_781.16    25_747.05       0.9547     0.639425       473.02
CAGRA-bw64 (query)                                    18_965.89     1_922.47    20_888.36       0.9700     0.325211       473.02
CAGRA-bw64 (self)                                     18_965.89     9_889.58    28_855.48       0.9645     0.507702       473.02
--------------------------------------------------------------------------------------------------------------------------------

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
================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             10.35    15_082.34    15_092.69       1.0000     0.000000        30.52
CPU-NNDescent (k=15)                                   4_520.44     1_087.76     5_608.20       1.0000     0.000003       286.87
GPU-NND bk=1x refine=0 (extract)                         845.98        42.18       888.16       0.9000     0.809792       144.96
GPU-NND bk=1x refine=0 (self-beam)                       845.98     1_216.14     2_062.11       0.9967     0.002730       144.96
GPU-NND bk=1x refine=1 (extract)                         852.49        42.16       894.65       0.9250     0.779051       144.96
GPU-NND bk=1x refine=1 (self-beam)                       852.49     1_213.26     2_065.75       0.9971     0.002429       144.96
GPU-NND bk=1x refine=2 (extract)                         962.99        41.71     1_004.71       0.9262     0.777891       144.96
GPU-NND bk=1x refine=2 (self-beam)                       962.99     1_209.98     2_172.97       0.9971     0.002411       144.96
GPU-NND bk=2x refine=0 (extract)                       1_238.89        41.66     1_280.56       0.9281     0.776172       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_238.89     1_209.87     2_448.76       0.9986     0.000811       144.96
GPU-NND bk=2x refine=1 (extract)                       1_531.99        41.18     1_573.18       0.9330     0.770913       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_531.99     1_213.78     2_745.77       0.9989     0.000556       144.96
GPU-NND bk=2x refine=2 (extract)                       1_813.43        41.17     1_854.60       0.9331     0.770847       144.96
GPU-NND bk=2x refine=2 (self-beam)                     1_813.43     1_213.40     3_026.83       0.9989     0.000537       144.96
GPU-NND bk=3x refine=0 (extract)                       2_200.30        42.47     2_242.77       0.9309     0.773083       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_200.30     1_211.15     3_411.45       0.9988     0.000521       144.96
GPU-NND bk=3x refine=1 (extract)                       2_829.99        41.55     2_871.54       0.9333     0.770632       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_829.99     1_209.68     4_039.66       0.9989     0.000373       144.96
GPU-NND bk=3x refine=2 (extract)                       3_515.01        41.10     3_556.10       0.9333     0.770627       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_515.01     1_211.10     4_726.10       0.9989     0.000391       144.96
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 250k samples, 64D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             21.84    21_611.65    21_633.49       1.0000     0.000000        61.04
CPU-NNDescent (k=15)                                   6_210.12     1_682.14     7_892.26       1.0000     0.000035       361.41
GPU-NND bk=1x refine=0 (extract)                       1_236.62        42.09     1_278.71       0.8697     3.054552       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_236.62     1_499.87     2_736.49       0.9892     0.031673       175.48
GPU-NND bk=1x refine=1 (extract)                       1_672.34        41.63     1_713.96       0.9126     2.856986       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_672.34     1_493.47     3_165.81       0.9905     0.027936       175.48
GPU-NND bk=1x refine=2 (extract)                       2_127.86        41.89     2_169.75       0.9155     2.845694       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_127.86     1_505.74     3_633.60       0.9907     0.027249       175.48
GPU-NND bk=2x refine=0 (extract)                       2_009.78        41.31     2_051.09       0.9198     2.832780       175.48
GPU-NND bk=2x refine=0 (self-beam)                     2_009.78     1_506.26     3_516.04       0.9945     0.011845       175.48
GPU-NND bk=2x refine=1 (extract)                       3_648.44        41.63     3_690.07       0.9319     2.784220       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_648.44     1_499.95     5_148.39       0.9957     0.008196       175.48
GPU-NND bk=2x refine=2 (extract)                       5_278.77        41.82     5_320.59       0.9322     2.783190       175.48
GPU-NND bk=2x refine=2 (self-beam)                     5_278.77     1_494.41     6_773.18       0.9958     0.007993       175.48
GPU-NND bk=3x refine=0 (extract)                       4_575.17        42.00     4_617.17       0.9282     2.797534       175.48
GPU-NND bk=3x refine=0 (self-beam)                     4_575.17     1_504.73     6_079.90       0.9954     0.007983       175.48
GPU-NND bk=3x refine=1 (extract)                       7_513.75        41.78     7_555.53       0.9332     2.779672       175.48
GPU-NND bk=3x refine=1 (self-beam)                     7_513.75     1_492.20     9_005.95       0.9960     0.006219       175.48
GPU-NND bk=3x refine=2 (extract)                      10_355.67        41.09    10_396.76       0.9332     2.779566       175.48
GPU-NND bk=3x refine=2 (self-beam)                    10_355.67     1_499.49    11_855.15       0.9960     0.006068       175.48
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 32 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             20.77    59_798.69    59_819.46       1.0000     0.000000        61.04
CPU-NNDescent (k=15)                                  10_549.68     2_660.28    13_209.96       1.0000     0.000004       563.78
GPU-NND bk=1x refine=0 (extract)                       1_625.40        84.31     1_709.71       0.8849     0.701455       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_625.40     2_450.48     4_075.88       0.9939     0.004049       289.92
GPU-NND bk=1x refine=1 (extract)                       1_905.19        84.52     1_989.71       0.9194     0.664143       289.92
GPU-NND bk=1x refine=1 (self-beam)                     1_905.19     2_440.96     4_346.15       0.9947     0.003535       289.92
GPU-NND bk=1x refine=2 (extract)                       2_188.42        84.62     2_273.04       0.9215     0.662186       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_188.42     2_442.43     4_630.85       0.9948     0.003439       289.92
GPU-NND bk=2x refine=0 (extract)                       2_470.53        84.33     2_554.86       0.9252     0.659264       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_470.53     2_446.08     4_916.61       0.9974     0.001177       289.92
GPU-NND bk=2x refine=1 (extract)                       3_519.40        83.80     3_603.20       0.9327     0.652324       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_519.40     2_441.45     5_960.85       0.9980     0.000755       289.92
GPU-NND bk=2x refine=2 (extract)                       4_613.40        84.46     4_697.86       0.9328     0.652243       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_613.40     2_443.53     7_056.93       0.9980     0.000745       289.92
GPU-NND bk=3x refine=0 (extract)                       4_548.60        83.61     4_632.21       0.9300     0.654640       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_548.60     2_452.10     7_000.71       0.9979     0.000708       289.92
GPU-NND bk=3x refine=1 (extract)                       6_526.69        84.18     6_610.88       0.9333     0.651838       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_526.69     2_442.80     8_969.50       0.9982     0.000507       289.92
GPU-NND bk=3x refine=2 (extract)                       8_513.30        84.84     8_598.14       0.9333     0.651828       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_513.30     2_451.42    10_964.72       0.9982     0.000525       289.92
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 500k samples, 64D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             37.32    85_838.46    85_875.78       1.0000     0.000000       122.07
CPU-NNDescent (k=15)                                  14_551.98     4_107.45    18_659.43       1.0000     0.000074       793.87
GPU-NND bk=1x refine=0 (extract)                       2_523.79        85.84     2_609.63       0.8341     2.774602       350.95
GPU-NND bk=1x refine=0 (self-beam)                     2_523.79     3_044.04     5_567.83       0.9784     0.056348       350.95
GPU-NND bk=1x refine=1 (extract)                       3_957.71        85.68     4_043.39       0.8948     2.514910       350.95
GPU-NND bk=1x refine=1 (self-beam)                     3_957.71     3_020.14     6_977.85       0.9810     0.048801       350.95
GPU-NND bk=1x refine=2 (extract)                       5_548.13        84.30     5_632.43       0.9009     2.493091       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_548.13     3_026.74     8_574.88       0.9815     0.047011       350.95
GPU-NND bk=2x refine=0 (extract)                       4_223.69        83.61     4_307.30       0.9098     2.467344       350.95
GPU-NND bk=2x refine=0 (self-beam)                     4_223.69     3_031.12     7_254.81       0.9895     0.020433       350.95
GPU-NND bk=2x refine=1 (extract)                       9_365.58        83.48     9_449.05       0.9300     2.395654       350.95
GPU-NND bk=2x refine=1 (self-beam)                     9_365.58     3_020.89    12_386.47       0.9920     0.013367       350.95
GPU-NND bk=2x refine=2 (extract)                      11_984.24        83.83    12_068.06       0.9307     2.393293       350.95
GPU-NND bk=2x refine=2 (self-beam)                    11_984.24     3_176.00    15_160.24       0.9922     0.012577       350.95
GPU-NND bk=3x refine=0 (extract)                       9_929.66        83.76    10_013.42       0.9252     2.410860       350.95
GPU-NND bk=3x refine=0 (self-beam)                     9_929.66     3_021.61    12_951.28       0.9919     0.012577       350.95
GPU-NND bk=3x refine=1 (extract)                      17_922.51        84.01    18_006.51       0.9329     2.386270       350.95
GPU-NND bk=3x refine=1 (self-beam)                    17_922.51     3_102.17    21_024.68       0.9932     0.009081       350.95
GPU-NND bk=3x refine=2 (extract)                      28_067.86        84.31    28_152.18       0.9330     2.385960       350.95
GPU-NND bk=3x refine=2 (self-beam)                    28_067.86     2_989.20    31_057.06       0.9932     0.008912       350.95
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 32 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             39.02   236_792.16   236_831.19       1.0000     0.000000       122.07
CPU-NNDescent (k=15)                                  22_541.26     5_995.57    28_536.83       1.0000     0.000013      1143.56
GPU-NND bk=1x refine=0 (extract)                       3_376.07       176.38     3_552.44       0.8552     0.687773       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_376.07     4_955.52     8_331.58       0.9871     0.008307       579.83
GPU-NND bk=1x refine=1 (extract)                       4_223.31       168.20     4_391.51       0.9063     0.631794       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_223.31     4_990.07     9_213.38       0.9890     0.006989       579.83
GPU-NND bk=1x refine=2 (extract)                       5_252.08       167.14     5_419.22       0.9110     0.627568       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_252.08     4_957.98    10_210.06       0.9893     0.006735       579.83
GPU-NND bk=2x refine=0 (extract)                       5_195.86       165.94     5_361.80       0.9189     0.621272       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_195.86     4_970.25    10_166.11       0.9949     0.002288       579.83
GPU-NND bk=2x refine=1 (extract)                       8_439.84       167.84     8_607.68       0.9318     0.609827       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_439.84     4_968.72    13_408.56       0.9961     0.001438       579.83
GPU-NND bk=2x refine=2 (extract)                      11_545.15       167.85    11_713.00       0.9321     0.609542       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_545.15     4_965.68    16_510.83       0.9962     0.001366       579.83
GPU-NND bk=3x refine=0 (extract)                       9_511.94       166.31     9_678.25       0.9282     0.612667       579.83
GPU-NND bk=3x refine=0 (self-beam)                     9_511.94     4_957.62    14_469.55       0.9960     0.001312       579.83
GPU-NND bk=3x refine=1 (extract)                      15_541.16       165.77    15_706.93       0.9332     0.608696       579.83
GPU-NND bk=3x refine=1 (self-beam)                    15_541.16     4_974.23    20_515.39       0.9966     0.000940       579.83
GPU-NND bk=3x refine=2 (extract)                      21_416.91       168.93    21_585.84       0.9332     0.608668       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_416.91     4_981.93    26_398.84       0.9966     0.000921       579.83
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 64 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 1000k samples, 64D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             74.62   341_441.20   341_515.82       1.0000     0.000000       244.14
CPU-NNDescent (k=15)                                  31_030.72     9_661.52    40_692.24       0.9999     0.000153      1559.74
GPU-NND bk=1x refine=0 (extract)                       5_015.36       171.32     5_186.68       0.8047     2.462733       701.90
GPU-NND bk=1x refine=0 (self-beam)                     5_015.36     6_111.38    11_126.74       0.9688     0.071291       701.90
GPU-NND bk=1x refine=1 (extract)                       8_886.30       168.21     9_054.51       0.8782     2.176138       701.90
GPU-NND bk=1x refine=1 (self-beam)                     8_886.30     6_093.07    14_979.37       0.9730     0.060390       701.90
GPU-NND bk=1x refine=2 (extract)                      12_921.06       167.85    13_088.92       0.8879     2.145254       701.90
GPU-NND bk=1x refine=2 (self-beam)                    12_921.06     6_092.75    19_013.81       0.9739     0.057780       701.90
GPU-NND bk=2x refine=0 (extract)                       8_871.27       165.67     9_036.95       0.9020     2.107166       701.90
GPU-NND bk=2x refine=0 (self-beam)                     8_871.27     6_141.26    15_012.53       0.9859     0.024003       701.90
GPU-NND bk=2x refine=1 (extract)                      21_296.38       167.40    21_463.78       0.9281     2.026944       701.90
GPU-NND bk=2x refine=1 (self-beam)                    21_296.38     6_107.18    27_403.55       0.9894     0.015350       701.90
GPU-NND bk=2x refine=2 (extract)                      33_533.30       166.31    33_699.62       0.9293     2.023623       701.90
GPU-NND bk=2x refine=2 (self-beam)                    33_533.30     6_119.86    39_653.17       0.9898     0.014308       701.90
GPU-NND bk=3x refine=0 (extract)                      20_822.92       166.35    20_989.26       0.9229     2.040863       701.90
GPU-NND bk=3x refine=0 (self-beam)                    20_822.92     6_130.58    26_953.50       0.9897     0.013798       701.90
GPU-NND bk=3x refine=1 (extract)                      42_667.20       166.14    42_833.34       0.9326     2.014631       701.90
GPU-NND bk=3x refine=1 (self-beam)                    42_667.20     6_163.98    48_831.18       0.9914     0.009694       701.90
GPU-NND bk=3x refine=2 (extract)                      64_536.52       166.82    64_703.35       0.9328     2.014218       701.90
GPU-NND bk=3x refine=2 (self-beam)                    64_536.52     6_134.98    70_671.50       0.9915     0.009591       701.90
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Let's do one large data set with 2.5m samples at 32 dimensions and see what
happens ... ?

<details>
<summary><b>Generation of a kNN graph with CAGRA (2.5m samples; 32 dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 2500k samples, 32D kNN graph generation (build_k x refinement)
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             93.84 1_480_815.79 1_480_909.62       1.0000     0.000000       305.18
CPU-NNDescent (k=15)                                  65_324.96    18_507.18    83_832.14       0.9999     0.000020      2990.81
GPU-NND bk=1x refine=0 (extract)                       7_519.96       428.49     7_948.45       0.8176     0.556663      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_519.96    12_629.32    20_149.27       0.9782     0.011272      1449.59
GPU-NND bk=1x refine=1 (extract)                      10_586.00       443.07    11_029.07       0.8880     0.491570      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    10_586.00    12_525.29    23_111.28       0.9819     0.009214      1449.59
GPU-NND bk=1x refine=2 (extract)                      13_771.84       441.01    14_212.85       0.8973     0.484653      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    13_771.84    12_527.63    26_299.47       0.9827     0.008698      1449.59
GPU-NND bk=2x refine=0 (extract)                      12_445.35       435.00    12_880.35       0.9125     0.474745      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    12_445.35    12_518.65    24_964.00       0.9928     0.002599      1449.59
GPU-NND bk=2x refine=1 (extract)                      23_182.61       435.58    23_618.19       0.9306     0.462261      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    23_182.61    12_617.33    35_799.93       0.9949     0.001505      1449.59
GPU-NND bk=2x refine=2 (extract)                      34_468.41       441.06    34_909.47       0.9313     0.461828      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    34_468.41    12_403.26    46_871.67       0.9951     0.001379      1449.59
GPU-NND bk=3x refine=0 (extract)                      24_149.67       425.77    24_575.44       0.9264     0.464844      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    24_149.67    12_401.14    36_550.81       0.9949     0.001321      1449.59
GPU-NND bk=3x refine=1 (extract)                      43_929.30       429.09    44_358.39       0.9330     0.460786      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    43_929.30    12_397.71    56_327.02       0.9959     0.000844      1449.59
GPU-NND bk=3x refine=2 (extract)                      63_781.51       428.81    64_210.32       0.9331     0.460742      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    63_781.51    12_405.32    76_186.83       0.9959     0.000830      1449.59
--------------------------------------------------------------------------------------------------------------------------------

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
*Last update: 2026/04/04 with version **0.2.11***
