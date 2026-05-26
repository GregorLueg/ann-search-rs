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
Exhaustive (query)                                         3.07     1_502.14     1_505.21       1.0000          1.0000        18.31
Exhaustive (self)                                          3.07    15_244.83    15_247.90       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     4.83       668.22       673.05       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      4.83     5_467.17     5_472.00       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               407.67       314.04       721.71       0.9972          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               407.67       360.47       768.14       0.9996          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               407.67       430.99       838.66       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     407.67     1_552.16     1_959.83       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               754.95       358.10     1_113.05       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               754.95       404.35     1_159.30       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     754.95     1_373.81     2_128.76       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_478.22       390.81     1_869.03       0.9931          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_478.22       349.03     1_827.25       0.9984          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_478.22       367.61     1_845.82       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_478.22     1_359.77     2_837.99       1.0000          1.0000         1.15
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
Exhaustive (query)                                         4.37     1_594.44     1_598.81       1.0000          1.0000        18.88
Exhaustive (self)                                          4.37    17_031.69    17_036.06       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.33       688.23       694.56       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      6.33     5_702.63     5_708.96       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               459.69       431.58       891.27       0.9977          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               459.69       268.85       728.54       0.9998          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               459.69       409.83       869.52       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     459.69     1_683.27     2_142.96       0.9999          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               782.70       375.96     1_158.66       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               782.70       416.69     1_199.38       0.9999          1.0000         1.15
IVF-GPU-nl387 (self)                                     782.70     1_524.15     2_306.85       0.9999          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_566.36       386.74     1_953.11       0.9941          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_566.36       366.25     1_932.62       0.9987          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_566.36       373.78     1_940.15       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_566.36     1_613.97     3_180.33       0.9999          1.0000         1.15
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
Exhaustive (query)                                         3.12     1_628.02     1_631.14       1.0000          1.0000        18.31
Exhaustive (self)                                          3.12    17_931.47    17_934.59       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.22       659.33       664.55       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.22     5_471.65     5_476.86       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               405.37       299.12       704.48       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               405.37       372.64       778.01       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               405.37       433.53       838.90       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     405.37     1_559.58     1_964.94       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               780.23       361.22     1_141.45       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               780.23       407.94     1_188.18       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     780.23     1_381.52     2_161.75       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_547.13       369.32     1_916.44       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_547.13       349.74     1_896.86       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_547.13       360.19     1_907.32       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_547.13     1_302.77     2_849.90       1.0000          1.0000         1.15
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
Exhaustive (query)                                         3.33     1_618.24     1_621.57       1.0000          1.0000        18.31
Exhaustive (self)                                          3.33    16_948.19    16_951.52       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.59       662.73       668.33       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.59     5_479.80     5_485.40       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               409.95       301.60       711.55       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               409.95       348.20       758.15       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               409.95       410.89       820.84       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     409.95     1_421.18     1_831.14       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               790.60       200.26       990.86       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               790.60       375.04     1_165.64       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     790.60     1_304.02     2_094.62       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_572.46       234.15     1_806.61       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_572.46       304.70     1_877.16       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_572.46       351.12     1_923.58       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_572.46     1_277.25     2_849.71       1.0000          1.0000         1.15
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
Exhaustive (query)                                        14.50     6_425.08     6_439.58       1.0000          1.0000        73.24
Exhaustive (self)                                         14.50    66_572.36    66_586.87       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    23.40     1_403.91     1_427.31       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     23.40    12_668.33    12_691.72       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               695.42       428.96     1_124.39       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               695.42       492.24     1_187.66       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               695.42       604.90     1_300.32       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     695.42     3_415.24     4_110.66       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_334.48       436.83     1_771.31       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_334.48       528.75     1_863.23       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_334.48     3_116.97     4_451.45       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_683.70       395.19     3_078.88       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_683.70       461.79     3_145.48       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_683.70       535.38     3_219.07       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_683.70     2_900.59     5_584.29       1.0000          1.0000         1.15
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
Exhaustive (query)                                        11.10     5_002.32     5_013.42       1.0000          1.0000        61.04
Exhaustive (self)                                         11.10    87_197.43    87_208.52       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                 1_233.13       321.46     1_554.58       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                 1_233.13       328.67     1_561.79       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                 1_233.13       445.53     1_678.66       1.0000          1.0000        61.12
IVF-nl353 (self)                                       1_233.13     6_759.63     7_992.75       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 2_250.75       311.46     2_562.21       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 2_250.75       347.12     2_597.87       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 2_250.75       401.21     2_651.96       1.0000          1.0000        61.16
IVF-nl500 (self)                                       2_250.75     5_595.84     7_846.59       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 4_484.10       288.13     4_772.22       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 4_484.10       352.97     4_837.06       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 4_484.10       367.57     4_851.66       1.0000          1.0000        61.21
IVF-nl707 (self)                                       4_484.10     4_832.19     9_316.29       1.0000          1.0000        61.21
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
Exhaustive (query)                                        11.20     5_289.76     5_300.96       1.0000          1.0000        61.04
Exhaustive (self)                                         11.20    88_058.54    88_069.75       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    17.63     1_440.16     1_457.78       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     17.63    21_599.26    21_616.88       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                             1_195.45       497.45     1_692.90       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_195.45       516.29     1_711.73       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_195.45       590.52     1_785.97       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_195.45     4_819.38     6_014.83       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_194.05       548.55     2_742.60       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_194.05       540.48     2_734.53       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_194.05       512.85     2_706.89       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_194.05     4_409.19     6_603.23       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_334.50       504.04     4_838.54       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_334.50       532.39     4_866.89       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_334.50       547.41     4_881.91       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_334.50     4_019.87     8_354.36       1.0000          1.0000         1.91
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
Exhaustive (query)                                        24.44    11_051.43    11_075.87       1.0000          1.0000       122.07
Exhaustive (self)                                         24.44   188_133.87   188_158.31       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                 1_118.50       652.66     1_771.16       0.9999          1.0000       122.25
IVF-nl353-np18 (query)                                 1_118.50       688.18     1_806.67       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                 1_118.50       936.32     2_054.81       1.0000          1.0000       122.25
IVF-nl353 (self)                                       1_118.50    14_985.15    16_103.64       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 2_206.78       637.80     2_844.58       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 2_206.78       692.82     2_899.61       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 2_206.78       840.07     3_046.85       1.0000          1.0000       122.32
IVF-nl500 (self)                                       2_206.78    13_369.59    15_576.38       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 4_477.34       590.37     5_067.71       1.0000          1.0000       122.42
IVF-nl707-np35 (query)                                 4_477.34       724.37     5_201.72       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 4_477.34       750.76     5_228.10       1.0000          1.0000       122.42
IVF-nl707 (self)                                       4_477.34    11_995.47    16_472.82       1.0000          1.0000       122.42
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
Exhaustive (query)                                        24.33    11_074.67    11_099.00       1.0000          1.0000       122.07
Exhaustive (self)                                         24.33   189_262.17   189_286.50       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.84     2_257.31     2_294.15       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.84    34_941.18    34_978.02       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                             1_180.10       621.42     1_801.52       0.9999          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_180.10       430.23     1_610.32       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_180.10       711.27     1_891.36       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_180.10     8_214.06     9_394.15       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_307.81       723.38     3_031.18       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_307.81       637.05     2_944.86       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_307.81       696.18     3_003.99       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_307.81     7_510.62     9_818.43       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_485.30       697.86     5_183.17       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_485.30       651.00     5_136.30       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_485.30       679.68     5_164.98       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_485.30     6_736.58    11_221.88       1.0000          1.0000         1.91
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
Exhaustive (query)                                        21.42    11_362.80    11_384.21       1.0000          1.0000       122.07
Exhaustive (self)                                         21.42   388_185.24   388_206.66       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 2_595.33       670.96     3_266.28       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 2_595.33       684.62     3_279.94       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 2_595.33       858.70     3_454.02       1.0000          1.0000       122.20
IVF-nl500 (self)                                       2_595.33    26_305.70    28_901.02       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 4_701.05       580.69     5_281.74       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 4_701.05       715.08     5_416.13       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 4_701.05       737.12     5_438.17       1.0000          1.0000       122.25
IVF-nl707 (self)                                       4_701.05    23_323.47    28_024.52       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                8_426.83       522.31     8_949.14       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                8_426.83       699.71     9_126.54       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                8_426.83       762.75     9_189.58       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      8_426.83    21_389.32    29_816.15       1.0000          1.0000       122.32
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
Exhaustive (query)                                        20.88    12_355.31    12_376.19       1.0000          1.0000       122.07
Exhaustive (self)                                         20.88   388_312.57   388_333.45       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    34.17     2_740.86     2_775.02       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     34.17    85_869.48    85_903.64       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             2_363.39       619.32     2_982.71       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_363.39       681.83     3_045.22       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_363.39       725.12     3_088.51       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_363.39    15_413.02    17_776.41       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_479.78       734.00     5_213.78       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_479.78       705.24     5_185.02       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_479.78       672.78     5_152.57       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_479.78    13_801.61    18_281.40       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            8_175.12       725.70     8_900.82       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            8_175.12       642.96     8_818.08       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            8_175.12       723.71     8_898.82       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  8_175.12    12_471.89    20_647.00       1.0000          1.0000         3.82
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
Exhaustive (query)                                        48.58    25_512.89    25_561.47       1.0000          1.0000       244.14
Exhaustive (self)                                         48.58   853_040.77   853_089.35       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_454.24     1_285.91     3_740.15       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 2_454.24     1_403.67     3_857.91       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 2_454.24     1_673.25     4_127.49       1.0000          1.0000       244.39
IVF-nl500 (self)                                       2_454.24    54_319.13    56_773.37       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 4_695.55     1_220.64     5_916.19       1.0000          1.0000       244.49
IVF-nl707-np35 (query)                                 4_695.55     1_438.97     6_134.52       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 4_695.55     1_482.31     6_177.86       1.0000          1.0000       244.49
IVF-nl707 (self)                                       4_695.55    48_434.43    53_129.98       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                               10_492.99     1_047.98    11_540.98       0.9999          1.0000       244.64
IVF-nl1000-np44 (query)                               10_492.99     1_358.35    11_851.35       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                               10_492.99     1_518.83    12_011.83       1.0000          1.0000       244.64
IVF-nl1000 (self)                                     10_492.99    44_241.10    54_734.10       1.0000          1.0000       244.64
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
Exhaustive (query)                                        46.18    25_309.32    25_355.50       1.0000          1.0000       244.14
Exhaustive (self)                                         46.18   885_310.59   885_356.77       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    70.53     4_416.91     4_487.44       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     70.53   138_932.47   139_003.00       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             2_515.63       966.61     3_482.24       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_515.63       974.00     3_489.63       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_515.63     1_124.94     3_640.57       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_515.63    27_438.39    29_954.02       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_841.85       996.34     5_838.19       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_841.85     1_030.16     5_872.01       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_841.85       997.08     5_838.94       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_841.85    24_471.43    29_313.28       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                           10_568.08       948.94    11_517.02       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                           10_568.08     1_008.74    11_576.82       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                           10_568.08     1_074.71    11_642.79       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                 10_568.08    22_146.25    32_714.33       1.0000          1.0000         3.82
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
CPU-Exhaustive (query)                                     3.15     1_470.56     1_473.71       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.15    15_336.87    15_340.02       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.49       657.09       662.58       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.49     5_480.52     5_486.02       1.0000          1.0000        18.31
CAGRA-auto (query)                                       855.82       121.41       977.23       0.9333          1.0045        86.98
CAGRA-auto (self)                                        855.82       652.59     1_508.41       0.9317          1.0047        86.98
CAGRA-bw16 (query)                                       855.82       152.62     1_008.44       0.9172          1.0054        86.98
CAGRA-bw16 (self)                                        855.82       308.72     1_164.54       0.9169          1.0056        86.98
CAGRA-bw30 (query)                                       855.82       126.15       981.97       0.9319          1.0045        86.98
CAGRA-bw30 (self)                                        855.82       599.63     1_455.45       0.9306          1.0047        86.98
CAGRA-bw48 (query)                                       855.82       201.42     1_057.24       0.9416          1.0039        86.98
CAGRA-bw48 (self)                                        855.82     1_108.51     1_964.33       0.9393          1.0042        86.98
CAGRA-bw64 (query)                                       855.82       266.21     1_122.03       0.9474          1.0035        86.98
CAGRA-bw64 (self)                                        855.82     1_683.51     2_539.33       0.9448          1.0038        86.98
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
CPU-Exhaustive (query)                                     3.92     1_568.77     1_572.69       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      3.92    16_431.98    16_435.89       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     5.87       674.40       680.27       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      5.87     5_698.46     5_704.33       1.0000          1.0000        18.88
CAGRA-auto (query)                                       899.00       175.13     1_074.13       0.9320          1.0047        87.55
CAGRA-auto (self)                                        899.00       660.59     1_559.59       0.9305          1.0049        87.55
CAGRA-bw16 (query)                                       899.00       124.83     1_023.83       0.9140          1.0058        87.55
CAGRA-bw16 (self)                                        899.00       317.64     1_216.64       0.9146          1.0060        87.55
CAGRA-bw30 (query)                                       899.00       201.51     1_100.51       0.9306          1.0047        87.55
CAGRA-bw30 (self)                                        899.00       606.79     1_505.78       0.9293          1.0050        87.55
CAGRA-bw48 (query)                                       899.00       247.03     1_146.03       0.9408          1.0040        87.55
CAGRA-bw48 (self)                                        899.00     1_129.69     2_028.69       0.9390          1.0043        87.55
CAGRA-bw64 (query)                                       899.00       313.83     1_212.83       0.9473          1.0036        87.55
CAGRA-bw64 (self)                                        899.00     1_715.87     2_614.87       0.9452          1.0038        87.55
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
CPU-Exhaustive (query)                                     3.45     1_539.84     1_543.29       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.45    16_419.04    16_422.49       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.45       655.19       660.64       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.45     5_469.01     5_474.46       1.0000          1.0000        18.31
CAGRA-auto (query)                                       824.62       117.65       942.27       0.9806          1.0010        86.98
CAGRA-auto (self)                                        824.62       634.46     1_459.08       0.9912          1.0005        86.98
CAGRA-bw16 (query)                                       824.62        90.09       914.71       0.9557          1.0025        86.98
CAGRA-bw16 (self)                                        824.62       318.16     1_142.78       0.9836          1.0009        86.98
CAGRA-bw30 (query)                                       824.62       193.38     1_018.00       0.9790          1.0011        86.98
CAGRA-bw30 (self)                                        824.62       584.45     1_409.08       0.9906          1.0005        86.98
CAGRA-bw48 (query)                                       824.62       179.03     1_003.66       0.9894          1.0006        86.98
CAGRA-bw48 (self)                                        824.62     1_083.85     1_908.47       0.9948          1.0003        86.98
CAGRA-bw64 (query)                                       824.62       236.81     1_061.44       0.9934          1.0003        86.98
CAGRA-bw64 (self)                                        824.62     1_644.93     2_469.55       0.9966          1.0002        86.98
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
CPU-Exhaustive (query)                                     3.35     1_587.83     1_591.18       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.35    16_736.98    16_740.33       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.34       653.11       658.46       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.34     5_484.74     5_490.08       1.0000          1.0000        18.31
CAGRA-auto (query)                                       820.96       118.00       938.95       0.9868          1.0008        86.98
CAGRA-auto (self)                                        820.96       635.41     1_456.37       0.9929          1.0005        86.98
CAGRA-bw16 (query)                                       820.96       149.19       970.15       0.9690          1.0019        86.98
CAGRA-bw16 (self)                                        820.96       305.81     1_126.77       0.9867          1.0009        86.98
CAGRA-bw30 (query)                                       820.96       118.55       939.51       0.9855          1.0009        86.98
CAGRA-bw30 (self)                                        820.96       586.02     1_406.97       0.9924          1.0005        86.98
CAGRA-bw48 (query)                                       820.96       178.97       999.92       0.9926          1.0005        86.98
CAGRA-bw48 (self)                                        820.96     1_087.83     1_908.79       0.9960          1.0003        86.98
CAGRA-bw64 (query)                                       820.96       236.95     1_057.91       0.9955          1.0003        86.98
CAGRA-bw64 (self)                                        820.96     1_655.64     2_476.60       0.9974          1.0002        86.98
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
CPU-Exhaustive (query)                                    14.33     6_210.62     6_224.95       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     14.33    65_196.18    65_210.51       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    23.58     1_394.38     1_417.96       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     23.58    12_662.53    12_686.10       1.0000          1.0000        73.24
CAGRA-auto (query)                                     2_730.34       291.13     3_021.47       0.9860          1.0007       141.91
CAGRA-auto (self)                                      2_730.34       787.93     3_518.27       0.9930          1.0005       141.91
CAGRA-bw16 (query)                                     2_730.34       204.25     2_934.59       0.9678          1.0016       141.91
CAGRA-bw16 (self)                                      2_730.34       422.05     3_152.39       0.9867          1.0008       141.91
CAGRA-bw30 (query)                                     2_730.34       235.96     2_966.30       0.9848          1.0007       141.91
CAGRA-bw30 (self)                                      2_730.34       724.15     3_454.49       0.9925          1.0005       141.91
CAGRA-bw48 (query)                                     2_730.34       289.00     3_019.34       0.9924          1.0004       141.91
CAGRA-bw48 (self)                                      2_730.34     1_281.38     4_011.72       0.9960          1.0003       141.91
CAGRA-bw64 (query)                                     2_730.34       350.54     3_080.88       0.9952          1.0002       141.91
CAGRA-bw64 (self)                                      2_730.34     1_896.14     4_626.47       0.9975          1.0002       141.91
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
CPU-Exhaustive (query)                                    11.07     8_427.86     8_438.94       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     11.07    87_559.58    87_570.65       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.60     2_355.91     2_374.51       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.60    21_632.47    21_651.06       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_616.67       356.70     2_973.37       0.9822          1.0010       175.48
CAGRA-auto (self)                                      2_616.67     1_164.35     3_781.02       0.9907          1.0006       175.48
CAGRA-bw16 (query)                                     2_616.67       306.59     2_923.27       0.9609          1.0022       175.48
CAGRA-bw16 (self)                                      2_616.67       588.45     3_205.13       0.9834          1.0011       175.48
CAGRA-bw30 (query)                                     2_616.67       374.85     2_991.52       0.9808          1.0010       175.48
CAGRA-bw30 (self)                                      2_616.67     1_076.47     3_693.14       0.9900          1.0007       175.48
CAGRA-bw48 (query)                                     2_616.67       464.58     3_081.26       0.9899          1.0006       175.48
CAGRA-bw48 (self)                                      2_616.67     1_965.70     4_582.37       0.9945          1.0004       175.48
CAGRA-bw64 (query)                                     2_616.67       537.22     3_153.90       0.9937          1.0003       175.48
CAGRA-bw64 (self)                                      2_616.67     2_968.94     5_585.62       0.9964          1.0002       175.48
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
CPU-Exhaustive (query)                                    24.82    18_433.68    18_458.50       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     24.82   189_276.69   189_301.50       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.44     3_691.52     3_727.97       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.44    34_946.74    34_983.18       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_124.53       575.82     6_700.36       0.9811          1.0009       236.51
CAGRA-auto (self)                                      6_124.53     1_348.27     7_472.81       0.9906          1.0006       236.51
CAGRA-bw16 (query)                                     6_124.53       516.12     6_640.66       0.9595          1.0020       236.51
CAGRA-bw16 (self)                                      6_124.53       712.95     6_837.48       0.9832          1.0011       236.51
CAGRA-bw30 (query)                                     6_124.53       561.27     6_685.80       0.9796          1.0010       236.51
CAGRA-bw30 (self)                                      6_124.53     1_250.42     7_374.95       0.9899          1.0007       236.51
CAGRA-bw48 (query)                                     6_124.53       663.56     6_788.09       0.9894          1.0005       236.51
CAGRA-bw48 (self)                                      6_124.53     2_190.21     8_314.75       0.9944          1.0004       236.51
CAGRA-bw64 (query)                                     6_124.53       793.42     6_917.95       0.9931          1.0003       236.51
CAGRA-bw64 (self)                                      6_124.53     3_238.84     9_363.38       0.9964          1.0002       236.51
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
CPU-Exhaustive (query)                                    20.81    40_562.71    40_583.52       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     20.81   385_845.02   385_865.84       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    32.57     8_768.97     8_801.54       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     32.57    85_821.01    85_853.58       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_118.92       744.96     6_863.89       0.9723          1.0017       350.95
CAGRA-auto (self)                                      6_118.92     2_382.32     8_501.24       0.9839          1.0012       350.95
CAGRA-bw16 (query)                                     6_118.92       608.87     6_727.79       0.9451          1.0035       350.95
CAGRA-bw16 (self)                                      6_118.92     1_167.64     7_286.56       0.9733          1.0019       350.95
CAGRA-bw30 (query)                                     6_118.92       766.95     6_885.87       0.9703          1.0018       350.95
CAGRA-bw30 (self)                                      6_118.92     2_183.06     8_301.98       0.9829          1.0012       350.95
CAGRA-bw48 (query)                                     6_118.92       899.60     7_018.52       0.9834          1.0010       350.95
CAGRA-bw48 (self)                                      6_118.92     4_026.76    10_145.68       0.9897          1.0007       350.95
CAGRA-bw64 (query)                                     6_118.92     1_112.99     7_231.91       0.9886          1.0007       350.95
CAGRA-bw64 (self)                                      6_118.92     6_119.75    12_238.67       0.9929          1.0005       350.95
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
CPU-Exhaustive (query)                                    47.05    84_711.73    84_758.79       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     47.05   853_423.29   853_470.35       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    70.78    14_004.58    14_075.36       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     70.78   137_404.50   137_475.27       1.0000          1.0000       244.14
CAGRA-auto (query)                                    15_654.98     1_117.60    16_772.59       0.9708          1.0015       473.02
CAGRA-auto (self)                                     15_654.98     2_672.75    18_327.74       0.9837          1.0012       473.02
CAGRA-bw16 (query)                                    15_654.98     1_014.86    16_669.84       0.9429          1.0031       473.02
CAGRA-bw16 (self)                                     15_654.98     1_398.77    17_053.76       0.9730          1.0019       473.02
CAGRA-bw30 (query)                                    15_654.98     1_124.88    16_779.86       0.9688          1.0016       473.02
CAGRA-bw30 (self)                                     15_654.98     2_477.11    18_132.09       0.9828          1.0012       473.02
CAGRA-bw48 (query)                                    15_654.98     1_302.54    16_957.52       0.9821          1.0009       473.02
CAGRA-bw48 (self)                                     15_654.98     4_363.85    20_018.83       0.9896          1.0007       473.02
CAGRA-bw64 (query)                                    15_654.98     1_507.08    17_162.07       0.9878          1.0006       473.02
CAGRA-bw64 (self)                                     15_654.98     6_467.26    22_122.24       0.9928          1.0005       473.02
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
GPU-Exhaustive (ground truth)                              8.57    15_052.86    15_061.43       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   5_186.80     1_189.35     6_376.15       1.0000          1.0000       276.93
GPU-NND bk=1x refine=0 (extract)                       1_211.62        41.55     1_253.17       0.7423          1.1127       144.96
GPU-NND bk=1x refine=0 (self-beam)                     1_211.62     1_091.64     2_303.26       0.9786          1.0019       144.96
GPU-NND bk=1x refine=1 (extract)                       1_291.58        40.72     1_332.30       0.8853          1.0873       144.96
GPU-NND bk=1x refine=1 (self-beam)                     1_291.58     1_084.19     2_375.77       0.9841          1.0013       144.96
GPU-NND bk=1x refine=2 (extract)                       1_334.57        40.72     1_375.29       0.9012          1.0853       144.96
GPU-NND bk=1x refine=2 (self-beam)                     1_334.57     1_080.98     2_415.56       0.9851          1.0012       144.96
GPU-NND bk=2x refine=0 (extract)                       1_402.85        39.92     1_442.77       0.7422          1.1127       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_402.85     1_084.83     2_487.68       0.9808          1.0016       144.96
GPU-NND bk=2x refine=1 (extract)                       1_708.82        40.69     1_749.51       0.9247          1.0827       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_708.82     1_071.88     2_780.70       0.9942          1.0003       144.96
GPU-NND bk=2x refine=2 (extract)                       2_071.39        40.28     2_111.67       0.9297          1.0822       144.96
GPU-NND bk=2x refine=2 (self-beam)                     2_071.39     1_073.23     3_144.62       0.9950          1.0002       144.96
GPU-NND bk=3x refine=0 (extract)                       1_769.03        40.52     1_809.56       0.7399          1.1132       144.96
GPU-NND bk=3x refine=0 (self-beam)                     1_769.03     1_094.52     2_863.55       0.9807          1.0016       144.96
GPU-NND bk=3x refine=1 (extract)                       2_592.86        40.43     2_633.29       0.9305          1.0821       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_592.86     1_070.85     3_663.71       0.9958          1.0002       144.96
GPU-NND bk=3x refine=2 (extract)                       3_315.01        39.99     3_355.00       0.9326          1.0819       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_315.01     1_069.48     4_384.48       0.9962          1.0001       144.96
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
GPU-Exhaustive (ground truth)                             19.34    21_549.47    21_568.81       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   6_487.20     1_653.54     8_140.74       1.0000          1.0000       365.97
GPU-NND bk=1x refine=0 (extract)                       1_596.65        41.02     1_637.67       0.7398          1.1129       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_596.65     1_224.53     2_821.18       0.9784          1.0019       175.48
GPU-NND bk=1x refine=1 (extract)                       1_983.10        40.71     2_023.81       0.8848          1.0873       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_983.10     1_160.77     3_143.87       0.9841          1.0013       175.48
GPU-NND bk=1x refine=2 (extract)                       2_354.04        40.71     2_394.75       0.9009          1.0853       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_354.04     1_159.27     3_513.31       0.9851          1.0012       175.48
GPU-NND bk=2x refine=0 (extract)                       1_844.66        40.07     1_884.73       0.7398          1.1129       175.48
GPU-NND bk=2x refine=0 (self-beam)                     1_844.66     1_168.44     3_013.10       0.9805          1.0016       175.48
GPU-NND bk=2x refine=1 (extract)                       3_367.07        40.89     3_407.96       0.9245          1.0826       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_367.07     1_153.17     4_520.24       0.9941          1.0003       175.48
GPU-NND bk=2x refine=2 (extract)                       4_712.35        40.46     4_752.81       0.9297          1.0821       175.48
GPU-NND bk=2x refine=2 (self-beam)                     4_712.35     1_148.51     5_860.86       0.9951          1.0002       175.48
GPU-NND bk=3x refine=0 (extract)                       2_253.04        40.71     2_293.75       0.7380          1.1133       175.48
GPU-NND bk=3x refine=0 (self-beam)                     2_253.04     1_174.20     3_427.24       0.9805          1.0016       175.48
GPU-NND bk=3x refine=1 (extract)                       5_051.62        40.69     5_092.31       0.9304          1.0820       175.48
GPU-NND bk=3x refine=1 (self-beam)                     5_051.62     1_151.12     6_202.74       0.9958          1.0002       175.48
GPU-NND bk=3x refine=2 (extract)                       7_568.21        40.40     7_608.61       0.9326          1.0818       175.48
GPU-NND bk=3x refine=2 (self-beam)                     7_568.21     1_154.90     8_723.11       0.9962          1.0001       175.48
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
GPU-Exhaustive (ground truth)                             17.93    59_405.23    59_423.17       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  13_110.26     2_849.91    15_960.17       0.9999          1.0000       631.89
GPU-NND bk=1x refine=0 (extract)                       2_018.68        82.28     2_100.96       0.6794          1.1260       289.92
GPU-NND bk=1x refine=0 (self-beam)                     2_018.68     2_218.58     4_237.26       0.9623          1.0035       289.92
GPU-NND bk=1x refine=1 (extract)                       2_280.10        80.53     2_360.63       0.8547          1.0912       289.92
GPU-NND bk=1x refine=1 (self-beam)                     2_280.10     2_190.40     4_470.50       0.9729          1.0023       289.92
GPU-NND bk=1x refine=2 (extract)                       2_612.28        80.97     2_693.25       0.8821          1.0874       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_612.28     2_182.63     4_794.92       0.9750          1.0021       289.92
GPU-NND bk=2x refine=0 (extract)                       2_454.58        80.25     2_534.83       0.6792          1.1260       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_454.58     2_207.25     4_661.82       0.9652          1.0032       289.92
GPU-NND bk=2x refine=1 (extract)                       3_536.96        81.23     3_618.18       0.9151          1.0835       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_536.96     2_167.00     5_703.96       0.9902          1.0006       289.92
GPU-NND bk=2x refine=2 (extract)                       4_543.83        80.99     4_624.82       0.9267          1.0822       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_543.83     2_163.33     6_707.16       0.9921          1.0004       289.92
GPU-NND bk=3x refine=0 (extract)                       3_128.89        81.92     3_210.81       0.6767          1.1266       289.92
GPU-NND bk=3x refine=0 (self-beam)                     3_128.89     2_219.26     5_348.15       0.9651          1.0032       289.92
GPU-NND bk=3x refine=1 (extract)                       5_473.54        80.29     5_553.82       0.9250          1.0824       289.92
GPU-NND bk=3x refine=1 (self-beam)                     5_473.54     2_167.33     7_640.86       0.9932          1.0003       289.92
GPU-NND bk=3x refine=2 (extract)                       7_227.58        80.96     7_308.54       0.9318          1.0817       289.92
GPU-NND bk=3x refine=2 (self-beam)                     7_227.58     2_165.48     9_393.06       0.9943          1.0002       289.92
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
GPU-Exhaustive (ground truth)                             34.89    85_196.28    85_231.17       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  14_555.89     4_062.61    18_618.51       0.9999          1.0000       803.96
GPU-NND bk=1x refine=0 (extract)                       3_013.17        80.75     3_093.92       0.6746          1.1267       350.95
GPU-NND bk=1x refine=0 (self-beam)                     3_013.17     2_440.06     5_453.23       0.9616          1.0035       350.95
GPU-NND bk=1x refine=1 (extract)                       4_272.74        82.06     4_354.80       0.8533          1.0912       350.95
GPU-NND bk=1x refine=1 (self-beam)                     4_272.74     2_362.23     6_634.97       0.9725          1.0023       350.95
GPU-NND bk=1x refine=2 (extract)                       5_398.80        80.63     5_479.43       0.8815          1.0873       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_398.80     2_355.47     7_754.27       0.9748          1.0021       350.95
GPU-NND bk=2x refine=0 (extract)                       3_347.62        81.61     3_429.24       0.6745          1.1267       350.95
GPU-NND bk=2x refine=0 (self-beam)                     3_347.62     2_381.31     5_728.93       0.9645          1.0032       350.95
GPU-NND bk=2x refine=1 (extract)                       8_559.42        80.76     8_640.18       0.9149          1.0834       350.95
GPU-NND bk=2x refine=1 (self-beam)                     8_559.42     2_329.51    10_888.93       0.9901          1.0006       350.95
GPU-NND bk=2x refine=2 (extract)                      13_054.52        80.81    13_135.34       0.9267          1.0821       350.95
GPU-NND bk=2x refine=2 (self-beam)                    13_054.52     2_326.57    15_381.10       0.9920          1.0004       350.95
GPU-NND bk=3x refine=0 (extract)                       4_217.22        81.32     4_298.55       0.6722          1.1273       350.95
GPU-NND bk=3x refine=0 (self-beam)                     4_217.22     2_373.35     6_590.57       0.9644          1.0032       350.95
GPU-NND bk=3x refine=1 (extract)                      13_739.15        80.78    13_819.92       0.9246          1.0824       350.95
GPU-NND bk=3x refine=1 (self-beam)                    13_739.15     2_329.74    16_068.88       0.9931          1.0003       350.95
GPU-NND bk=3x refine=2 (extract)                      21_917.89        81.26    21_999.15       0.9318          1.0816       350.95
GPU-NND bk=3x refine=2 (self-beam)                    21_917.89     2_339.01    24_256.89       0.9943          1.0002       350.95
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
GPU-Exhaustive (ground truth)                             35.32   236_019.75   236_055.08       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  24_275.26     6_874.82    31_150.08       0.9999          1.0000      1295.77
GPU-NND bk=1x refine=0 (extract)                       3_668.01       165.17     3_833.19       0.6161          1.1417       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_668.01     4_581.55     8_249.56       0.9405          1.0059       579.83
GPU-NND bk=1x refine=1 (extract)                       4_777.44       162.27     4_939.70       0.8166          1.0966       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_777.44     4_488.53     9_265.96       0.9580          1.0038       579.83
GPU-NND bk=1x refine=2 (extract)                       5_886.43       162.43     6_048.86       0.8582          1.0903       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_886.43     4_436.04    10_322.47       0.9621          1.0033       579.83
GPU-NND bk=2x refine=0 (extract)                       4_608.55       160.95     4_769.50       0.6160          1.1417       579.83
GPU-NND bk=2x refine=0 (self-beam)                     4_608.55     4_510.34     9_118.88       0.9439          1.0055       579.83
GPU-NND bk=2x refine=1 (extract)                       8_185.91       163.57     8_349.47       0.8995          1.0850       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_185.91     4_410.20    12_596.11       0.9845          1.0010       579.83
GPU-NND bk=2x refine=2 (extract)                      11_259.38       162.07    11_421.46       0.9222          1.0825       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_259.38     4_400.51    15_659.89       0.9883          1.0006       579.83
GPU-NND bk=3x refine=0 (extract)                       6_109.47       164.29     6_273.76       0.6138          1.1423       579.83
GPU-NND bk=3x refine=0 (self-beam)                     6_109.47     4_522.98    10_632.44       0.9436          1.0055       579.83
GPU-NND bk=3x refine=1 (extract)                      12_646.53       162.50    12_809.03       0.9134          1.0835       579.83
GPU-NND bk=3x refine=1 (self-beam)                    12_646.53     4_395.20    17_041.73       0.9893          1.0006       579.83
GPU-NND bk=3x refine=2 (extract)                      18_408.59       163.98    18_572.57       0.9304          1.0817       579.83
GPU-NND bk=3x refine=2 (self-beam)                    18_408.59     4_395.54    22_804.13       0.9921          1.0003       579.83
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
GPU-Exhaustive (ground truth)                             67.88   338_860.30   338_928.18       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  32_406.25     9_825.84    42_232.09       0.9999          1.0000      1487.90
GPU-NND bk=1x refine=0 (extract)                       5_275.47       164.69     5_440.16       0.6166          1.1408       701.90
GPU-NND bk=1x refine=0 (self-beam)                     5_275.47     4_918.27    10_193.74       0.9407          1.0058       701.90
GPU-NND bk=1x refine=1 (extract)                       9_472.81       164.11     9_636.92       0.8169          1.0963       701.90
GPU-NND bk=1x refine=1 (self-beam)                     9_472.81     4_824.82    14_297.62       0.9581          1.0038       701.90
GPU-NND bk=1x refine=2 (extract)                      13_417.20       163.11    13_580.30       0.8583          1.0901       701.90
GPU-NND bk=1x refine=2 (self-beam)                    13_417.20     4_778.51    18_195.70       0.9623          1.0033       701.90
GPU-NND bk=2x refine=0 (extract)                       6_621.13       161.88     6_783.01       0.6165          1.1409       701.90
GPU-NND bk=2x refine=0 (self-beam)                     6_621.13     4_902.59    11_523.71       0.9441          1.0054       701.90
GPU-NND bk=2x refine=1 (extract)                      21_733.63       173.85    21_907.48       0.8998          1.0849       701.90
GPU-NND bk=2x refine=1 (self-beam)                    21_733.63     4_808.21    26_541.84       0.9846          1.0010       701.90
GPU-NND bk=2x refine=2 (extract)                      35_873.34       163.68    36_037.01       0.9223          1.0824       701.90
GPU-NND bk=2x refine=2 (self-beam)                    35_873.34     4_779.69    40_653.03       0.9883          1.0006       701.90
GPU-NND bk=3x refine=0 (extract)                       8_279.09       162.96     8_442.05       0.6140          1.1415       701.90
GPU-NND bk=3x refine=0 (self-beam)                     8_279.09     4_913.41    13_192.50       0.9437          1.0054       701.90
GPU-NND bk=3x refine=1 (extract)                      36_495.59       163.04    36_658.63       0.9136          1.0834       701.90
GPU-NND bk=3x refine=1 (self-beam)                    36_495.59     4_778.03    41_273.62       0.9893          1.0006       701.90
GPU-NND bk=3x refine=2 (extract)                      61_079.65       164.23    61_243.88       0.9305          1.0816       701.90
GPU-NND bk=3x refine=2 (self-beam)                    61_079.65     4_747.11    65_826.76       0.9920          1.0003       701.90
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
GPU-Exhaustive (ground truth)                             85.13 1_480_568.22 1_480_653.36       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  69_229.23    21_228.18    90_457.42       0.9997          1.0000      3487.41
GPU-NND bk=1x refine=0 (extract)                       7_532.96       412.92     7_945.87       0.4813          1.1868      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_532.96    11_818.67    19_351.62       0.8865          1.0128      1449.59
GPU-NND bk=1x refine=1 (extract)                      11_314.73       434.38    11_749.11       0.7243          1.1131      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    11_314.73    11_434.54    22_749.27       0.9237          1.0077      1449.59
GPU-NND bk=1x refine=2 (extract)                      14_950.27       435.04    15_385.31       0.8021          1.0984      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    14_950.27    11_394.43    26_344.70       0.9341          1.0064      1449.59
GPU-NND bk=2x refine=0 (extract)                       9_695.79       417.71    10_113.50       0.4813          1.1868      1449.59
GPU-NND bk=2x refine=0 (self-beam)                     9_695.79    11_816.75    21_512.55       0.8899          1.0122      1449.59
GPU-NND bk=2x refine=1 (extract)                      23_021.71       412.86    23_434.57       0.8496          1.0914      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    23_021.71    11_289.47    34_311.18       0.9705          1.0023      1449.59
GPU-NND bk=2x refine=2 (extract)                      34_748.93       436.95    35_185.87       0.9102          1.0836      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    34_748.93    11_247.58    45_996.51       0.9810          1.0012      1449.59
GPU-NND bk=3x refine=0 (extract)                      13_364.00       409.69    13_773.69       0.4813          1.1868      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    13_364.00    11_816.02    25_180.02       0.8900          1.0122      1449.59
GPU-NND bk=3x refine=1 (extract)                      37_762.26       413.23    38_175.50       0.8497          1.0919      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    37_762.26    11_290.13    49_052.40       0.9747          1.0019      1449.59
GPU-NND bk=3x refine=2 (extract)                      59_666.86       412.39    60_079.25       0.9249          1.0820      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    59_666.86    11_247.10    70_913.96       0.9878          1.0005      1449.59
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
