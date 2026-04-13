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
Exhaustive (query)                                         3.21     1_540.41     1_543.62       1.0000          1.0000        18.31
Exhaustive (self)                                          3.21    15_578.17    15_581.37       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     4.89       638.79       643.68       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      4.89     5_405.14     5_410.03       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               394.45       295.71       690.17       0.9875          1.0010         1.15
IVF-GPU-nl273-np16 (query)                               394.45       337.18       731.63       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               394.45       412.81       807.27       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     394.45     1_585.85     1_980.30       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               758.77       285.70     1_044.47       0.9925          1.0006         1.15
IVF-GPU-nl387-np27 (query)                               758.77       401.99     1_160.76       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                     758.77     1_384.06     2_142.83       0.9998          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_455.99       284.93     1_740.92       0.9888          1.0008         1.15
IVF-GPU-nl547-np27 (query)                             1_455.99       326.94     1_782.93       0.9969          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             1_455.99       349.71     1_805.70       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_455.99     1_320.59     2_776.58       0.9998          1.0000         1.15
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
Exhaustive (query)                                         3.73     1_474.89     1_478.61       1.0000          1.0000        18.88
Exhaustive (self)                                          3.73    15_374.45    15_378.18       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     5.70       662.65       668.35       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      5.70     5_612.73     5_618.43       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               371.82       287.99       659.81       0.9881          1.0009         1.15
IVF-GPU-nl273-np16 (query)                               371.82       337.32       709.14       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               371.82       415.55       787.37       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     371.82     1_562.13     1_933.95       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               723.62       284.20     1_007.82       0.9930          1.0005         1.15
IVF-GPU-nl387-np27 (query)                               723.62       400.36     1_123.98       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                     723.62     1_401.53     2_125.15       0.9997          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_400.80       287.91     1_688.71       0.9899          1.0007         1.15
IVF-GPU-nl547-np27 (query)                             1_400.80       324.29     1_725.09       0.9972          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             1_400.80       341.85     1_742.65       0.9998          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_400.80     1_336.54     2_737.33       0.9998          1.0000         1.15
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
Exhaustive (query)                                         3.12     1_589.52     1_592.64       1.0000          1.0000        18.31
Exhaustive (self)                                          3.12    16_721.01    16_724.13       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.21       641.79       647.00       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.21     5_407.23     5_412.44       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               395.83       280.41       676.24       0.9999          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               395.83       239.50       635.33       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               395.83       371.04       766.87       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     395.83     1_470.77     1_866.60       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               769.31       273.43     1_042.73       0.9999          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               769.31       383.50     1_152.81       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     769.31     1_292.42     2_061.73       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_495.69       290.09     1_785.79       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_495.69       318.82     1_814.51       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_495.69       333.59     1_829.28       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_495.69     1_235.15     2_730.85       1.0000          1.0000         1.15
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
Exhaustive (query)                                         3.18     1_614.02     1_617.19       1.0000          1.0000        18.31
Exhaustive (self)                                          3.18    16_900.91    16_904.08       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.11       636.61       641.72       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.11     5_407.52     5_412.62       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               392.89       278.73       671.62       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               392.89       329.53       722.42       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               392.89       399.94       792.83       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     392.89     1_466.91     1_859.80       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               760.50       270.11     1_030.60       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               760.50       391.21     1_151.71       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     760.50     1_273.41     2_033.91       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_489.03       149.92     1_638.94       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_489.03       286.31     1_775.34       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_489.03       324.47     1_813.49       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_489.03     1_227.44     2_716.46       1.0000          1.0000         1.15
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
Exhaustive (query)                                        14.39     6_275.53     6_289.91       1.0000          1.0000        73.24
Exhaustive (self)                                         14.39    66_659.09    66_673.47       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    23.62     1_353.29     1_376.91       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     23.62    12_418.64    12_442.26       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               443.33       442.53       885.87       0.9998          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               443.33       515.80       959.13       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               443.33       627.22     1_070.56       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     443.33     3_913.72     4_357.06       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               784.89       461.24     1_246.13       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               784.89       578.96     1_363.85       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     784.89     3_365.23     4_150.12       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_533.73       325.68     1_859.42       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_533.73       440.50     1_974.23       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_533.73       542.90     2_076.64       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_533.73     3_095.82     4_629.56       1.0000          1.0000         1.15
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
Exhaustive (query)                                        10.95     4_968.96     4_979.91       1.0000          1.0000        61.04
Exhaustive (self)                                         10.95    85_102.26    85_113.22       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                 1_205.28       338.49     1_543.77       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                 1_205.28       366.38     1_571.66       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                 1_205.28       512.25     1_717.54       1.0000          1.0000        61.12
IVF-nl353 (self)                                       1_205.28     7_714.93     8_920.21       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 2_432.77       320.07     2_752.83       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 2_432.77       372.65     2_805.41       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 2_432.77       454.23     2_887.00       1.0000          1.0000        61.16
IVF-nl500 (self)                                       2_432.77     6_442.74     8_875.50       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 4_443.28       279.48     4_722.76       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 4_443.28       373.78     4_817.05       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 4_443.28       397.84     4_841.12       1.0000          1.0000        61.21
IVF-nl707 (self)                                       4_443.28     5_097.22     9_540.50       1.0000          1.0000        61.21
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
Exhaustive (query)                                        10.91     5_495.20     5_506.11       1.0000          1.0000        61.04
Exhaustive (self)                                         10.91    91_072.17    91_083.08       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.22     1_414.25     1_432.47       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.22    21_480.93    21_499.15       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                             1_150.48       522.81     1_673.29       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_150.48       513.62     1_664.10       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_150.48       603.99     1_754.46       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_150.48     5_250.99     6_401.47       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_349.05       487.41     2_836.46       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_349.05       543.63     2_892.68       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_349.05       577.16     2_926.21       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_349.05     4_696.28     7_045.33       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_511.69       416.61     4_928.30       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_511.69       526.14     5_037.83       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_511.69       551.31     5_063.00       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_511.69     4_075.86     8_587.55       1.0000          1.0000         1.91
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
Exhaustive (query)                                        24.21    11_456.82    11_481.03       1.0000          1.0000       122.07
Exhaustive (self)                                         24.21   190_937.51   190_961.73       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                   756.68       710.66     1_467.34       1.0000          1.0000       122.25
IVF-nl353-np18 (query)                                   756.68       750.84     1_507.51       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                   756.68     1_079.35     1_836.03       1.0000          1.0000       122.25
IVF-nl353 (self)                                         756.68    17_346.00    18_102.67       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 1_365.05       663.97     2_029.02       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 1_365.05       750.51     2_115.56       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 1_365.05       932.06     2_297.12       1.0000          1.0000       122.32
IVF-nl500 (self)                                       1_365.05    14_996.93    16_361.98       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 2_697.92       573.01     3_270.93       0.9999          1.0000       122.42
IVF-nl707-np35 (query)                                 2_697.92       758.04     3_455.96       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 2_697.92       801.19     3_499.11       1.0000          1.0000       122.42
IVF-nl707 (self)                                       2_697.92    12_765.53    15_463.45       1.0000          1.0000       122.42
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
Exhaustive (query)                                        24.72    11_553.62    11_578.34       1.0000          1.0000       122.07
Exhaustive (self)                                         24.72   198_815.57   198_840.28       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.93     2_220.35     2_257.28       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.93    34_852.51    34_889.44       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                               755.33       605.74     1_361.07       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               755.33       653.80     1_409.13       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               755.33       814.28     1_569.60       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                     755.33     9_241.42     9_996.75       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             1_407.22       633.34     2_040.55       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             1_407.22       645.23     2_052.45       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             1_407.22       788.99     2_196.21       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   1_407.22     8_083.38     9_490.60       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             2_714.37       578.74     3_293.10       0.9999          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             2_714.37       643.73     3_358.10       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             2_714.37       673.00     3_387.37       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   2_714.37     6_997.10     9_711.47       1.0000          1.0000         1.91
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
Exhaustive (query)                                        21.30    12_183.93    12_205.23       1.0000          1.0000       122.07
Exhaustive (self)                                         21.30   383_512.28   383_533.58       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 2_546.86       659.00     3_205.87       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 2_546.86       733.70     3_280.56       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 2_546.86       892.76     3_439.62       1.0000          1.0000       122.20
IVF-nl500 (self)                                       2_546.86    28_483.14    31_030.00       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 4_711.87       577.48     5_289.35       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 4_711.87       752.25     5_464.12       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 4_711.87       793.58     5_505.45       1.0000          1.0000       122.25
IVF-nl707 (self)                                       4_711.87    25_529.50    30_241.37       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                9_136.18       547.76     9_683.94       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                9_136.18       766.61     9_902.79       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                9_136.18       840.46     9_976.64       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      9_136.18    21_539.05    30_675.22       1.0000          1.0000       122.32
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
Exhaustive (query)                                        21.09    12_882.65    12_903.74       1.0000          1.0000       122.07
Exhaustive (self)                                         21.09   402_972.89   402_993.98       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    35.28     2_706.38     2_741.66       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     35.28    85_869.64    85_904.92       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             2_656.43       632.42     3_288.84       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_656.43       662.77     3_319.19       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_656.43       755.61     3_412.03       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_656.43    16_113.52    18_769.95       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             5_390.19       597.56     5_987.75       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             5_390.19       667.06     6_057.25       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             5_390.19       670.10     6_060.29       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   5_390.19    14_220.34    19_610.54       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                           10_144.20       553.32    10_697.51       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                           10_144.20       625.39    10_769.59       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                           10_144.20       659.99    10_804.19       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                 10_144.20    12_183.52    22_327.72       1.0000          1.0000         3.82
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
Exhaustive (query)                                        47.84    25_761.55    25_809.39       1.0000          1.0000       244.14
Exhaustive (self)                                         47.84   867_970.68   868_018.52       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 1_668.58     1_355.76     3_024.34       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 1_668.58     1_508.59     3_177.17       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 1_668.58     1_849.30     3_517.88       1.0000          1.0000       244.39
IVF-nl500 (self)                                       1_668.58    60_362.64    62_031.22       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 2_983.50     1_171.46     4_154.97       0.9999          1.0000       244.49
IVF-nl707-np35 (query)                                 2_983.50     1_533.16     4_516.66       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 2_983.50     1_649.42     4_632.92       1.0000          1.0000       244.49
IVF-nl707 (self)                                       2_983.50    52_739.65    55_723.16       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                                6_500.52     1_022.16     7_522.67       0.9998          1.0000       244.64
IVF-nl1000-np44 (query)                                6_500.52     1_399.42     7_899.94       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                                6_500.52     1_566.77     8_067.29       1.0000          1.0000       244.64
IVF-nl1000 (self)                                      6_500.52    45_095.09    51_595.61       1.0000          1.0000       244.64
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
Exhaustive (query)                                        46.82    26_389.35    26_436.17       1.0000          1.0000       244.14
Exhaustive (self)                                         46.82   886_607.05   886_653.88       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    78.33     4_361.49     4_439.82       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     78.33   138_841.98   138_920.31       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             1_601.26       986.87     2_588.14       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             1_601.26       986.18     2_587.44       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             1_601.26     1_205.10     2_806.36       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   1_601.26    29_604.44    31_205.70       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             2_958.18       913.20     3_871.38       0.9999          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             2_958.18     1_020.63     3_978.82       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             2_958.18     1_027.28     3_985.47       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   2_958.18    25_811.98    28_770.17       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            6_567.03       792.23     7_359.26       0.9998          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            6_567.03       996.22     7_563.24       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            6_567.03     1_026.06     7_593.09       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  6_567.03    22_229.59    28_796.62       1.0000          1.0000         3.82
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
CPU-Exhaustive (query)                                     3.14     1_577.61     1_580.76       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.14    17_126.88    17_130.02       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.32       641.96       647.28       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.32     5_451.04     5_456.36       1.0000          1.0000        18.31
CAGRA-auto (query)                                       618.64       129.05       747.69       0.9406          1.0034        86.98
CAGRA-auto (self)                                        618.64       779.99     1_398.63       0.9373          1.0049        86.98
CAGRA-bw16 (query)                                       618.64        90.40       709.04       0.9188          1.0046        86.98
CAGRA-bw16 (self)                                        618.64       351.35       969.99       0.9147          1.0068        86.98
CAGRA-bw30 (query)                                       618.64       125.49       744.13       0.9384          1.0035        86.98
CAGRA-bw30 (self)                                        618.64       715.60     1_334.24       0.9349          1.0051        86.98
CAGRA-bw48 (query)                                       618.64       190.43       809.07       0.9550          1.0025        86.98
CAGRA-bw48 (self)                                        618.64     1_383.50     2_002.14       0.9523          1.0037        86.98
CAGRA-bw64 (query)                                       618.64       281.39       900.03       0.9645          1.0020        86.98
CAGRA-bw64 (self)                                        618.64     2_149.37     2_768.01       0.9623          1.0030        86.98
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
CPU-Exhaustive (query)                                     3.93     1_665.60     1_669.52       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      3.93    17_365.31    17_369.24       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.29       670.06       676.36       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      6.29     5_638.51     5_644.80       1.0000          1.0000        18.88
CAGRA-auto (query)                                       634.92       197.17       832.09       0.9422          1.0034        87.55
CAGRA-auto (self)                                        634.92       787.80     1_422.72       0.9391          1.0047        87.55
CAGRA-bw16 (query)                                       634.92       116.14       751.07       0.9203          1.0047        87.55
CAGRA-bw16 (self)                                        634.92       357.19       992.11       0.9167          1.0066        87.55
CAGRA-bw30 (query)                                       634.92       151.55       786.48       0.9400          1.0036        87.55
CAGRA-bw30 (self)                                        634.92       719.81     1_354.73       0.9366          1.0049        87.55
CAGRA-bw48 (query)                                       634.92       263.44       898.36       0.9564          1.0026        87.55
CAGRA-bw48 (self)                                        634.92     1_395.08     2_030.00       0.9539          1.0035        87.55
CAGRA-bw64 (query)                                       634.92       331.70       966.63       0.9659          1.0020        87.55
CAGRA-bw64 (self)                                        634.92     2_164.22     2_799.14       0.9637          1.0028        87.55
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
CPU-Exhaustive (query)                                     3.12     1_616.40     1_619.52       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.12    16_706.91    16_710.03       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.56       651.23       656.79       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.56     5_462.19     5_467.75       1.0000          1.0000        18.31
CAGRA-auto (query)                                       567.13       124.97       692.10       0.9951          1.0003        86.98
CAGRA-auto (self)                                        567.13       717.46     1_284.59       0.9969          1.0002        86.98
CAGRA-bw16 (query)                                       567.13        88.21       655.34       0.9867          1.0007        86.98
CAGRA-bw16 (self)                                        567.13       340.24       907.36       0.9924          1.0004        86.98
CAGRA-bw30 (query)                                       567.13       124.91       692.04       0.9946          1.0003        86.98
CAGRA-bw30 (self)                                        567.13       663.09     1_230.22       0.9966          1.0002        86.98
CAGRA-bw48 (query)                                       567.13       183.23       750.35       0.9977          1.0001        86.98
CAGRA-bw48 (self)                                        567.13     1_229.42     1_796.54       0.9985          1.0001        86.98
CAGRA-bw64 (query)                                       567.13       242.89       810.02       0.9986          1.0001        86.98
CAGRA-bw64 (self)                                        567.13     1_861.13     2_428.26       0.9991          1.0001        86.98
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
CPU-Exhaustive (query)                                     3.23     1_590.31     1_593.54       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.23    16_713.89    16_717.13       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.56       647.34       652.90       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.56     5_466.91     5_472.47       1.0000          1.0000        18.31
CAGRA-auto (query)                                       544.49       123.01       667.50       0.9986          1.0001        86.98
CAGRA-auto (self)                                        544.49       713.12     1_257.61       0.9986          1.0001        86.98
CAGRA-bw16 (query)                                       544.49        86.51       630.99       0.9953          1.0002        86.98
CAGRA-bw16 (self)                                        544.49       338.98       883.46       0.9956          1.0003        86.98
CAGRA-bw30 (query)                                       544.49       125.20       669.68       0.9984          1.0001        86.98
CAGRA-bw30 (self)                                        544.49       656.76     1_201.24       0.9984          1.0001        86.98
CAGRA-bw48 (query)                                       544.49       257.16       801.65       0.9994          1.0000        86.98
CAGRA-bw48 (self)                                        544.49     1_209.73     1_754.22       0.9994          1.0001        86.98
CAGRA-bw64 (query)                                       544.49       239.29       783.77       0.9997          1.0000        86.98
CAGRA-bw64 (self)                                        544.49     1_816.32     2_360.81       0.9997          1.0000        86.98
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
CPU-Exhaustive (query)                                    14.25     6_308.38     6_322.63       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     14.25    65_223.92    65_238.17       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    22.37     1_360.24     1_382.61       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     22.37    12_566.56    12_588.93       1.0000          1.0000        73.24
CAGRA-auto (query)                                     3_152.02       272.07     3_424.09       0.9940          1.0003       141.91
CAGRA-auto (self)                                      3_152.02     1_187.26     4_339.27       0.9932          1.0005       141.91
CAGRA-bw16 (query)                                     3_152.02       215.35     3_367.37       0.9877          1.0006       141.91
CAGRA-bw16 (self)                                      3_152.02       617.56     3_769.58       0.9856          1.0009       141.91
CAGRA-bw30 (query)                                     3_152.02       266.59     3_418.61       0.9936          1.0004       141.91
CAGRA-bw30 (self)                                      3_152.02     1_109.90     4_261.92       0.9926          1.0005       141.91
CAGRA-bw48 (query)                                     3_152.02       347.62     3_499.64       0.9969          1.0002       141.91
CAGRA-bw48 (self)                                      3_152.02     1_899.89     5_051.91       0.9964          1.0003       141.91
CAGRA-bw64 (query)                                     3_152.02       426.58     3_578.60       0.9982          1.0001       141.91
CAGRA-bw64 (self)                                      3_152.02     2_730.17     5_882.19       0.9978          1.0002       141.91
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
CPU-Exhaustive (query)                                    11.16     8_453.20     8_464.35       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     11.16    88_476.19    88_487.35       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.47     2_333.73     2_352.20       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.47    21_557.24    21_575.71       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_521.03       384.49     2_905.53       0.9950          1.0002       175.48
CAGRA-auto (self)                                      2_521.03     1_490.20     4_011.24       0.9939          1.0004       175.48
CAGRA-bw16 (query)                                     2_521.03       303.67     2_824.70       0.9886          1.0005       175.48
CAGRA-bw16 (self)                                      2_521.03       735.17     3_256.21       0.9865          1.0008       175.48
CAGRA-bw30 (query)                                     2_521.03       371.37     2_892.40       0.9945          1.0003       175.48
CAGRA-bw30 (self)                                      2_521.03     1_380.17     3_901.20       0.9933          1.0004       175.48
CAGRA-bw48 (query)                                     2_521.03       479.09     3_000.12       0.9975          1.0001       175.48
CAGRA-bw48 (self)                                      2_521.03     2_484.41     5_005.45       0.9968          1.0002       175.48
CAGRA-bw64 (query)                                     2_521.03       673.52     3_194.55       0.9986          1.0001       175.48
CAGRA-bw64 (self)                                      2_521.03     3_686.43     6_207.46       0.9981          1.0001       175.48
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
CPU-Exhaustive (query)                                    23.91    18_660.71    18_684.62       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     23.91   190_696.83   190_720.74       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    37.47     3_650.13     3_687.60       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     37.47    34_841.92    34_879.39       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_914.03       621.37     7_535.40       0.9926          1.0004       236.51
CAGRA-auto (self)                                      6_914.03     2_016.29     8_930.32       0.9909          1.0006       236.51
CAGRA-bw16 (query)                                     6_914.03       537.34     7_451.37       0.9850          1.0008       236.51
CAGRA-bw16 (self)                                      6_914.03     1_053.29     7_967.32       0.9817          1.0012       236.51
CAGRA-bw30 (query)                                     6_914.03       658.10     7_572.13       0.9919          1.0004       236.51
CAGRA-bw30 (self)                                      6_914.03     1_875.33     8_789.35       0.9901          1.0007       236.51
CAGRA-bw48 (query)                                     6_914.03       758.12     7_672.15       0.9958          1.0002       236.51
CAGRA-bw48 (self)                                      6_914.03     3_220.47    10_134.50       0.9950          1.0004       236.51
CAGRA-bw64 (query)                                     6_914.03       896.49     7_810.51       0.9974          1.0001       236.51
CAGRA-bw64 (self)                                      6_914.03     4_652.64    11_566.67       0.9969          1.0002       236.51
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
CPU-Exhaustive (query)                                    20.55    40_514.26    40_534.81       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     20.55   399_625.00   399_645.55       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    32.53     8_731.18     8_763.71       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     32.53    85_867.91    85_900.45       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_175.98       795.38     6_971.37       0.9910          1.0005       350.95
CAGRA-auto (self)                                      6_175.98     3_036.44     9_212.42       0.9893          1.0007       350.95
CAGRA-bw16 (query)                                     6_175.98       631.16     6_807.14       0.9825          1.0009       350.95
CAGRA-bw16 (self)                                      6_175.98     1_494.29     7_670.28       0.9795          1.0013       350.95
CAGRA-bw30 (query)                                     6_175.98       762.25     6_938.23       0.9903          1.0005       350.95
CAGRA-bw30 (self)                                      6_175.98     2_801.09     8_977.07       0.9885          1.0008       350.95
CAGRA-bw48 (query)                                     6_175.98       986.58     7_162.56       0.9949          1.0003       350.95
CAGRA-bw48 (self)                                      6_175.98     5_063.29    11_239.28       0.9939          1.0004       350.95
CAGRA-bw64 (query)                                     6_175.98     1_242.14     7_418.12       0.9969          1.0002       350.95
CAGRA-bw64 (self)                                      6_175.98     7_526.76    13_702.74       0.9961          1.0003       350.95
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
CPU-Exhaustive (query)                                    46.11    87_013.66    87_059.77       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     46.11   867_978.48   868_024.59       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    79.81    14_101.25    14_181.06       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     79.81   138_826.04   138_905.84       1.0000          1.0000       244.14
CAGRA-auto (query)                                    16_858.81     1_298.08    18_156.89       0.9875          1.0007       473.02
CAGRA-auto (self)                                     16_858.81     4_079.83    20_938.64       0.9849          1.0010       473.02
CAGRA-bw16 (query)                                    16_858.81     1_162.40    18_021.21       0.9775          1.0013       473.02
CAGRA-bw16 (self)                                     16_858.81     2_147.98    19_006.79       0.9732          1.0018       473.02
CAGRA-bw30 (query)                                    16_858.81     1_294.30    18_153.11       0.9867          1.0008       473.02
CAGRA-bw30 (self)                                     16_858.81     3_812.72    20_671.53       0.9838          1.0011       473.02
CAGRA-bw48 (query)                                    16_858.81     1_525.52    18_384.33       0.9924          1.0005       473.02
CAGRA-bw48 (self)                                     16_858.81     6_512.51    23_371.32       0.9908          1.0006       473.02
CAGRA-bw64 (query)                                    16_858.81     1_818.28    18_677.09       0.9950          1.0003       473.02
CAGRA-bw64 (self)                                     16_858.81     9_471.82    26_330.63       0.9939          1.0004       473.02
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
GPU-Exhaustive (ground truth)                             10.09    15_101.80    15_111.90       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   4_796.61     1_122.61     5_919.22       1.0000          1.0000       279.88
GPU-NND bk=1x refine=0 (extract)                         841.88        42.09       883.97       0.8935          1.0888       144.96
GPU-NND bk=1x refine=0 (self-beam)                       841.88     1_224.98     2_066.86       0.9954          1.0004       144.96
GPU-NND bk=1x refine=1 (extract)                         891.38        41.28       932.66       0.9227          1.0848       144.96
GPU-NND bk=1x refine=1 (self-beam)                       891.38     1_225.73     2_117.11       0.9960          1.0003       144.96
GPU-NND bk=1x refine=2 (extract)                         948.06        42.36       990.42       0.9242          1.0847       144.96
GPU-NND bk=1x refine=2 (self-beam)                       948.06     1_226.24     2_174.29       0.9961          1.0003       144.96
GPU-NND bk=2x refine=0 (extract)                       1_215.54        41.61     1_257.15       0.9267          1.0844       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_215.54     1_220.87     2_436.41       0.9980          1.0001       144.96
GPU-NND bk=2x refine=1 (extract)                       1_532.60        41.46     1_574.06       0.9329          1.0837       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_532.60     1_216.46     2_749.06       0.9984          1.0001       144.96
GPU-NND bk=2x refine=2 (extract)                       1_866.35        41.81     1_908.16       0.9330          1.0837       144.96
GPU-NND bk=2x refine=2 (self-beam)                     1_866.35     1_224.40     3_090.75       0.9984          1.0001       144.96
GPU-NND bk=3x refine=0 (extract)                       2_212.68        41.04     2_253.73       0.9305          1.0840       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_212.68     1_225.00     3_437.68       0.9983          1.0001       144.96
GPU-NND bk=3x refine=1 (extract)                       2_911.06        41.77     2_952.83       0.9333          1.0837       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_911.06     1_221.82     4_132.88       0.9985          1.0001       144.96
GPU-NND bk=3x refine=2 (extract)                       3_604.38        41.54     3_645.91       0.9333          1.0837       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_604.38     1_231.01     4_835.39       0.9985          1.0001       144.96
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                              9.67    15_094.87    15_104.54       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   4_945.65     1_146.49     6_092.14       1.0000          1.0000       279.88
GPU-NND bk=1x refine=0 (extract)                         805.05        41.89       846.94       0.8935          1.0888       144.96
GPU-NND bk=1x refine=0 (self-beam)                       805.05     1_218.77     2_023.82       0.9954          1.0004       144.96
GPU-NND bk=1x refine=1 (extract)                         879.60        41.35       920.95       0.9227          1.0848       144.96
GPU-NND bk=1x refine=1 (self-beam)                       879.60     1_225.36     2_104.96       0.9960          1.0003       144.96
GPU-NND bk=1x refine=2 (extract)                         932.12        41.92       974.04       0.9242          1.0847       144.96
GPU-NND bk=1x refine=2 (self-beam)                       932.12     1_213.51     2_145.63       0.9961          1.0003       144.96
GPU-NND bk=2x refine=0 (extract)                       1_212.40        40.92     1_253.31       0.9266          1.0844       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_212.40     1_214.48     2_426.88       0.9980          1.0001       144.96
GPU-NND bk=2x refine=1 (extract)                       1_537.45        41.91     1_579.36       0.9329          1.0837       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_537.45     1_218.00     2_755.45       0.9984          1.0001       144.96
GPU-NND bk=2x refine=2 (extract)                       1_841.67        41.01     1_882.69       0.9330          1.0837       144.96
GPU-NND bk=2x refine=2 (self-beam)                     1_841.67     1_220.52     3_062.19       0.9984          1.0001       144.96
GPU-NND bk=3x refine=0 (extract)                       2_208.99        41.23     2_250.22       0.9305          1.0840       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_208.99     1_218.40     3_427.39       0.9983          1.0001       144.96
GPU-NND bk=3x refine=1 (extract)                       2_892.29        41.37     2_933.66       0.9333          1.0837       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_892.29     1_224.03     4_116.32       0.9985          1.0001       144.96
GPU-NND bk=3x refine=2 (extract)                       3_597.04        41.74     3_638.78       0.9333          1.0837       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_597.04     1_221.89     4_818.93       0.9985          1.0001       144.96
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
GPU-Exhaustive (ground truth)                             18.49    59_796.21    59_814.70       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  11_081.40     2_732.82    13_814.22       1.0000          1.0000       627.78
GPU-NND bk=1x refine=0 (extract)                       1_633.67        83.78     1_717.44       0.8738          1.0914       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_633.67     2_469.58     4_103.25       0.9916          1.0007       289.92
GPU-NND bk=1x refine=1 (extract)                       1_876.92        83.45     1_960.37       0.9149          1.0854       289.92
GPU-NND bk=1x refine=1 (self-beam)                     1_876.92     2_457.97     4_334.89       0.9928          1.0006       289.92
GPU-NND bk=1x refine=2 (extract)                       2_268.21        84.05     2_352.27       0.9178          1.0850       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_268.21     2_466.13     4_734.34       0.9930          1.0006       289.92
GPU-NND bk=2x refine=0 (extract)                       2_538.01        82.55     2_620.56       0.9229          1.0845       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_538.01     2_463.74     5_001.75       0.9966          1.0002       289.92
GPU-NND bk=2x refine=1 (extract)                       3_632.14        82.64     3_714.79       0.9325          1.0834       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_632.14     2_456.33     6_088.47       0.9974          1.0001       289.92
GPU-NND bk=2x refine=2 (extract)                       4_667.43        82.58     4_750.01       0.9326          1.0834       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_667.43     2_461.84     7_129.27       0.9974          1.0001       289.92
GPU-NND bk=3x refine=0 (extract)                       4_600.23        82.42     4_682.66       0.9293          1.0838       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_600.23     2_468.44     7_068.67       0.9972          1.0001       289.92
GPU-NND bk=3x refine=1 (extract)                       6_696.59        83.91     6_780.50       0.9332          1.0834       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_696.59     2_471.15     9_167.75       0.9976          1.0001       289.92
GPU-NND bk=3x refine=2 (extract)                       8_749.25        84.53     8_833.79       0.9333          1.0834       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_749.25     2_461.35    11_210.60       0.9976          1.0001       289.92
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 500k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             18.33    59_889.61    59_907.94       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  11_218.61     2_666.91    13_885.52       1.0000          1.0000       627.78
GPU-NND bk=1x refine=0 (extract)                       1_631.33        84.06     1_715.39       0.8738          1.0914       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_631.33     2_472.33     4_103.66       0.9916          1.0007       289.92
GPU-NND bk=1x refine=1 (extract)                       1_886.54        82.83     1_969.37       0.9148          1.0854       289.92
GPU-NND bk=1x refine=1 (self-beam)                     1_886.54     2_465.60     4_352.14       0.9928          1.0006       289.92
GPU-NND bk=1x refine=2 (extract)                       2_254.10        84.05     2_338.15       0.9178          1.0850       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_254.10     2_465.47     4_719.57       0.9930          1.0006       289.92
GPU-NND bk=2x refine=0 (extract)                       2_483.22        83.21     2_566.43       0.9229          1.0845       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_483.22     2_467.03     4_950.25       0.9965          1.0002       289.92
GPU-NND bk=2x refine=1 (extract)                       3_627.96        83.47     3_711.43       0.9324          1.0834       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_627.96     2_458.30     6_086.25       0.9973          1.0001       289.92
GPU-NND bk=2x refine=2 (extract)                       4_732.20        84.91     4_817.11       0.9326          1.0834       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_732.20     2_462.19     7_194.40       0.9974          1.0001       289.92
GPU-NND bk=3x refine=0 (extract)                       4_647.78        84.36     4_732.14       0.9293          1.0838       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_647.78     2_463.72     7_111.50       0.9972          1.0001       289.92
GPU-NND bk=3x refine=1 (extract)                       6_738.64        83.63     6_822.27       0.9332          1.0834       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_738.64     2_469.35     9_207.99       0.9976          1.0001       289.92
GPU-NND bk=3x refine=2 (extract)                       8_797.46        83.75     8_881.20       0.9333          1.0834       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_797.46     2_463.26    11_260.72       0.9976          1.0001       289.92
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
GPU-Exhaustive (ground truth)                             36.68   238_332.95   238_369.63       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  25_062.62     6_349.24    31_411.86       1.0000          1.0000      1143.55
GPU-NND bk=1x refine=0 (extract)                       3_460.19       172.13     3_632.32       0.8562          1.0938       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_460.19     5_008.02     8_468.21       0.9869          1.0011       579.83
GPU-NND bk=1x refine=1 (extract)                       4_458.92       168.73     4_627.65       0.9061          1.0862       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_458.92     4_981.39     9_440.31       0.9887          1.0010       579.83
GPU-NND bk=1x refine=2 (extract)                       5_291.90       170.55     5_462.45       0.9107          1.0856       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_291.90     5_000.35    10_292.25       0.9891          1.0009       579.83
GPU-NND bk=2x refine=0 (extract)                       5_340.22       167.58     5_507.80       0.9186          1.0847       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_340.22     4_999.75    10_339.97       0.9947          1.0003       579.83
GPU-NND bk=2x refine=1 (extract)                       8_425.14       167.10     8_592.24       0.9317          1.0832       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_425.14     4_991.48    13_416.62       0.9960          1.0002       579.83
GPU-NND bk=2x refine=2 (extract)                      11_698.46       168.16    11_866.61       0.9321          1.0832       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_698.46     4_983.97    16_682.43       0.9961          1.0002       579.83
GPU-NND bk=3x refine=0 (extract)                       9_542.36       170.27     9_712.63       0.9280          1.0836       579.83
GPU-NND bk=3x refine=0 (self-beam)                     9_542.36     4_999.40    14_541.76       0.9959          1.0002       579.83
GPU-NND bk=3x refine=1 (extract)                      15_667.55       170.87    15_838.42       0.9332          1.0831       579.83
GPU-NND bk=3x refine=1 (self-beam)                    15_667.55     5_000.15    20_667.70       0.9966          1.0001       579.83
GPU-NND bk=3x refine=2 (extract)                      21_727.35       173.59    21_900.94       0.9332          1.0831       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_727.35     5_012.48    26_739.83       0.9966          1.0001       579.83
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Generation of a kNN graph with CAGRA (1m samples; 64 dimensions)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 1000k samples, 32D kNN graph generation (build_k x refinement)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
GPU-Exhaustive (ground truth)                             36.42   238_383.68   238_420.09       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  25_212.08     6_395.85    31_607.92       1.0000          1.0000      1143.55
GPU-NND bk=1x refine=0 (extract)                       3_394.70       171.61     3_566.32       0.8562          1.0938       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_394.70     4_981.40     8_376.10       0.9869          1.0011       579.83
GPU-NND bk=1x refine=1 (extract)                       4_307.63       166.90     4_474.54       0.9061          1.0862       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_307.63     5_000.09     9_307.72       0.9887          1.0010       579.83
GPU-NND bk=1x refine=2 (extract)                       5_326.96       171.92     5_498.88       0.9107          1.0856       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_326.96     4_992.63    10_319.59       0.9891          1.0009       579.83
GPU-NND bk=2x refine=0 (extract)                       5_303.74       167.20     5_470.94       0.9186          1.0847       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_303.74     4_974.00    10_277.73       0.9947          1.0003       579.83
GPU-NND bk=2x refine=1 (extract)                       8_533.87       167.38     8_701.26       0.9317          1.0832       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_533.87     4_979.11    13_512.99       0.9960          1.0002       579.83
GPU-NND bk=2x refine=2 (extract)                      11_941.45       167.75    12_109.20       0.9321          1.0832       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_941.45     4_990.81    16_932.26       0.9961          1.0002       579.83
GPU-NND bk=3x refine=0 (extract)                       9_538.41       166.84     9_705.25       0.9280          1.0836       579.83
GPU-NND bk=3x refine=0 (self-beam)                     9_538.41     4_988.31    14_526.72       0.9959          1.0002       579.83
GPU-NND bk=3x refine=1 (extract)                      15_663.32       167.88    15_831.20       0.9332          1.0831       579.83
GPU-NND bk=3x refine=1 (self-beam)                    15_663.32     4_987.22    20_650.55       0.9965          1.0001       579.83
GPU-NND bk=3x refine=2 (extract)                      21_820.07       165.93    21_986.00       0.9332          1.0831       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_820.07     4_987.24    26_807.31       0.9966          1.0001       579.83
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
GPU-Exhaustive (ground truth)                             85.53 1_480_339.18 1_480_424.71       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  69_749.50    19_767.31    89_516.81       0.9999          1.0000      3254.84
GPU-NND bk=1x refine=0 (extract)                       7_349.06       430.97     7_780.04       0.7873          1.1048      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_349.06    12_807.03    20_156.09       0.9668          1.0032      1449.59
GPU-NND bk=1x refine=1 (extract)                      10_859.22       431.37    11_290.59       0.8696          1.0903      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    10_859.22    12_830.86    23_690.09       0.9722          1.0026      1449.59
GPU-NND bk=1x refine=2 (extract)                      14_115.40       432.69    14_548.09       0.8825          1.0884      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    14_115.40    12_795.74    26_911.14       0.9736          1.0024      1449.59
GPU-NND bk=2x refine=0 (extract)                      12_779.76       427.44    13_207.20       0.9035          1.0858      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    12_779.76    12_787.45    25_567.21       0.9884          1.0008      1449.59
GPU-NND bk=2x refine=1 (extract)                      23_965.43       428.80    24_394.22       0.9285          1.0828      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    23_965.43    12_754.76    36_720.19       0.9917          1.0004      1449.59
GPU-NND bk=2x refine=2 (extract)                      35_030.03       426.72    35_456.75       0.9297          1.0827      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    35_030.03    12_752.96    47_782.99       0.9921          1.0004      1449.59
GPU-NND bk=3x refine=0 (extract)                      24_921.42       421.61    25_343.03       0.9237          1.0833      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    24_921.42    12_809.37    37_730.79       0.9920          1.0004      1449.59
GPU-NND bk=3x refine=1 (extract)                      45_414.12       423.55    45_837.67       0.9327          1.0824      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    45_414.12    12_778.24    58_192.36       0.9936          1.0003      1449.59
GPU-NND bk=3x refine=2 (extract)                      66_246.38       418.54    66_664.92       0.9329          1.0824      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    66_246.38    12_772.00    79_018.38       0.9937          1.0003      1449.59
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
