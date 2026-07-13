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
Exhaustive (query)                                         3.11     1_531.15     1_534.26       1.0000          1.0000        18.31
Exhaustive (self)                                          3.11    14_356.56    14_359.67       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.40       644.60       650.00       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.40     5_459.74     5_465.13       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               384.98       249.14       634.12       0.9972          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               384.98       284.71       669.69       0.9996          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               384.98       331.05       716.03       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     384.98     1_465.30     1_850.28       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               722.71       286.49     1_009.20       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               722.71       322.47     1_045.17       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     722.71     1_315.86     2_038.56       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_577.36       334.65     1_912.00       0.9931          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_577.36       299.90     1_877.26       0.9984          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_577.36       360.03     1_937.39       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_577.36     1_399.03     2_976.39       1.0000          1.0000         1.15
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
Exhaustive (query)                                         4.04     1_579.94     1_583.98       1.0000          1.0000        18.88
Exhaustive (self)                                          4.04    16_560.89    16_564.93       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.29       711.40       717.69       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      6.29     5_641.56     5_647.85       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               381.26       287.00       668.27       0.9977          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               381.26       203.13       584.39       0.9998          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               381.26       342.41       723.68       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     381.26     1_519.45     1_900.72       0.9999          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               730.31       294.96     1_025.27       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               730.31       329.93     1_060.24       0.9999          1.0000         1.15
IVF-GPU-nl387 (self)                                     730.31     1_356.97     2_087.28       0.9999          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_424.82       281.41     1_706.22       0.9941          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_424.82       276.01     1_700.83       0.9987          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_424.82       294.76     1_719.58       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_424.82     1_415.48     2_840.29       0.9999          1.0000         1.15
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
Exhaustive (query)                                         3.03     1_504.85     1_507.88       1.0000          1.0000        18.31
Exhaustive (self)                                          3.03    15_813.97    15_816.99       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.25       638.78       644.03       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.25     5_432.55     5_437.80       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               397.33       286.53       683.86       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               397.33       277.00       674.33       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               397.33       335.87       733.19       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     397.33     1_476.66     1_873.99       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               746.89       285.57     1_032.46       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               746.89       332.99     1_079.88       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     746.89     1_313.89     2_060.79       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_450.20       281.09     1_731.29       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_450.20       267.02     1_717.22       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_450.20       291.24     1_741.44       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_450.20     1_255.92     2_706.12       1.0000          1.0000         1.15
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
Exhaustive (query)                                         3.14     1_530.22     1_533.35       1.0000          1.0000        18.31
Exhaustive (self)                                          3.14    16_003.17    16_006.30       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.59       638.90       644.49       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.59     5_447.16     5_452.75       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               385.31       247.98       633.28       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               385.31       272.55       657.85       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               385.31       327.53       712.84       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     385.31     1_327.42     1_712.73       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               749.95       169.08       919.03       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               749.95       303.67     1_053.63       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     749.95     1_243.79     1_993.74       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_458.18       172.91     1_631.10       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_458.18       249.28     1_707.47       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_458.18       282.76     1_740.94       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_458.18     1_240.29     2_698.48       1.0000          1.0000         1.15
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
Exhaustive (query)                                        14.50     6_423.14     6_437.64       1.0000          1.0000        73.24
Exhaustive (self)                                         14.50    64_809.30    64_823.80       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    21.86     1_412.33     1_434.19       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     21.86    12_431.66    12_453.53       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               612.59       405.54     1_018.13       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               612.59       416.01     1_028.61       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               612.59       515.55     1_128.15       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     612.59     3_267.56     3_880.15       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_154.78       355.20     1_509.98       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_154.78       461.81     1_616.59       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_154.78     2_965.77     4_120.55       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_423.23       309.67     2_732.90       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_423.23       389.42     2_812.65       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_423.23       464.88     2_888.11       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_423.23     2_832.05     5_255.28       1.0000          1.0000         1.15
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
Exhaustive (query)                                        10.58     5_025.13     5_035.71       1.0000          1.0000        61.04
Exhaustive (self)                                         10.58    88_484.19    88_494.77       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                 1_238.69       331.31     1_570.00       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                 1_238.69       320.39     1_559.08       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                 1_238.69       434.11     1_672.80       1.0000          1.0000        61.12
IVF-nl353 (self)                                       1_238.69     6_460.32     7_699.01       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 2_156.36       310.48     2_466.84       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 2_156.36       337.43     2_493.78       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 2_156.36       395.79     2_552.15       1.0000          1.0000        61.16
IVF-nl500 (self)                                       2_156.36     5_426.43     7_582.79       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 4_335.96       309.66     4_645.61       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 4_335.96       361.43     4_697.38       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 4_335.96       383.76     4_719.72       1.0000          1.0000        61.21
IVF-nl707 (self)                                       4_335.96     4_741.97     9_077.93       1.0000          1.0000        61.21
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
Exhaustive (query)                                        10.81     4_642.51     4_653.33       1.0000          1.0000        61.04
Exhaustive (self)                                         10.81    84_620.44    84_631.25       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    17.50     1_453.44     1_470.93       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     17.50    21_420.57    21_438.06       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                             1_120.29       407.07     1_527.35       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_120.29       422.39     1_542.67       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_120.29       491.55     1_611.84       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_120.29     4_689.97     5_810.26       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_105.31       453.91     2_559.22       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_105.31       430.33     2_535.64       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_105.31       428.48     2_533.79       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_105.31     4_343.55     6_448.86       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_155.94       413.54     4_569.48       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_155.94       433.30     4_589.24       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_155.94       444.73     4_600.67       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_155.94     3_982.85     8_138.79       1.0000          1.0000         1.91
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
Exhaustive (query)                                        23.10    10_521.08    10_544.18       1.0000          1.0000       122.07
Exhaustive (self)                                         23.10   178_279.78   178_302.88       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                 1_110.47       638.36     1_748.83       0.9999          1.0000       122.25
IVF-nl353-np18 (query)                                 1_110.47       670.39     1_780.86       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                 1_110.47       900.15     2_010.62       1.0000          1.0000       122.25
IVF-nl353 (self)                                       1_110.47    14_582.39    15_692.86       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 1_923.22       621.62     2_544.84       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 1_923.22       677.45     2_600.67       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 1_923.22       813.02     2_736.24       1.0000          1.0000       122.32
IVF-nl500 (self)                                       1_923.22    13_339.87    15_263.10       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 4_328.52       577.04     4_905.55       1.0000          1.0000       122.42
IVF-nl707-np35 (query)                                 4_328.52       701.80     5_030.32       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 4_328.52       740.13     5_068.65       1.0000          1.0000       122.42
IVF-nl707 (self)                                       4_328.52    11_666.38    15_994.89       1.0000          1.0000       122.42
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
Exhaustive (query)                                        23.44    10_273.74    10_297.18       1.0000          1.0000       122.07
Exhaustive (self)                                         23.44   176_053.13   176_076.57       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.36     2_201.24     2_237.60       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.36    34_503.46    34_539.81       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                             1_044.39       549.54     1_593.93       0.9999          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_044.39       420.48     1_464.87       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_044.39       647.40     1_691.78       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_044.39     8_031.46     9_075.85       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             1_994.73       606.78     2_601.51       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             1_994.73       545.35     2_540.08       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             1_994.73       623.99     2_618.73       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   1_994.73     7_311.35     9_306.08       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_356.55       583.06     4_939.61       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_356.55       556.63     4_913.19       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_356.55       567.19     4_923.74       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_356.55     6_653.60    11_010.15       1.0000          1.0000         1.91
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
Exhaustive (query)                                        20.77    10_899.62    10_920.39       1.0000          1.0000       122.07
Exhaustive (self)                                         20.77   369_218.67   369_239.44       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 2_421.85       620.09     3_041.95       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 2_421.85       674.75     3_096.60       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 2_421.85       801.01     3_222.87       1.0000          1.0000       122.20
IVF-nl500 (self)                                       2_421.85    26_201.09    28_622.94       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 4_484.46       571.04     5_055.50       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 4_484.46       693.25     5_177.71       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 4_484.46       722.81     5_207.27       1.0000          1.0000       122.25
IVF-nl707 (self)                                       4_484.46    22_986.21    27_470.67       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                8_263.40       518.41     8_781.80       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                8_263.40       673.35     8_936.74       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                8_263.40       742.26     9_005.66       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      8_263.40    20_832.33    29_095.73       1.0000          1.0000       122.32
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
Exhaustive (query)                                        20.05    10_445.20    10_465.25       1.0000          1.0000       122.07
Exhaustive (self)                                         20.05   367_924.01   367_944.06       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    31.69     2_734.87     2_766.55       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     31.69    85_177.30    85_208.99       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             2_325.22       566.76     2_891.98       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_325.22       586.63     2_911.85       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_325.22       642.17     2_967.39       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_325.22    15_196.49    17_521.71       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_361.62       601.65     4_963.28       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_361.62       606.01     4_967.64       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_361.62       584.93     4_946.55       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_361.62    13_710.29    18_071.92       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            8_002.21       600.73     8_602.94       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            8_002.21       561.68     8_563.90       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            8_002.21       625.41     8_627.63       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  8_002.21    12_634.76    20_636.97       1.0000          1.0000         3.82
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
Exhaustive (query)                                        45.08    24_336.48    24_381.56       1.0000          1.0000       244.14
Exhaustive (self)                                         45.08   814_771.70   814_816.79       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_179.42     1_265.58     3_445.00       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 2_179.42     1_379.36     3_558.78       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 2_179.42     1_634.54     3_813.96       1.0000          1.0000       244.39
IVF-nl500 (self)                                       2_179.42    53_392.11    55_571.54       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 4_581.81     1_152.17     5_733.99       1.0000          1.0000       244.49
IVF-nl707-np35 (query)                                 4_581.81     1_397.04     5_978.86       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 4_581.81     1_474.44     6_056.25       1.0000          1.0000       244.49
IVF-nl707 (self)                                       4_581.81    47_812.05    52_393.86       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                               10_218.04     1_033.23    11_251.27       0.9999          1.0000       244.64
IVF-nl1000-np44 (query)                               10_218.04     1_326.89    11_544.93       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                               10_218.04     1_478.00    11_696.04       1.0000          1.0000       244.64
IVF-nl1000 (self)                                     10_218.04    43_678.18    53_896.22       1.0000          1.0000       244.64
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
Exhaustive (query)                                        44.88    24_832.56    24_877.44       1.0000          1.0000       244.14
Exhaustive (self)                                         44.88   836_664.72   836_709.60       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    70.12     4_343.56     4_413.68       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     70.12   137_867.51   137_937.63       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             2_211.06       865.28     3_076.34       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_211.06       871.58     3_082.63       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_211.06     1_026.55     3_237.60       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_211.06    27_023.83    29_234.89       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_625.90       890.30     5_516.21       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_625.90       923.71     5_549.62       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_625.90       906.46     5_532.37       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_625.90    24_017.18    28_643.09       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                           10_297.61       836.56    11_134.18       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                           10_297.61       886.66    11_184.27       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                           10_297.61       955.62    11_253.23       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                 10_297.61    21_891.81    32_189.42       1.0000          1.0000         3.82
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
CPU-Exhaustive (query)                                     3.18     1_483.93     1_487.11       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.18    15_174.19    15_177.36       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.37       644.09       649.46       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.37     5_453.12     5_458.48       1.0000          1.0000        18.31
CAGRA-auto (query)                                     1_115.90       147.14     1_263.04       0.9321          1.0046        86.98
CAGRA-auto (self)                                      1_115.90       647.63     1_763.53       0.9307          1.0048        86.98
CAGRA-bw16 (query)                                     1_115.90       116.46     1_232.35       0.9189          1.0054        86.98
CAGRA-bw16 (self)                                      1_115.90       303.33     1_419.23       0.9198          1.0055        86.98
CAGRA-bw30 (query)                                     1_115.90       163.97     1_279.87       0.9309          1.0047        86.98
CAGRA-bw30 (self)                                      1_115.90       593.09     1_708.99       0.9296          1.0049        86.98
CAGRA-bw48 (query)                                     1_115.90       211.97     1_327.87       0.9406          1.0040        86.98
CAGRA-bw48 (self)                                      1_115.90     1_082.89     2_198.79       0.9384          1.0043        86.98
CAGRA-bw64 (query)                                     1_115.90       267.84     1_383.74       0.9466          1.0036        86.98
CAGRA-bw64 (self)                                      1_115.90     1_636.78     2_752.68       0.9441          1.0039        86.98
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
CPU-Exhaustive (query)                                     3.99     1_479.92     1_483.92       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      3.99    15_707.57    15_711.56       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.61       655.21       661.82       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      6.61     5_650.20     5_656.82       1.0000          1.0000        18.88
CAGRA-auto (query)                                     1_225.64       195.53     1_421.17       0.9312          1.0047        87.55
CAGRA-auto (self)                                      1_225.64       657.89     1_883.53       0.9307          1.0049        87.55
CAGRA-bw16 (query)                                     1_225.64       166.31     1_391.95       0.9161          1.0056        87.55
CAGRA-bw16 (self)                                      1_225.64       316.02     1_541.66       0.9184          1.0058        87.55
CAGRA-bw30 (query)                                     1_225.64       186.17     1_411.82       0.9299          1.0048        87.55
CAGRA-bw30 (self)                                      1_225.64       608.33     1_833.98       0.9295          1.0050        87.55
CAGRA-bw48 (query)                                     1_225.64       238.41     1_464.06       0.9404          1.0041        87.55
CAGRA-bw48 (self)                                      1_225.64     1_102.72     2_328.37       0.9394          1.0043        87.55
CAGRA-bw64 (query)                                     1_225.64       297.76     1_523.41       0.9473          1.0036        87.55
CAGRA-bw64 (self)                                      1_225.64     1_667.07     2_892.72       0.9457          1.0038        87.55
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
CPU-Exhaustive (query)                                     3.00     1_558.88     1_561.88       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.00    16_502.66    16_505.67       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.17       631.59       636.76       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.17     5_444.78     5_449.96       1.0000          1.0000        18.31
CAGRA-auto (query)                                     1_072.83       149.74     1_222.58       0.9813          1.0010        86.98
CAGRA-auto (self)                                      1_072.83       631.95     1_704.78       0.9928          1.0004        86.98
CAGRA-bw16 (query)                                     1_072.83        93.14     1_165.97       0.9573          1.0023        86.98
CAGRA-bw16 (self)                                      1_072.83       299.64     1_372.47       0.9873          1.0006        86.98
CAGRA-bw30 (query)                                     1_072.83       119.92     1_192.76       0.9798          1.0011        86.98
CAGRA-bw30 (self)                                      1_072.83       578.91     1_651.74       0.9923          1.0004        86.98
CAGRA-bw48 (query)                                     1_072.83       167.60     1_240.43       0.9898          1.0005        86.98
CAGRA-bw48 (self)                                      1_072.83     1_054.29     2_127.12       0.9957          1.0002        86.98
CAGRA-bw64 (query)                                     1_072.83       242.04     1_314.87       0.9936          1.0003        86.98
CAGRA-bw64 (self)                                      1_072.83     1_592.41     2_665.25       0.9972          1.0002        86.98
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
CPU-Exhaustive (query)                                     3.04     1_555.88     1_558.91       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.04    16_420.48    16_423.52       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.06       636.14       641.20       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.06     5_460.68     5_465.74       1.0000          1.0000        18.31
CAGRA-auto (query)                                     1_072.27       175.20     1_247.47       0.9873          1.0007        86.98
CAGRA-auto (self)                                      1_072.27       635.63     1_707.90       0.9942          1.0004        86.98
CAGRA-bw16 (query)                                     1_072.27       113.30     1_185.57       0.9704          1.0018        86.98
CAGRA-bw16 (self)                                      1_072.27       300.17     1_372.43       0.9890          1.0006        86.98
CAGRA-bw30 (query)                                     1_072.27       118.68     1_190.95       0.9862          1.0008        86.98
CAGRA-bw30 (self)                                      1_072.27       582.42     1_654.69       0.9937          1.0004        86.98
CAGRA-bw48 (query)                                     1_072.27       166.52     1_238.79       0.9930          1.0004        86.98
CAGRA-bw48 (self)                                      1_072.27     1_070.18     2_142.44       0.9968          1.0002        86.98
CAGRA-bw64 (query)                                     1_072.27       219.30     1_291.57       0.9957          1.0002        86.98
CAGRA-bw64 (self)                                      1_072.27     1_608.91     2_681.17       0.9980          1.0001        86.98
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
CPU-Exhaustive (query)                                    13.90     6_009.70     6_023.60       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     13.90    62_237.29    62_251.19       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    21.52     1_354.70     1_376.21       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     21.52    12_465.14    12_486.66       1.0000          1.0000        73.24
CAGRA-auto (query)                                     3_490.53       271.40     3_761.93       0.9867          1.0006       141.91
CAGRA-auto (self)                                      3_490.53       756.15     4_246.68       0.9942          1.0004       141.91
CAGRA-bw16 (query)                                     3_490.53       232.28     3_722.81       0.9693          1.0014       141.91
CAGRA-bw16 (self)                                      3_490.53       387.82     3_878.36       0.9890          1.0006       141.91
CAGRA-bw30 (query)                                     3_490.53       260.16     3_750.69       0.9857          1.0007       141.91
CAGRA-bw30 (self)                                      3_490.53       720.80     4_211.34       0.9937          1.0004       141.91
CAGRA-bw48 (query)                                     3_490.53       314.60     3_805.13       0.9927          1.0003       141.91
CAGRA-bw48 (self)                                      3_490.53     1_233.32     4_723.86       0.9967          1.0002       141.91
CAGRA-bw64 (query)                                     3_490.53       374.65     3_865.18       0.9954          1.0002       141.91
CAGRA-bw64 (self)                                      3_490.53     1_813.52     5_304.06       0.9979          1.0001       141.91
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
CPU-Exhaustive (query)                                    10.66     8_748.61     8_759.26       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     10.66    85_107.20    85_117.85       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    16.75     2_300.76     2_317.51       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     16.75    21_416.02    21_432.77       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_895.22       391.50     3_286.72       0.9830          1.0009       175.48
CAGRA-auto (self)                                      2_895.22     1_141.83     4_037.05       0.9923          1.0005       175.48
CAGRA-bw16 (query)                                     2_895.22       327.94     3_223.15       0.9623          1.0021       175.48
CAGRA-bw16 (self)                                      2_895.22       550.41     3_445.63       0.9862          1.0008       175.48
CAGRA-bw30 (query)                                     2_895.22       377.01     3_272.23       0.9816          1.0010       175.48
CAGRA-bw30 (self)                                      2_895.22     1_060.25     3_955.47       0.9918          1.0005       175.48
CAGRA-bw48 (query)                                     2_895.22       466.64     3_361.85       0.9903          1.0005       175.48
CAGRA-bw48 (self)                                      2_895.22     1_904.85     4_800.07       0.9955          1.0003       175.48
CAGRA-bw64 (query)                                     2_895.22       560.09     3_455.31       0.9939          1.0003       175.48
CAGRA-bw64 (self)                                      2_895.22     2_867.64     5_762.86       0.9971          1.0002       175.48
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
CPU-Exhaustive (query)                                    23.10    17_970.85    17_993.95       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     23.10   181_591.23   181_614.34       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    37.08     3_667.96     3_705.04       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     37.08    34_524.35    34_561.43       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_846.23       570.49     7_416.72       0.9817          1.0008       236.51
CAGRA-auto (self)                                      6_846.23     1_270.08     8_116.31       0.9923          1.0005       236.51
CAGRA-bw16 (query)                                     6_846.23       537.09     7_383.32       0.9608          1.0019       236.51
CAGRA-bw16 (self)                                      6_846.23       646.77     7_493.01       0.9861          1.0008       236.51
CAGRA-bw30 (query)                                     6_846.23       584.14     7_430.37       0.9803          1.0009       236.51
CAGRA-bw30 (self)                                      6_846.23     1_207.95     8_054.18       0.9918          1.0005       236.51
CAGRA-bw48 (query)                                     6_846.23       674.99     7_521.22       0.9898          1.0005       236.51
CAGRA-bw48 (self)                                      6_846.23     2_091.86     8_938.09       0.9955          1.0003       236.51
CAGRA-bw64 (query)                                     6_846.23       776.67     7_622.90       0.9935          1.0003       236.51
CAGRA-bw64 (self)                                      6_846.23     3_096.28     9_942.51       0.9971          1.0002       236.51
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
CPU-Exhaustive (query)                                    19.78    35_967.76    35_987.54       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     19.78   372_729.40   372_749.18       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    31.39     8_670.88     8_702.27       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     31.39    85_224.94    85_256.33       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_621.00       745.37     7_366.38       0.9738          1.0015       350.95
CAGRA-auto (self)                                      6_621.00     2_321.10     8_942.10       0.9870          1.0008       350.95
CAGRA-bw16 (query)                                     6_621.00       663.56     7_284.56       0.9479          1.0032       350.95
CAGRA-bw16 (self)                                      6_621.00     1_129.06     7_750.06       0.9789          1.0013       350.95
CAGRA-bw30 (query)                                     6_621.00       832.56     7_453.57       0.9719          1.0016       350.95
CAGRA-bw30 (self)                                      6_621.00     2_149.55     8_770.55       0.9862          1.0009       350.95
CAGRA-bw48 (query)                                     6_621.00       901.92     7_522.92       0.9842          1.0009       350.95
CAGRA-bw48 (self)                                      6_621.00     3_891.13    10_512.13       0.9917          1.0005       350.95
CAGRA-bw64 (query)                                     6_621.00     1_127.03     7_748.03       0.9894          1.0006       350.95
CAGRA-bw64 (self)                                      6_621.00     5_884.09    12_505.09       0.9942          1.0004       350.95
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
CPU-Exhaustive (query)                                    44.37    82_259.54    82_303.91       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     44.37   834_427.21   834_471.59       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    74.09    14_004.72    14_078.81       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     74.09   137_916.01   137_990.10       1.0000          1.0000       244.14
CAGRA-auto (query)                                    16_997.13     1_214.07    18_211.21       0.9724          1.0013       473.02
CAGRA-auto (self)                                     16_997.13     2_646.44    19_643.57       0.9869          1.0008       473.02
CAGRA-bw16 (query)                                    16_997.13     1_010.02    18_007.15       0.9459          1.0028       473.02
CAGRA-bw16 (self)                                     16_997.13     1_334.02    18_331.16       0.9787          1.0013       473.02
CAGRA-bw30 (query)                                    16_997.13     1_180.78    18_177.91       0.9705          1.0014       473.02
CAGRA-bw30 (self)                                     16_997.13     2_460.93    19_458.06       0.9861          1.0009       473.02
CAGRA-bw48 (query)                                    16_997.13     1_327.41    18_324.55       0.9832          1.0008       473.02
CAGRA-bw48 (self)                                     16_997.13     4_274.79    21_271.93       0.9916          1.0005       473.02
CAGRA-bw64 (query)                                    16_997.13     1_539.80    18_536.93       0.9886          1.0005       473.02
CAGRA-bw64 (self)                                     16_997.13     6_356.87    23_354.00       0.9942          1.0004       473.02
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
GPU-Exhaustive (ground truth)                              9.02    15_069.99    15_079.01       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   4_692.80     1_192.55     5_885.35       1.0000          1.0000       276.93
GPU-NND bk=1x refine=0 (extract)                       1_198.18        43.53     1_241.71       0.8499          1.0923       144.96
GPU-NND bk=1x refine=0 (self-beam)                     1_198.18     1_120.09     2_318.27       0.9837          1.0013       144.96
GPU-NND bk=1x refine=1 (extract)                       1_202.86        40.90     1_243.76       0.9021          1.0852       144.96
GPU-NND bk=1x refine=1 (self-beam)                     1_202.86     1_063.79     2_266.65       0.9858          1.0011       144.96
GPU-NND bk=1x refine=2 (extract)                       1_276.80        40.92     1_317.72       0.9066          1.0846       144.96
GPU-NND bk=1x refine=2 (self-beam)                     1_276.80     1_062.97     2_339.77       0.9862          1.0010       144.96
GPU-NND bk=2x refine=0 (extract)                       1_589.98        40.81     1_630.79       0.9126          1.0840       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_589.98     1_055.03     2_645.01       0.9930          1.0004       144.96
GPU-NND bk=2x refine=1 (extract)                       1_890.00        41.38     1_931.38       0.9304          1.0821       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_890.00     1_050.06     2_940.06       0.9950          1.0002       144.96
GPU-NND bk=2x refine=2 (extract)                       2_190.15        40.50     2_230.64       0.9309          1.0821       144.96
GPU-NND bk=2x refine=2 (self-beam)                     2_190.15     1_050.53     3_240.67       0.9952          1.0002       144.96
GPU-NND bk=3x refine=0 (extract)                       2_583.48        40.79     2_624.27       0.9260          1.0825       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_583.48     1_053.82     3_637.30       0.9951          1.0002       144.96
GPU-NND bk=3x refine=1 (extract)                       3_231.73        41.30     3_273.04       0.9329          1.0819       144.96
GPU-NND bk=3x refine=1 (self-beam)                     3_231.73     1_051.41     4_283.14       0.9962          1.0001       144.96
GPU-NND bk=3x refine=2 (extract)                       4_029.84        41.11     4_070.95       0.9330          1.0819       144.96
GPU-NND bk=3x refine=2 (self-beam)                     4_029.84     1_052.58     5_082.42       0.9963          1.0001       144.96
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
GPU-Exhaustive (ground truth)                             17.55    21_518.22    21_535.77       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   5_987.57     1_701.16     7_688.73       1.0000          1.0000       365.97
GPU-NND bk=1x refine=0 (extract)                       1_654.82        45.88     1_700.70       0.8494          1.0923       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_654.82     1_196.41     2_851.23       0.9836          1.0013       175.48
GPU-NND bk=1x refine=1 (extract)                       1_947.97        41.18     1_989.14       0.9019          1.0851       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_947.97     1_133.00     3_080.96       0.9857          1.0011       175.48
GPU-NND bk=1x refine=2 (extract)                       2_378.76        40.68     2_419.44       0.9064          1.0846       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_378.76     1_129.68     3_508.44       0.9861          1.0011       175.48
GPU-NND bk=2x refine=0 (extract)                       2_504.37        40.38     2_544.75       0.9126          1.0839       175.48
GPU-NND bk=2x refine=0 (self-beam)                     2_504.37     1_122.70     3_627.07       0.9929          1.0004       175.48
GPU-NND bk=2x refine=1 (extract)                       3_728.15        40.32     3_768.47       0.9304          1.0820       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_728.15     1_121.52     4_849.67       0.9951          1.0002       175.48
GPU-NND bk=2x refine=2 (extract)                       5_173.68        40.31     5_213.99       0.9309          1.0820       175.48
GPU-NND bk=2x refine=2 (self-beam)                     5_173.68     1_117.16     6_290.85       0.9952          1.0002       175.48
GPU-NND bk=3x refine=0 (extract)                       5_193.05        40.82     5_233.87       0.9261          1.0824       175.48
GPU-NND bk=3x refine=0 (self-beam)                     5_193.05     1_122.24     6_315.29       0.9952          1.0002       175.48
GPU-NND bk=3x refine=1 (extract)                       7_690.29        40.34     7_730.63       0.9329          1.0818       175.48
GPU-NND bk=3x refine=1 (self-beam)                     7_690.29     1_124.79     8_815.08       0.9962          1.0001       175.48
GPU-NND bk=3x refine=2 (extract)                      10_155.35        40.49    10_195.84       0.9330          1.0818       175.48
GPU-NND bk=3x refine=2 (self-beam)                    10_155.35     1_120.28    11_275.63       0.9962          1.0001       175.48
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
GPU-Exhaustive (ground truth)                             17.19    59_486.20    59_503.39       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  10_676.56     3_026.14    13_702.70       0.9999          1.0000       631.89
GPU-NND bk=1x refine=0 (extract)                       1_994.79        89.64     2_084.43       0.8143          1.0975       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_994.79     2_180.03     4_174.82       0.9719          1.0024       289.92
GPU-NND bk=1x refine=1 (extract)                       2_392.06        82.64     2_474.70       0.8827          1.0872       289.92
GPU-NND bk=1x refine=1 (self-beam)                     2_392.06     2_163.13     4_555.19       0.9758          1.0020       289.92
GPU-NND bk=1x refine=2 (extract)                       2_492.06        81.72     2_573.78       0.8906          1.0862       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_492.06     2_159.68     4_651.74       0.9766          1.0019       289.92
GPU-NND bk=2x refine=0 (extract)                       2_910.98        81.60     2_992.58       0.9019          1.0849       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_910.98     2_138.31     5_049.29       0.9884          1.0007       289.92
GPU-NND bk=2x refine=1 (extract)                       3_828.91        81.46     3_910.37       0.9279          1.0821       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_828.91     2_132.94     5_961.85       0.9921          1.0004       289.92
GPU-NND bk=2x refine=2 (extract)                       4_829.89        81.76     4_911.65       0.9290          1.0820       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_829.89     2_130.00     6_959.89       0.9924          1.0004       289.92
GPU-NND bk=3x refine=0 (extract)                       4_995.48        81.63     5_077.12       0.9227          1.0826       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_995.48     2_134.83     7_130.31       0.9926          1.0003       289.92
GPU-NND bk=3x refine=1 (extract)                       6_835.60        81.82     6_917.42       0.9325          1.0817       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_835.60     2_131.08     8_966.68       0.9944          1.0002       289.92
GPU-NND bk=3x refine=2 (extract)                       8_665.26        81.62     8_746.88       0.9327          1.0816       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_665.26     2_136.76    10_802.02       0.9945          1.0002       289.92
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
GPU-Exhaustive (ground truth)                             33.38    85_340.91    85_374.29       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  14_013.36     4_335.14    18_348.50       0.9999          1.0000       803.96
GPU-NND bk=1x refine=0 (extract)                       3_127.67        88.92     3_216.58       0.8129          1.0975       350.95
GPU-NND bk=1x refine=0 (self-beam)                     3_127.67     2_355.77     5_483.44       0.9715          1.0024       350.95
GPU-NND bk=1x refine=1 (extract)                       4_266.46        81.88     4_348.34       0.8820          1.0872       350.95
GPU-NND bk=1x refine=1 (self-beam)                     4_266.46     2_300.10     6_566.56       0.9755          1.0020       350.95
GPU-NND bk=1x refine=2 (extract)                       5_539.89        80.81     5_620.70       0.8902          1.0862       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_539.89     2_295.70     7_835.59       0.9763          1.0019       350.95
GPU-NND bk=2x refine=0 (extract)                       4_883.78        83.02     4_966.80       0.9015          1.0849       350.95
GPU-NND bk=2x refine=0 (self-beam)                     4_883.78     2_279.77     7_163.56       0.9883          1.0007       350.95
GPU-NND bk=2x refine=1 (extract)                       9_454.41        80.49     9_534.91       0.9277          1.0820       350.95
GPU-NND bk=2x refine=1 (self-beam)                     9_454.41     2_270.51    11_724.92       0.9920          1.0004       350.95
GPU-NND bk=2x refine=2 (extract)                      14_032.24        80.59    14_112.83       0.9289          1.0819       350.95
GPU-NND bk=2x refine=2 (self-beam)                    14_032.24     2_279.23    16_311.47       0.9924          1.0004       350.95
GPU-NND bk=3x refine=0 (extract)                      10_936.06        81.51    11_017.57       0.9227          1.0825       350.95
GPU-NND bk=3x refine=0 (self-beam)                    10_936.06     2_269.97    13_206.03       0.9926          1.0003       350.95
GPU-NND bk=3x refine=1 (extract)                      19_210.36        88.83    19_299.19       0.9325          1.0816       350.95
GPU-NND bk=3x refine=1 (self-beam)                    19_210.36     2_290.35    21_500.71       0.9944          1.0002       350.95
GPU-NND bk=3x refine=2 (extract)                      27_594.30        86.93    27_681.23       0.9327          1.0816       350.95
GPU-NND bk=3x refine=2 (self-beam)                    27_594.30     2_301.43    29_895.73       0.9945          1.0002       350.95
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
GPU-Exhaustive (ground truth)                             32.79   237_066.93   237_099.72       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  22_957.88     7_332.63    30_290.51       0.9999          1.0000      1295.77
GPU-NND bk=1x refine=0 (extract)                       3_951.11       182.01     4_133.12       0.7757          1.1037       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_951.11     4_429.16     8_380.27       0.9564          1.0040       579.83
GPU-NND bk=1x refine=1 (extract)                       4_726.50       163.17     4_889.67       0.8584          1.0901       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_726.50     4_405.67     9_132.17       0.9628          1.0032       579.83
GPU-NND bk=1x refine=2 (extract)                       5_806.18       162.52     5_968.70       0.8711          1.0884       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_806.18     4_376.81    10_182.99       0.9643          1.0031       579.83
GPU-NND bk=2x refine=0 (extract)                       5_897.62       161.16     6_058.79       0.8894          1.0862       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_897.62     4_334.73    10_232.36       0.9826          1.0012       579.83
GPU-NND bk=2x refine=1 (extract)                       8_993.76       163.25     9_157.01       0.9241          1.0822       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_993.76     4_315.43    13_309.19       0.9883          1.0006       579.83
GPU-NND bk=2x refine=2 (extract)                      12_148.33       161.51    12_309.84       0.9263          1.0820       579.83
GPU-NND bk=2x refine=2 (self-beam)                    12_148.33     4_327.64    16_475.97       0.9891          1.0005       579.83
GPU-NND bk=3x refine=0 (extract)                      10_517.72       161.32    10_679.03       0.9187          1.0828       579.83
GPU-NND bk=3x refine=0 (self-beam)                    10_517.72     4_344.69    14_862.40       0.9896          1.0005       579.83
GPU-NND bk=3x refine=1 (extract)                      16_038.43       163.34    16_201.77       0.9319          1.0815       579.83
GPU-NND bk=3x refine=1 (self-beam)                    16_038.43     4_333.81    20_372.24       0.9924          1.0003       579.83
GPU-NND bk=3x refine=2 (extract)                      21_615.39       169.43    21_784.82       0.9322          1.0815       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_615.39     4_347.56    25_962.95       0.9925          1.0003       579.83
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
GPU-Exhaustive (ground truth)                             62.71   341_681.45   341_744.15       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  30_134.61    10_359.87    40_494.48       0.9999          1.0000      1487.90
GPU-NND bk=1x refine=0 (extract)                       5_769.94       178.65     5_948.59       0.7761          1.1033       701.90
GPU-NND bk=1x refine=0 (self-beam)                     5_769.94     4_732.79    10_502.73       0.9565          1.0039       701.90
GPU-NND bk=1x refine=1 (extract)                       9_451.52       163.00     9_614.52       0.8586          1.0899       701.90
GPU-NND bk=1x refine=1 (self-beam)                     9_451.52     4_708.28    14_159.79       0.9629          1.0032       701.90
GPU-NND bk=1x refine=2 (extract)                      13_281.22       161.55    13_442.77       0.8712          1.0883       701.90
GPU-NND bk=1x refine=2 (self-beam)                    13_281.22     4_695.31    17_976.53       0.9644          1.0030       701.90
GPU-NND bk=2x refine=0 (extract)                       9_982.59       161.43    10_144.01       0.8895          1.0861       701.90
GPU-NND bk=2x refine=0 (self-beam)                     9_982.59     4_689.77    14_672.36       0.9827          1.0011       701.90
GPU-NND bk=2x refine=1 (extract)                      23_810.65       165.27    23_975.93       0.9242          1.0821       701.90
GPU-NND bk=2x refine=1 (self-beam)                    23_810.65     4_676.91    28_487.56       0.9883          1.0006       701.90
GPU-NND bk=2x refine=2 (extract)                      37_508.51       165.84    37_674.35       0.9262          1.0819       701.90
GPU-NND bk=2x refine=2 (self-beam)                    37_508.51     4_650.79    42_159.31       0.9890          1.0005       701.90
GPU-NND bk=3x refine=0 (extract)                      23_342.45       164.37    23_506.81       0.9188          1.0827       701.90
GPU-NND bk=3x refine=0 (self-beam)                    23_342.45     4_685.02    28_027.47       0.9895          1.0005       701.90
GPU-NND bk=3x refine=1 (extract)                      47_889.43       177.73    48_067.16       0.9319          1.0814       701.90
GPU-NND bk=3x refine=1 (self-beam)                    47_889.43     4_708.07    52_597.50       0.9923          1.0003       701.90
GPU-NND bk=3x refine=2 (extract)                      72_503.40       164.52    72_667.91       0.9322          1.0814       701.90
GPU-NND bk=3x refine=2 (self-beam)                    72_503.40     4_644.16    77_147.56       0.9925          1.0002       701.90
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
GPU-Exhaustive (ground truth)                             84.33 1_470_494.25 1_470_578.58       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  65_226.80    22_332.54    87_559.34       0.9997          1.0000      3487.41
GPU-NND bk=1x refine=0 (extract)                       8_361.26       461.32     8_822.58       0.6961          1.1193      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     8_361.26    11_404.64    19_765.90       0.9212          1.0080      1449.59
GPU-NND bk=1x refine=1 (extract)                      11_672.56       455.10    12_127.65       0.8038          1.0980      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    11_672.56    11_275.57    22_948.13       0.9338          1.0064      1449.59
GPU-NND bk=1x refine=2 (extract)                      15_148.53       447.04    15_595.57       0.8277          1.0943      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    15_148.53    11_196.12    26_344.66       0.9375          1.0060      1449.59
GPU-NND bk=2x refine=0 (extract)                      13_751.46       441.61    14_193.08       0.8660          1.0890      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    13_751.46    11_068.13    24_819.59       0.9718          1.0021      1449.59
GPU-NND bk=2x refine=1 (extract)                      25_310.64       427.52    25_738.16       0.9159          1.0829      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    25_310.64    11_040.83    36_351.47       0.9815          1.0011      1449.59
GPU-NND bk=2x refine=2 (extract)                      36_724.71       436.98    37_161.69       0.9203          1.0824      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    36_724.71    11_190.45    47_915.17       0.9831          1.0010      1449.59
GPU-NND bk=3x refine=0 (extract)                      27_171.04       459.41    27_630.45       0.9112          1.0833      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    27_171.04    11_053.01    38_224.05       0.9846          1.0008      1449.59
GPU-NND bk=3x refine=1 (extract)                      47_731.64       437.90    48_169.53       0.9304          1.0814      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    47_731.64    11_034.29    58_765.93       0.9893          1.0004      1449.59
GPU-NND bk=3x refine=2 (extract)                      68_829.38       460.84    69_290.22       0.9311          1.0813      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    68_829.38    11_037.05    79_866.42       0.9897          1.0004      1449.59
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
