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
cargo run --example gridsearch_nsg --features gpu --release
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

The GPU acceleration is particularly notable for the exhaustive index.

<details>
<summary><b>GPU - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 150k samples, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        11.57     1_397.60     1_409.17       1.0000          1.0000        18.31
Exhaustive (self)                                         11.57    14_622.74    14_634.30       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    12.75       408.81       421.56       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     12.75     3_218.18     3_230.93       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                                99.61        93.91       193.53       0.9972          1.0002         1.15
IVF-GPU-nl273-np16 (query)                                99.61       137.47       237.08       0.9997          1.0000         1.15
IVF-GPU-nl273-np23 (query)                                99.61       225.65       325.26       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                      99.61       840.35       939.96       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               163.27       191.60       354.87       0.9991          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               163.27       222.84       386.12       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     163.27       739.00       902.27       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               236.96       194.17       431.13       0.9937          1.0004         1.15
IVF-GPU-nl547-np27 (query)                               236.96       134.04       371.00       0.9987          1.0001         1.15
IVF-GPU-nl547-np33 (query)                               236.96       148.47       385.43       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     236.96       684.88       921.84       1.0000          1.0000         1.15
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
Exhaustive (query)                                        11.83     1_509.53     1_521.36       1.0000          1.0000        18.88
Exhaustive (self)                                         11.83    16_309.83    16_321.66       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                    13.18       397.67       410.86       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                     13.18     3_109.01     3_122.19       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               173.56       103.99       277.55       0.9980          1.0001         1.15
IVF-GPU-nl273-np16 (query)                               173.56       140.71       314.27       0.9998          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               173.56       157.84       331.39       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     173.56       907.89     1_081.45       0.9999          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               204.41       197.87       402.28       0.9991          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               204.41       237.77       442.18       0.9999          1.0000         1.15
IVF-GPU-nl387 (self)                                     204.41       770.61       975.02       0.9999          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               298.23       197.79       496.02       0.9946          1.0003         1.15
IVF-GPU-nl547-np27 (query)                               298.23       157.48       455.71       0.9987          1.0001         1.15
IVF-GPU-nl547-np33 (query)                               298.23       152.63       450.86       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                     298.23       717.23     1_015.46       0.9999          1.0000         1.15
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
Exhaustive (query)                                        11.50     1_537.60     1_549.09       1.0000          1.0000        18.31
Exhaustive (self)                                         11.50    16_110.19    16_121.69       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    12.91       407.08       419.99       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     12.91     3_221.18     3_234.09       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                                96.41       191.22       287.63       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                                96.41       138.91       235.32       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                                96.41       145.25       241.66       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                      96.41       711.63       808.03       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               158.01       191.99       350.00       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               158.01       150.03       308.04       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     158.01       645.09       803.11       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               232.04       197.61       429.64       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               232.04       212.63       444.67       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               232.04       152.99       385.03       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     232.04       731.64       963.68       1.0000          1.0000         1.15
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
Exhaustive (query)                                        11.23     1_547.62     1_558.84       1.0000          1.0000        18.31
Exhaustive (self)                                         11.23    16_122.48    16_133.71       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    12.48       404.55       417.03       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     12.48     3_222.59     3_235.07       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                                97.44        86.31       183.75       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                                97.44       127.25       224.69       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                                97.44       217.81       315.25       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                      97.44       774.19       871.63       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               161.30       188.87       350.16       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               161.30       220.51       381.81       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     161.30       681.62       842.92       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               236.45       196.17       432.62       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               236.45       206.61       443.06       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               236.45       227.05       463.50       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     236.45       723.74       960.18       1.0000          1.0000         1.15
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
Exhaustive (query)                                        49.17     5_916.26     5_965.43       1.0000          1.0000        73.24
Exhaustive (self)                                         49.17    61_783.62    61_832.79       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    54.14       853.44       907.58       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     54.14     7_492.05     7_546.19       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               348.68       264.60       613.28       0.9999          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               348.68       273.84       622.52       0.9999          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               348.68       329.84       678.52       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     348.68     1_725.86     2_074.54       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               510.65       184.19       694.84       0.9999          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               510.65       293.39       804.04       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     510.65     1_510.37     2_021.02       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                               804.74       176.20       980.93       0.9999          1.0000         1.15
IVF-GPU-nl547-np27 (query)                               804.74       195.21       999.95       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                               804.74       303.35     1_108.08       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                     804.74     1_510.05     2_314.78       1.0000          1.0000         1.15
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
Exhaustive (query)                                        50.75     5_347.72     5_398.47       1.0000          1.0000        61.04
Exhaustive (self)                                         50.75    84_615.36    84_666.11       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                   958.46       414.36     1_372.82       0.9997          1.0001        61.12
IVF-nl353-np18 (query)                                   958.46       435.36     1_393.81       0.9997          1.0001        61.12
IVF-nl353-np26 (query)                                   958.46       623.06     1_581.52       0.9999          1.0000        61.12
IVF-nl353 (self)                                         958.46     9_795.25    10_753.70       0.9999          1.0000        61.12
IVF-nl500-np22 (query)                                 1_864.18       420.13     2_284.31       0.9998          1.0000        61.16
IVF-nl500-np25 (query)                                 1_864.18       436.32     2_300.50       0.9999          1.0000        61.16
IVF-nl500-np31 (query)                                 1_864.18       538.71     2_402.89       1.0000          1.0000        61.16
IVF-nl500 (self)                                       1_864.18     8_271.21    10_135.39       0.9999          1.0000        61.16
IVF-nl707-np26 (query)                                 3_758.38       344.36     4_102.74       0.9999          1.0000        61.21
IVF-nl707-np35 (query)                                 3_758.38       457.37     4_215.75       0.9999          1.0000        61.21
IVF-nl707-np37 (query)                                 3_758.38       480.90     4_239.28       1.0000          1.0000        61.21
IVF-nl707 (self)                                       3_758.38     7_231.28    10_989.66       0.9999          1.0000        61.21
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
Exhaustive (query)                                        46.10     4_877.84     4_923.94       1.0000          1.0000        61.04
Exhaustive (self)                                         46.10    82_492.41    82_538.51       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    52.52       929.30       981.83       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     52.52    12_638.45    12_690.97       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                               288.92       270.25       559.18       0.9997          1.0001         1.91
IVF-GPU-nl353-np18 (query)                               288.92       265.64       554.56       0.9997          1.0001         1.91
IVF-GPU-nl353-np26 (query)                               288.92       312.45       601.37       0.9999          1.0000         1.91
IVF-GPU-nl353 (self)                                     288.92     2_489.93     2_778.86       0.9999          1.0000         1.91
IVF-GPU-nl500-np22 (query)                               407.17       254.50       661.67       0.9998          1.0000         1.91
IVF-GPU-nl500-np25 (query)                               407.17       279.66       686.84       0.9999          1.0000         1.91
IVF-GPU-nl500-np31 (query)                               407.17       294.30       701.47       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                     407.17     2_238.88     2_646.05       0.9999          1.0000         1.91
IVF-GPU-nl707-np26 (query)                               633.67       250.72       884.39       0.9999          1.0000         1.91
IVF-GPU-nl707-np35 (query)                               633.67       280.48       914.15       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                               633.67       290.05       923.72       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                     633.67     2_139.41     2_773.08       0.9999          1.0000         1.91
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
Exhaustive (query)                                        98.04    10_773.05    10_871.08       1.0000          1.0000       122.07
Exhaustive (self)                                         98.04   182_540.06   182_638.09       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                 1_052.76       857.77     1_910.53       0.9999          1.0000       122.25
IVF-nl353-np18 (query)                                 1_052.76       906.27     1_959.03       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                 1_052.76     1_289.70     2_342.47       1.0000          1.0000       122.25
IVF-nl353 (self)                                       1_052.76    21_322.94    22_375.71       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 1_976.78       800.79     2_777.57       0.9999          1.0000       122.32
IVF-nl500-np25 (query)                                 1_976.78       902.98     2_879.76       0.9999          1.0000       122.32
IVF-nl500-np31 (query)                                 1_976.78     1_118.27     3_095.05       1.0000          1.0000       122.32
IVF-nl500 (self)                                       1_976.78    18_342.82    20_319.60       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 3_986.33       681.83     4_668.16       1.0000          1.0000       122.42
IVF-nl707-np35 (query)                                 3_986.33       897.78     4_884.11       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 3_986.33       949.47     4_935.80       1.0000          1.0000       122.42
IVF-nl707 (self)                                       3_986.33    15_453.77    19_440.10       1.0000          1.0000       122.42
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
Exhaustive (query)                                        85.05    10_640.21    10_725.26       1.0000          1.0000       122.07
Exhaustive (self)                                         85.05   180_570.05   180_655.10       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   117.35     1_390.65     1_508.01       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    117.35    21_023.12    21_140.47       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                               553.79       398.96       952.74       0.9999          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               553.79       346.82       900.60       0.9999          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               553.79       418.09       971.87       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                     553.79     4_110.94     4_664.72       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                               862.69       357.44     1_220.12       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                               862.69       367.55     1_230.24       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                               862.69       395.36     1_258.04       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                     862.69     3_707.66     4_570.34       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             1_371.89       343.22     1_715.11       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             1_371.89       365.20     1_737.09       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             1_371.89       380.68     1_752.57       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   1_371.89     3_242.86     4_614.75       1.0000          1.0000         1.91
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

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
Exhaustive (query)                                       136.88    11_772.99    11_909.87       1.0000          1.0000       122.07
Exhaustive (self)                                        136.88   380_880.36   381_017.24       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 2_405.98       814.72     3_220.70       0.9999          1.0000       122.20
IVF-nl500-np25 (query)                                 2_405.98       926.06     3_332.03       0.9999          1.0000       122.20
IVF-nl500-np31 (query)                                 2_405.98     1_137.14     3_543.11       1.0000          1.0000       122.20
IVF-nl500 (self)                                       2_405.98    37_621.32    40_027.30       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 4_136.06       697.15     4_833.21       0.9999          1.0000       122.25
IVF-nl707-np35 (query)                                 4_136.06       931.26     5_067.31       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 4_136.06       982.12     5_118.18       1.0000          1.0000       122.25
IVF-nl707 (self)                                       4_136.06    32_326.91    36_462.97       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                8_329.60       607.14     8_936.74       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                8_329.60       853.11     9_182.72       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                8_329.60       969.34     9_298.94       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      8_329.60    27_779.45    36_109.05       1.0000          1.0000       122.32
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
Exhaustive (query)                                       141.00    11_691.35    11_832.35       1.0000          1.0000       122.07
Exhaustive (self)                                        141.00   374_865.38   375_006.38       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   144.50     1_659.31     1_803.81       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    144.50    51_074.68    51_219.18       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                               636.44       369.01     1_005.45       0.9999          1.0000         3.82
IVF-GPU-nl500-np25 (query)                               636.44       385.73     1_022.17       0.9999          1.0000         3.82
IVF-GPU-nl500-np31 (query)                               636.44       446.42     1_082.86       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                     636.44     8_451.45     9_087.90       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                               867.50       353.53     1_221.03       0.9999          1.0000         3.82
IVF-GPU-nl707-np35 (query)                               867.50       411.68     1_279.18       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                               867.50       391.77     1_259.27       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                     867.50     7_492.12     8_359.62       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            1_302.16       335.12     1_637.28       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            1_302.16       367.07     1_669.23       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            1_302.16       389.36     1_691.52       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  1_302.16     6_459.30     7_761.46       1.0000          1.0000         3.82
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
Exhaustive (query)                                       279.68    24_938.54    25_218.22       1.0000          1.0000       244.14
Exhaustive (self)                                        279.68   865_361.53   865_641.21       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_209.09     1_602.08     3_811.16       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 2_209.09     1_811.49     4_020.57       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 2_209.09     2_239.55     4_448.64       1.0000          1.0000       244.39
IVF-nl500 (self)                                       2_209.09    74_364.16    76_573.24       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 4_304.61     1_328.69     5_633.30       1.0000          1.0000       244.49
IVF-nl707-np35 (query)                                 4_304.61     1_779.98     6_084.59       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 4_304.61     1_877.21     6_181.81       1.0000          1.0000       244.49
IVF-nl707 (self)                                       4_304.61    62_322.17    66_626.78       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                                9_440.29     1_119.58    10_559.87       1.0000          1.0000       244.64
IVF-nl1000-np44 (query)                                9_440.29     1_584.66    11_024.95       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                                9_440.29     1_793.24    11_233.53       1.0000          1.0000       244.64
IVF-nl1000 (self)                                      9_440.29    53_217.05    62_657.34       1.0000          1.0000       244.64
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
Exhaustive (query)                                       265.16    24_652.08    24_917.24       1.0000          1.0000       244.14
Exhaustive (self)                                        265.16   849_284.17   849_549.33       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                   276.78     2_694.14     2_970.91       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                    276.78    83_280.51    83_557.28       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             1_146.98       467.87     1_614.85       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             1_146.98       477.33     1_624.31       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             1_146.98       530.02     1_677.00       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   1_146.98    12_297.66    13_444.64       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             1_604.34       429.32     2_033.66       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             1_604.34       473.01     2_077.35       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             1_604.34       481.74     2_086.08       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   1_604.34    10_603.31    12_207.65       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            2_536.99       407.96     2_944.94       1.0000          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            2_536.99       449.57     2_986.56       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            2_536.99       479.08     3_016.07       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  2_536.99     9_307.17    11_844.16       1.0000          1.0000         3.82
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

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
`CagraGpuSearchParams::from_graph()`.

#### Parameter guidance

The two key build parameters are `build_k` (internal NNDescent degree before
CAGRA pruning) and `refine_knn` (number of 2-hop refinement sweeps after
NNDescent convergence).

**Key parameters:**

* `build_k`: Internal NNDescent degree before CAGRA pruning. Defaults to
  `1.5 * k`. Higher values give CAGRA more edges to select from when building
  the navigational graph, at the cost of build time. 3 * k shows diminishing
  returns.
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
CPU-Exhaustive (query)                                    11.31     1_479.27     1_490.58       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.31    15_263.42    15_274.73       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    12.86       408.17       421.03       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     12.86     3_235.57     3_248.44       1.0000          1.0000        18.31
CAGRA-auto (query)                                       442.55       124.19       566.75       0.9318          1.0046        61.23
CAGRA-auto (self)                                        442.55       637.94     1_080.49       0.9297          1.0049        61.23
CAGRA-bw16 (query)                                       442.55        92.94       535.50       0.9186          1.0054        61.23
CAGRA-bw16 (self)                                        442.55       303.22       745.77       0.9187          1.0057        61.23
CAGRA-bw30 (query)                                       442.55       115.72       558.28       0.9306          1.0047        61.23
CAGRA-bw30 (self)                                        442.55       598.12     1_040.67       0.9286          1.0050        61.23
CAGRA-bw48 (query)                                       442.55       171.59       614.14       0.9404          1.0041        61.23
CAGRA-bw48 (self)                                        442.55     1_094.12     1_536.67       0.9375          1.0044        61.23
CAGRA-bw64 (query)                                       442.55       248.61       691.16       0.9462          1.0037        61.23
CAGRA-bw64 (self)                                        442.55     1_640.74     2_083.30       0.9432          1.0040        61.23
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
CPU-Exhaustive (query)                                    12.06     1_481.34     1_493.40       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                     12.06    15_741.16    15_753.22       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                    13.83       397.52       411.36       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                     13.83     3_107.28     3_121.11       1.0000          1.0000        18.88
CAGRA-auto (query)                                       527.39       204.46       731.84       0.9309          1.0048        61.80
CAGRA-auto (self)                                        527.39       652.23     1_179.61       0.9298          1.0050        61.80
CAGRA-bw16 (query)                                       527.39       157.22       684.61       0.9156          1.0057        61.80
CAGRA-bw16 (self)                                        527.39       309.19       836.57       0.9175          1.0059        61.80
CAGRA-bw30 (query)                                       527.39       184.30       711.69       0.9297          1.0049        61.80
CAGRA-bw30 (self)                                        527.39       610.25     1_137.63       0.9286          1.0051        61.80
CAGRA-bw48 (query)                                       527.39       267.76       795.15       0.9401          1.0042        61.80
CAGRA-bw48 (self)                                        527.39     1_247.77     1_775.16       0.9386          1.0044        61.80
CAGRA-bw64 (query)                                       527.39       297.14       824.52       0.9470          1.0037        61.80
CAGRA-bw64 (self)                                        527.39     1_680.39     2_207.78       0.9449          1.0039        61.80
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
CPU-Exhaustive (query)                                    11.23     1_527.63     1_538.86       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.23    16_272.13    16_283.36       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    12.64       413.36       426.00       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     12.64     3_223.66     3_236.30       1.0000          1.0000        18.31
CAGRA-auto (query)                                       487.62       122.14       609.76       0.9814          1.0010        61.23
CAGRA-auto (self)                                        487.62       623.37     1_110.99       0.9928          1.0004        61.23
CAGRA-bw16 (query)                                       487.62        93.62       581.24       0.9574          1.0023        61.23
CAGRA-bw16 (self)                                        487.62       299.08       786.71       0.9873          1.0006        61.23
CAGRA-bw30 (query)                                       487.62       115.41       603.04       0.9798          1.0010        61.23
CAGRA-bw30 (self)                                        487.62       583.87     1_071.49       0.9922          1.0004        61.23
CAGRA-bw48 (query)                                       487.62       163.07       650.69       0.9897          1.0005        61.23
CAGRA-bw48 (self)                                        487.62     1_063.35     1_550.97       0.9957          1.0002        61.23
CAGRA-bw64 (query)                                       487.62       223.37       710.99       0.9935          1.0003        61.23
CAGRA-bw64 (self)                                        487.62     1_595.88     2_083.51       0.9972          1.0002        61.23
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
CPU-Exhaustive (query)                                    11.23     1_550.67     1_561.89       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                     11.23    16_910.62    16_921.85       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                    14.81       448.17       462.98       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                     14.81     3_211.26     3_226.07       1.0000          1.0000        18.31
CAGRA-auto (query)                                       437.23       124.09       561.33       0.9872          1.0007        61.23
CAGRA-auto (self)                                        437.23       634.65     1_071.89       0.9940          1.0004        61.23
CAGRA-bw16 (query)                                       437.23       116.64       553.88       0.9704          1.0018        61.23
CAGRA-bw16 (self)                                        437.23       300.13       737.36       0.9888          1.0007        61.23
CAGRA-bw30 (query)                                       437.23       137.76       574.99       0.9861          1.0008        61.23
CAGRA-bw30 (self)                                        437.23       587.33     1_024.56       0.9936          1.0004        61.23
CAGRA-bw48 (query)                                       437.23       168.05       605.28       0.9930          1.0004        61.23
CAGRA-bw48 (self)                                        437.23     1_072.15     1_509.38       0.9966          1.0002        61.23
CAGRA-bw64 (query)                                       437.23       224.76       662.00       0.9957          1.0002        61.23
CAGRA-bw64 (self)                                        437.23     1_612.74     2_049.97       0.9979          1.0001        61.23
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
CPU-Exhaustive (query)                                    49.59     5_917.32     5_966.91       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     49.59    62_799.23    62_848.82       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    56.81       850.54       907.35       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     56.81     7_494.70     7_551.51       1.0000          1.0000        73.24
CAGRA-auto (query)                                     1_538.76       227.80     1_766.56       0.9999          1.0000       116.16
CAGRA-auto (self)                                      1_538.76       684.61     2_223.37       0.9999          1.0000       116.16
CAGRA-bw16 (query)                                     1_538.76       200.83     1_739.59       0.9998          1.0000       116.16
CAGRA-bw16 (self)                                      1_538.76       381.43     1_920.19       0.9998          1.0000       116.16
CAGRA-bw30 (query)                                     1_538.76       227.21     1_765.97       0.9999          1.0000       116.16
CAGRA-bw30 (self)                                      1_538.76       654.69     2_193.45       0.9999          1.0000       116.16
CAGRA-bw48 (query)                                     1_538.76       244.80     1_783.55       0.9999          1.0000       116.16
CAGRA-bw48 (self)                                      1_538.76     1_046.45     2_585.21       1.0000          1.0000       116.16
CAGRA-bw64 (query)                                     1_538.76       314.04     1_852.80       1.0000          1.0000       116.16
CAGRA-bw64 (self)                                      1_538.76     1_488.29     3_027.05       1.0000          1.0000       116.16
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
CPU-Exhaustive (query)                                    46.84     8_005.96     8_052.81       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     46.84    85_981.60    86_028.45       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    51.00     1_419.02     1_470.02       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     51.00    12_648.50    12_699.50       1.0000          1.0000        61.04
CAGRA-auto (query)                                     1_479.48       361.77     1_841.25       0.9999          1.0000       132.56
CAGRA-auto (self)                                      1_479.48     1_023.94     2_503.42       0.9999          1.0000       132.56
CAGRA-bw16 (query)                                     1_479.48       314.44     1_793.92       0.9998          1.0000       132.56
CAGRA-bw16 (self)                                      1_479.48       542.13     2_021.62       0.9998          1.0000       132.56
CAGRA-bw30 (query)                                     1_479.48       358.56     1_838.04       0.9999          1.0000       132.56
CAGRA-bw30 (self)                                      1_479.48       960.52     2_440.01       0.9999          1.0000       132.56
CAGRA-bw48 (query)                                     1_479.48       424.87     1_904.35       1.0000          1.0000       132.56
CAGRA-bw48 (self)                                      1_479.48     1_615.80     3_095.29       1.0000          1.0000       132.56
CAGRA-bw64 (query)                                     1_479.48       496.88     1_976.36       1.0000          1.0000       132.56
CAGRA-bw64 (self)                                      1_479.48     2_343.00     3_822.48       1.0000          1.0000       132.56
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
CPU-Exhaustive (query)                                    91.53    17_988.43    18_079.96       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     91.53   182_046.42   182_137.95       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   101.98     2_232.40     2_334.38       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    101.98    20_794.22    20_896.20       1.0000          1.0000       122.07
CAGRA-auto (query)                                     3_117.64       534.40     3_652.04       0.9999          1.0000       193.60
CAGRA-auto (self)                                      3_117.64     1_140.39     4_258.03       0.9999          1.0000       193.60
CAGRA-bw16 (query)                                     3_117.64       459.83     3_577.47       0.9998          1.0000       193.60
CAGRA-bw16 (self)                                      3_117.64       633.61     3_751.25       0.9998          1.0000       193.60
CAGRA-bw30 (query)                                     3_117.64       527.16     3_644.80       0.9999          1.0000       193.60
CAGRA-bw30 (self)                                      3_117.64     1_093.08     4_210.72       0.9999          1.0000       193.60
CAGRA-bw48 (query)                                     3_117.64       571.39     3_689.03       1.0000          1.0000       193.60
CAGRA-bw48 (self)                                      3_117.64     1_744.59     4_862.24       1.0000          1.0000       193.60
CAGRA-bw64 (query)                                     3_117.64       665.59     3_783.23       1.0000          1.0000       193.60
CAGRA-bw64 (self)                                      3_117.64     2_478.08     5_595.72       1.0000          1.0000       193.60
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
CPU-Exhaustive (query)                                   125.72    36_450.76    36_576.48       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                    125.72   376_953.62   377_079.34       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                   136.74     5_217.06     5_353.80       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                    136.74    50_477.51    50_614.25       1.0000          1.0000       122.07
CAGRA-auto (query)                                     3_109.43       699.74     3_809.16       0.9999          1.0000       265.12
CAGRA-auto (self)                                      3_109.43     2_069.79     5_179.22       0.9999          1.0000       265.12
CAGRA-bw16 (query)                                     3_109.43       587.38     3_696.81       0.9998          1.0000       265.12
CAGRA-bw16 (self)                                      3_109.43     1_117.06     4_226.49       0.9998          1.0000       265.12
CAGRA-bw30 (query)                                     3_109.43       702.68     3_812.11       0.9999          1.0000       265.12
CAGRA-bw30 (self)                                      3_109.43     1_939.66     5_049.09       0.9999          1.0000       265.12
CAGRA-bw48 (query)                                     3_109.43       892.76     4_002.19       1.0000          1.0000       265.12
CAGRA-bw48 (self)                                      3_109.43     3_545.14     6_654.57       1.0000          1.0000       265.12
CAGRA-bw64 (query)                                     3_109.43     1_013.52     4_122.94       1.0000          1.0000       265.12
CAGRA-bw64 (self)                                      3_109.43     4_860.96     7_970.39       1.0000          1.0000       265.12
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
CPU-Exhaustive (query)                                   258.34    91_282.97    91_541.31       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                    258.34   878_309.34   878_567.68       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                   279.83     8_545.74     8_825.57       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                    279.83    83_810.96    84_090.79       1.0000          1.0000       244.14
CAGRA-auto (query)                                     6_457.69     1_053.03     7_510.72       0.9999          1.0000       387.19
CAGRA-auto (self)                                      6_457.69     2_346.03     8_803.72       0.9999          1.0000       387.19
CAGRA-bw16 (query)                                     6_457.69       940.84     7_398.52       0.9998          1.0000       387.19
CAGRA-bw16 (self)                                      6_457.69     1_291.42     7_749.10       0.9998          1.0000       387.19
CAGRA-bw30 (query)                                     6_457.69     1_038.21     7_495.90       0.9999          1.0000       387.19
CAGRA-bw30 (self)                                      6_457.69     2_204.79     8_662.47       0.9998          1.0000       387.19
CAGRA-bw48 (query)                                     6_457.69     1_193.10     7_650.78       0.9999          1.0000       387.19
CAGRA-bw48 (self)                                      6_457.69     3_519.73     9_977.42       0.9999          1.0000       387.19
CAGRA-bw64 (query)                                     6_457.69     1_304.01     7_761.70       0.9999          1.0000       387.19
CAGRA-bw64 (self)                                      6_457.69     4_976.46    11_434.15       0.9999          1.0000       387.19
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
GPU-Exhaustive (ground truth)                             25.41     8_898.88     8_924.29       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   3_718.16     1_193.12     4_911.28       1.0000          1.0000       276.93
GPU-NND bk=1x refine=0 (extract)                         621.62        43.52       665.14       0.8498          1.0924       102.04
GPU-NND bk=1x refine=0 (self-beam)                       621.62     1_069.62     1_691.24       0.9833          1.0014       102.04
GPU-NND bk=1x refine=1 (extract)                         508.52        43.40       551.92       0.9019          1.0852       102.04
GPU-NND bk=1x refine=1 (self-beam)                       508.52     1_062.68     1_571.19       0.9855          1.0011       102.04
GPU-NND bk=1x refine=2 (extract)                         513.24        40.16       553.41       0.9065          1.0846       102.04
GPU-NND bk=1x refine=2 (self-beam)                       513.24     1_071.44     1_584.68       0.9860          1.0011       102.04
GPU-NND bk=2x refine=0 (extract)                         857.38        40.12       897.51       0.9124          1.0840       102.04
GPU-NND bk=2x refine=0 (self-beam)                       857.38     1_056.11     1_913.50       0.9928          1.0004       102.04
GPU-NND bk=2x refine=1 (extract)                         956.72        40.55       997.27       0.9303          1.0821       102.04
GPU-NND bk=2x refine=1 (self-beam)                       956.72     1_053.67     2_010.39       0.9950          1.0002       102.04
GPU-NND bk=2x refine=2 (extract)                       1_038.93        40.06     1_078.99       0.9309          1.0821       102.04
GPU-NND bk=2x refine=2 (self-beam)                     1_038.93     1_059.13     2_098.06       0.9952          1.0002       102.04
GPU-NND bk=3x refine=0 (extract)                       1_343.08        41.00     1_384.08       0.9260          1.0825       102.04
GPU-NND bk=3x refine=0 (self-beam)                     1_343.08     1_053.91     2_396.99       0.9951          1.0002       102.04
GPU-NND bk=3x refine=1 (extract)                       1_474.57        41.56     1_516.13       0.9329          1.0819       102.04
GPU-NND bk=3x refine=1 (self-beam)                     1_474.57     1_062.43     2_537.00       0.9962          1.0001       102.04
GPU-NND bk=3x refine=2 (extract)                       1_674.09        41.38     1_715.48       0.9330          1.0819       102.04
GPU-NND bk=3x refine=2 (self-beam)                     1_674.09     1_052.92     2_727.01       0.9962          1.0001       102.04
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
GPU-Exhaustive (ground truth)                             57.27    12_734.54    12_791.82       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   4_957.55     1_734.92     6_692.47       1.0000          1.0000       365.97
GPU-NND bk=1x refine=0 (extract)                         900.03        46.59       946.62       0.8488          1.0924       132.56
GPU-NND bk=1x refine=0 (self-beam)                       900.03     1_152.65     2_052.68       0.9832          1.0014       132.56
GPU-NND bk=1x refine=1 (extract)                         932.59        43.80       976.39       0.9014          1.0851       132.56
GPU-NND bk=1x refine=1 (self-beam)                       932.59     1_134.18     2_066.78       0.9855          1.0011       132.56
GPU-NND bk=1x refine=2 (extract)                       1_091.41        41.39     1_132.80       0.9061          1.0846       132.56
GPU-NND bk=1x refine=2 (self-beam)                     1_091.41     1_134.14     2_225.55       0.9858          1.0011       132.56
GPU-NND bk=2x refine=0 (extract)                       1_424.33        41.15     1_465.48       0.9124          1.0839       132.56
GPU-NND bk=2x refine=0 (self-beam)                     1_424.33     1_138.23     2_562.57       0.9928          1.0004       132.56
GPU-NND bk=2x refine=1 (extract)                       2_209.69        40.62     2_250.31       0.9303          1.0820       132.56
GPU-NND bk=2x refine=1 (self-beam)                     2_209.69     1_127.28     3_336.96       0.9949          1.0002       132.56
GPU-NND bk=2x refine=2 (extract)                       3_021.67        42.69     3_064.36       0.9309          1.0820       132.56
GPU-NND bk=2x refine=2 (self-beam)                     3_021.67     1_138.21     4_159.88       0.9951          1.0002       132.56
GPU-NND bk=3x refine=0 (extract)                       2_518.38        41.98     2_560.36       0.9259          1.0824       132.56
GPU-NND bk=3x refine=0 (self-beam)                     2_518.38     1_129.09     3_647.47       0.9951          1.0002       132.56
GPU-NND bk=3x refine=1 (extract)                       4_499.77        44.40     4_544.17       0.9329          1.0818       132.56
GPU-NND bk=3x refine=1 (self-beam)                     4_499.77     1_152.09     5_651.87       0.9962          1.0001       132.56
GPU-NND bk=3x refine=2 (extract)                       6_291.09        42.76     6_333.85       0.9330          1.0818       132.56
GPU-NND bk=3x refine=2 (self-beam)                     6_291.09     1_162.44     7_453.53       0.9962          1.0001       132.56
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
GPU-Exhaustive (ground truth)                             60.36    34_793.09    34_853.46       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   9_572.82     2_945.45    12_518.27       0.9999          1.0000       631.89
GPU-NND bk=1x refine=0 (extract)                       1_150.94        88.87     1_239.81       0.8134          1.0976       204.09
GPU-NND bk=1x refine=0 (self-beam)                     1_150.94     2_189.12     3_340.06       0.9711          1.0025       204.09
GPU-NND bk=1x refine=1 (extract)                       1_148.21        89.46     1_237.68       0.8817          1.0874       204.09
GPU-NND bk=1x refine=1 (self-beam)                     1_148.21     2_177.57     3_325.78       0.9752          1.0021       204.09
GPU-NND bk=1x refine=2 (extract)                       1_307.12        81.65     1_388.78       0.8902          1.0863       204.09
GPU-NND bk=1x refine=2 (self-beam)                     1_307.12     2_181.50     3_488.62       0.9761          1.0020       204.09
GPU-NND bk=2x refine=0 (extract)                       1_822.96        80.67     1_903.63       0.9014          1.0850       204.09
GPU-NND bk=2x refine=0 (self-beam)                     1_822.96     2_154.76     3_977.72       0.9882          1.0007       204.09
GPU-NND bk=2x refine=1 (extract)                       2_606.07        80.64     2_686.71       0.9277          1.0821       204.09
GPU-NND bk=2x refine=1 (self-beam)                     2_606.07     2_139.70     4_745.77       0.9919          1.0004       204.09
GPU-NND bk=2x refine=2 (extract)                       3_372.30        80.76     3_453.07       0.9289          1.0820       204.09
GPU-NND bk=2x refine=2 (self-beam)                     3_372.30     2_133.16     5_505.47       0.9923          1.0004       204.09
GPU-NND bk=3x refine=0 (extract)                       2_707.10        80.66     2_787.77       0.9224          1.0826       204.09
GPU-NND bk=3x refine=0 (self-beam)                     2_707.10     2_158.55     4_865.66       0.9925          1.0003       204.09
GPU-NND bk=3x refine=1 (extract)                       4_530.24        80.96     4_611.21       0.9325          1.0817       204.09
GPU-NND bk=3x refine=1 (self-beam)                     4_530.24     2_151.29     6_681.53       0.9944          1.0002       204.09
GPU-NND bk=3x refine=2 (extract)                       6_350.43        81.38     6_431.81       0.9327          1.0816       204.09
GPU-NND bk=3x refine=2 (self-beam)                     6_350.43     2_152.40     8_502.83       0.9945          1.0002       204.09
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
GPU-Exhaustive (ground truth)                            139.64    50_603.56    50_743.20       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  12_944.63     4_414.87    17_359.50       0.9999          1.0000       803.96
GPU-NND bk=1x refine=0 (extract)                       1_598.42        88.21     1_686.63       0.8121          1.0977       265.12
GPU-NND bk=1x refine=0 (self-beam)                     1_598.42     2_489.95     4_088.37       0.9708          1.0025       265.12
GPU-NND bk=1x refine=1 (extract)                       2_163.90        86.62     2_250.52       0.8811          1.0873       265.12
GPU-NND bk=1x refine=1 (self-beam)                     2_163.90     2_307.12     4_471.02       0.9749          1.0021       265.12
GPU-NND bk=1x refine=2 (extract)                       2_809.80        81.03     2_890.82       0.8896          1.0862       265.12
GPU-NND bk=1x refine=2 (self-beam)                     2_809.80     2_302.35     5_112.15       0.9758          1.0020       265.12
GPU-NND bk=2x refine=0 (extract)                       2_848.20        80.63     2_928.83       0.9011          1.0849       265.12
GPU-NND bk=2x refine=0 (self-beam)                     2_848.20     2_322.15     5_170.35       0.9880          1.0008       265.12
GPU-NND bk=2x refine=1 (extract)                       5_853.50        82.83     5_936.33       0.9276          1.0820       265.12
GPU-NND bk=2x refine=1 (self-beam)                     5_853.50     2_302.93     8_156.43       0.9919          1.0004       265.12
GPU-NND bk=2x refine=2 (extract)                       8_959.17        80.94     9_040.11       0.9288          1.0819       265.12
GPU-NND bk=2x refine=2 (self-beam)                     8_959.17     2_300.00    11_259.17       0.9923          1.0004       265.12
GPU-NND bk=3x refine=0 (extract)                       5_429.05        80.75     5_509.80       0.9224          1.0825       265.12
GPU-NND bk=3x refine=0 (self-beam)                     5_429.05     2_296.32     7_725.37       0.9925          1.0003       265.12
GPU-NND bk=3x refine=1 (extract)                      12_473.27        80.91    12_554.18       0.9325          1.0816       265.12
GPU-NND bk=3x refine=1 (self-beam)                    12_473.27     2_302.50    14_775.76       0.9944          1.0002       265.12
GPU-NND bk=3x refine=2 (extract)                      19_455.99        84.23    19_540.21       0.9327          1.0816       265.12
GPU-NND bk=3x refine=2 (self-beam)                    19_455.99     2_315.61    21_771.59       0.9945          1.0002       265.12
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
GPU-Exhaustive (ground truth)                            124.89   139_138.17   139_263.06       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  20_464.59     7_176.47    27_641.06       0.9999          1.0000      1295.77
GPU-NND bk=1x refine=0 (extract)                       2_186.88       176.78     2_363.66       0.7743          1.1040       408.17
GPU-NND bk=1x refine=0 (self-beam)                     2_186.88     4_466.04     6_652.91       0.9553          1.0041       408.17
GPU-NND bk=1x refine=1 (extract)                       2_635.50       185.67     2_821.17       0.8571          1.0904       408.17
GPU-NND bk=1x refine=1 (self-beam)                     2_635.50     4_427.64     7_063.15       0.9620          1.0034       408.17
GPU-NND bk=1x refine=2 (extract)                       3_297.19       177.80     3_474.99       0.8703          1.0886       408.17
GPU-NND bk=1x refine=2 (self-beam)                     3_297.19     4_456.36     7_753.55       0.9636          1.0032       408.17
GPU-NND bk=2x refine=0 (extract)                       3_747.26       176.73     3_924.00       0.8886          1.0863       408.17
GPU-NND bk=2x refine=0 (self-beam)                     3_747.26     4_372.44     8_119.70       0.9822          1.0012       408.17
GPU-NND bk=2x refine=1 (extract)                       6_662.78       173.50     6_836.28       0.9239          1.0823       408.17
GPU-NND bk=2x refine=1 (self-beam)                     6_662.78     4_359.50    11_022.29       0.9881          1.0006       408.17
GPU-NND bk=2x refine=2 (extract)                       9_435.06       186.38     9_621.44       0.9261          1.0821       408.17
GPU-NND bk=2x refine=2 (self-beam)                     9_435.06     4_364.03    13_799.09       0.9889          1.0006       408.17
GPU-NND bk=3x refine=0 (extract)                       5_816.95       175.95     5_992.90       0.9184          1.0828       408.17
GPU-NND bk=3x refine=0 (self-beam)                     5_816.95     4_364.07    10_181.02       0.9894          1.0005       408.17
GPU-NND bk=3x refine=1 (extract)                      12_032.94       170.89    12_203.83       0.9318          1.0815       408.17
GPU-NND bk=3x refine=1 (self-beam)                    12_032.94     4_343.76    16_376.71       0.9923          1.0003       408.17
GPU-NND bk=3x refine=2 (extract)                      18_767.25       187.16    18_954.40       0.9322          1.0815       408.17
GPU-NND bk=3x refine=2 (self-beam)                    18_767.25     4_360.63    23_127.88       0.9925          1.0003       408.17
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
GPU-Exhaustive (ground truth)                            315.24   205_198.49   205_513.73       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  27_773.31    10_206.38    37_979.69       0.9999          1.0000      1487.90
GPU-NND bk=1x refine=0 (extract)                       3_286.13       182.50     3_468.63       0.7746          1.1036       530.24
GPU-NND bk=1x refine=0 (self-beam)                     3_286.13     4_851.20     8_137.33       0.9553          1.0041       530.24
GPU-NND bk=1x refine=1 (extract)                       4_856.49       176.90     5_033.39       0.8572          1.0902       530.24
GPU-NND bk=1x refine=1 (self-beam)                     4_856.49     4_737.91     9_594.40       0.9620          1.0033       530.24
GPU-NND bk=1x refine=2 (extract)                       6_741.75       175.96     6_917.71       0.8703          1.0884       530.24
GPU-NND bk=1x refine=2 (self-beam)                     6_741.75     4_783.44    11_525.19       0.9636          1.0031       530.24
GPU-NND bk=2x refine=0 (extract)                       6_028.77       175.06     6_203.84       0.8887          1.0862       530.24
GPU-NND bk=2x refine=0 (self-beam)                     6_028.77     4_727.45    10_756.22       0.9822          1.0012       530.24
GPU-NND bk=2x refine=1 (extract)                      14_728.41       169.96    14_898.37       0.9239          1.0822       530.24
GPU-NND bk=2x refine=1 (self-beam)                    14_728.41     4_670.08    19_398.49       0.9881          1.0006       530.24
GPU-NND bk=2x refine=2 (extract)                      23_103.49       182.52    23_286.01       0.9261          1.0820       530.24
GPU-NND bk=2x refine=2 (self-beam)                    23_103.49     4_746.83    27_850.32       0.9888          1.0006       530.24
GPU-NND bk=3x refine=0 (extract)                      11_968.70       175.17    12_143.87       0.9184          1.0827       530.24
GPU-NND bk=3x refine=0 (self-beam)                    11_968.70     4_773.24    16_741.94       0.9894          1.0005       530.24
GPU-NND bk=3x refine=1 (extract)                      31_146.56       180.47    31_327.03       0.9318          1.0814       530.24
GPU-NND bk=3x refine=1 (self-beam)                    31_146.56     4_762.90    35_909.46       0.9923          1.0003       530.24
GPU-NND bk=3x refine=2 (extract)                      50_421.31       164.77    50_586.09       0.9322          1.0814       530.24
GPU-NND bk=3x refine=2 (self-beam)                    50_421.31     4_817.24    55_238.55       0.9924          1.0003       530.24
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
GPU-Exhaustive (ground truth)                            290.71   844_557.55   844_848.25       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  62_051.51    21_874.43    83_925.94       0.9997          1.0000      3487.41
GPU-NND bk=1x refine=0 (extract)                       6_003.24       461.17     6_464.42       0.6935          1.1199      1020.43
GPU-NND bk=1x refine=0 (self-beam)                     6_003.24    11_777.13    17_780.37       0.9193          1.0083      1020.43
GPU-NND bk=1x refine=1 (extract)                       7_754.56       472.30     8_226.86       0.8013          1.0985      1020.43
GPU-NND bk=1x refine=1 (self-beam)                     7_754.56    11_551.44    19_306.00       0.9323          1.0067      1020.43
GPU-NND bk=1x refine=2 (extract)                      10_115.09       441.92    10_557.01       0.8260          1.0946      1020.43
GPU-NND bk=1x refine=2 (self-beam)                    10_115.09    11_544.72    21_659.81       0.9362          1.0062      1020.43
GPU-NND bk=2x refine=0 (extract)                      10_282.31       440.22    10_722.52       0.8648          1.0892      1020.43
GPU-NND bk=2x refine=0 (self-beam)                    10_282.31    11_381.02    21_663.33       0.9712          1.0022      1020.43
GPU-NND bk=2x refine=1 (extract)                      20_348.17       433.45    20_781.63       0.9155          1.0829      1020.43
GPU-NND bk=2x refine=1 (self-beam)                    20_348.17    11_351.91    31_700.08       0.9812          1.0012      1020.43
GPU-NND bk=2x refine=2 (extract)                      30_862.03       433.82    31_295.86       0.9201          1.0824      1020.43
GPU-NND bk=2x refine=2 (self-beam)                    30_862.03    11_229.19    42_091.22       0.9829          1.0010      1020.43
GPU-NND bk=3x refine=0 (extract)                      16_370.49       435.02    16_805.51       0.9108          1.0834      1020.43
GPU-NND bk=3x refine=0 (self-beam)                    16_370.49    11_303.95    27_674.43       0.9844          1.0008      1020.43
GPU-NND bk=3x refine=1 (extract)                      36_872.90       434.11    37_307.01       0.9303          1.0814      1020.43
GPU-NND bk=3x refine=1 (self-beam)                    36_872.90    11_642.24    48_515.14       0.9892          1.0004      1020.43
GPU-NND bk=3x refine=2 (extract)                      56_346.77       412.28    56_759.06       0.9311          1.0813      1020.43
GPU-NND bk=3x refine=2 (self-beam)                    56_346.77    11_568.94    67_915.71       0.9896          1.0004      1020.43
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

### Navigating Spread-out Graph (NSG) with GPU-accelerated kNN generation

NSG needs an initial kNN graph to run the NSG optimisation on top. The standard
CPU version is using NNDescent; however, you can also use the NNDescent
algorithm that is used for CAGRA to initialise the kNN. This does make a
difference on larger and larger data sets with higher and higher
dimensionality.

<details>
<summary><b>NSG with CPU initialisation</b>:</summary>
<pre><code>
===================================================================================================================================
Benchmark: 250k samples, 128D
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       107.62    11_747.72    11_855.34       1.0000          1.0000       122.07
Exhaustive (self)                                        107.62   203_033.67   203_141.29       1.0000          1.0000       122.07
NSG-R24-L50-ef50 (query)                              11_053.42       179.38    11_232.80       0.9984          1.0060       144.96
NSG-R24-L50-efauto (query)                            11_053.42       294.62    11_348.05       0.9994          1.0025       144.96
NSG-R24-L50-ef150 (query)                             11_053.42       393.32    11_446.75       0.9998          1.0008       144.96
NSG-R24-L50 (self)                                    11_053.42     2_850.61    13_904.04       1.0000          1.0000       144.96
NSG-R24-L100-ef50 (query)                             12_946.36       205.68    13_152.03       0.9992          1.0028       144.96
NSG-R24-L100-efauto (query)                           12_946.36       311.77    13_258.12       0.9999          1.0003       144.96
NSG-R24-L100-ef150 (query)                            12_946.36       421.32    13_367.67       1.0000          1.0000       144.96
NSG-R24-L100 (self)                                   12_946.36     3_113.63    16_059.99       1.0000          1.0000       144.96
NSG-R24-L150-ef50 (query)                             15_450.21       196.06    15_646.27       0.9992          1.0028       144.96
NSG-R24-L150-efauto (query)                           15_450.21       294.21    15_744.42       0.9999          1.0003       144.96
NSG-R24-L150-ef150 (query)                            15_450.21       419.38    15_869.59       1.0000          1.0000       144.96
NSG-R24-L150 (self)                                   15_450.21     3_056.07    18_506.28       1.0000          1.0000       144.96
NSG-R32-L50-ef50 (query)                              11_212.85       206.50    11_419.35       0.9985          1.0057       152.59
NSG-R32-L50-efauto (query)                            11_212.85       294.24    11_507.09       0.9995          1.0021       152.59
NSG-R32-L50-ef150 (query)                             11_212.85       379.85    11_592.69       0.9999          1.0006       152.59
NSG-R32-L50 (self)                                    11_212.85     2_967.17    14_180.01       1.0000          1.0000       152.59
NSG-R32-L100-ef50 (query)                             12_192.04       186.95    12_378.99       0.9991          1.0031       152.59
NSG-R32-L100-efauto (query)                           12_192.04       284.76    12_476.80       0.9999          1.0003       152.59
NSG-R32-L100-ef150 (query)                            12_192.04       379.53    12_571.57       1.0000          1.0000       152.59
NSG-R32-L100 (self)                                   12_192.04     2_980.71    15_172.74       1.0000          1.0000       152.59
NSG-R32-L150-ef50 (query)                             13_530.31       186.66    13_716.97       0.9991          1.0031       152.59
NSG-R32-L150-efauto (query)                           13_530.31       283.13    13_813.44       0.9999          1.0003       152.59
NSG-R32-L150-ef150 (query)                            13_530.31       380.34    13_910.65       1.0000          1.0000       152.59
NSG-R32-L150 (self)                                   13_530.31     2_969.92    16_500.23       1.0000          1.0000       152.59
NSG-R48-L50-ef50 (query)                              10_384.56       192.31    10_576.87       0.9986          1.0054       167.85
NSG-R48-L50-efauto (query)                            10_384.56       289.20    10_673.75       0.9995          1.0021       167.85
NSG-R48-L50-ef150 (query)                             10_384.56       386.83    10_771.38       0.9999          1.0006       167.85
NSG-R48-L50 (self)                                    10_384.56     3_046.40    13_430.96       1.0000          1.0000       167.85
NSG-R48-L100-ef50 (query)                             12_263.60       191.14    12_454.74       0.9991          1.0031       167.85
NSG-R48-L100-efauto (query)                           12_263.60       291.10    12_554.69       0.9999          1.0003       167.85
NSG-R48-L100-ef150 (query)                            12_263.60       389.07    12_652.66       1.0000          1.0000       167.85
NSG-R48-L100 (self)                                   12_263.60     3_064.64    15_328.23       1.0000          1.0000       167.85
NSG-R48-L150-ef50 (query)                             13_616.83       192.02    13_808.85       0.9991          1.0031       167.85
NSG-R48-L150-efauto (query)                           13_616.83       295.34    13_912.17       0.9999          1.0003       167.85
NSG-R48-L150-ef150 (query)                            13_616.83       393.94    14_010.76       1.0000          1.0000       167.85
NSG-R48-L150 (self)                                   13_616.83     3_312.45    16_929.28       1.0000          1.0000       167.85
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
Exhaustive (query)                                       107.90    11_058.69    11_166.59       1.0000          1.0000       122.07
Exhaustive (self)                                        107.90   188_639.65   188_747.55       1.0000          1.0000       122.07
NSG-GPU-R24-L50-ef50 (query)                           6_400.22       177.79     6_578.01       0.9979          1.0077       144.96
NSG-GPU-R24-L50-efauto (query)                         6_400.22       272.14     6_672.36       0.9994          1.0025       144.96
NSG-GPU-R24-L50-ef150 (query)                          6_400.22       353.54     6_753.76       0.9998          1.0008       144.96
NSG-GPU-R24-L50 (self)                                 6_400.22     2_731.84     9_132.06       1.0000          1.0000       144.96
NSG-GPU-R24-L100-ef50 (query)                          8_337.84       176.22     8_514.06       0.9987          1.0041       144.96
NSG-GPU-R24-L100-efauto (query)                        8_337.84       265.54     8_603.38       0.9999          1.0004       144.96
NSG-GPU-R24-L100-ef150 (query)                         8_337.84       354.50     8_692.34       0.9999          1.0001       144.96
NSG-GPU-R24-L100 (self)                                8_337.84     2_762.47    11_100.31       1.0000          1.0000       144.96
NSG-GPU-R24-L150-ef50 (query)                          9_755.45       176.16     9_931.62       0.9987          1.0041       144.96
NSG-GPU-R24-L150-efauto (query)                        9_755.45       265.93    10_021.38       0.9999          1.0004       144.96
NSG-GPU-R24-L150-ef150 (query)                         9_755.45       355.78    10_111.23       0.9999          1.0001       144.96
NSG-GPU-R24-L150 (self)                                9_755.45     2_781.57    12_537.03       1.0000          1.0000       144.96
NSG-GPU-R32-L50-ef50 (query)                           6_501.55       190.62     6_692.17       0.9982          1.0067       152.59
NSG-GPU-R32-L50-efauto (query)                         6_501.55       286.11     6_787.66       0.9995          1.0021       152.59
NSG-GPU-R32-L50-ef150 (query)                          6_501.55       380.30     6_881.86       0.9999          1.0006       152.59
NSG-GPU-R32-L50 (self)                                 6_501.55     3_018.32     9_519.87       1.0000          1.0000       152.59
NSG-GPU-R32-L100-ef50 (query)                          8_332.98       192.75     8_525.73       0.9988          1.0040       152.59
NSG-GPU-R32-L100-efauto (query)                        8_332.98       292.75     8_625.74       0.9999          1.0004       152.59
NSG-GPU-R32-L100-ef150 (query)                         8_332.98       380.69     8_713.67       0.9999          1.0001       152.59
NSG-GPU-R32-L100 (self)                                8_332.98     3_031.54    11_364.52       1.0000          1.0000       152.59
NSG-GPU-R32-L150-ef50 (query)                          9_753.86       190.77     9_944.63       0.9988          1.0040       152.59
NSG-GPU-R32-L150-efauto (query)                        9_753.86       286.60    10_040.46       0.9999          1.0004       152.59
NSG-GPU-R32-L150-ef150 (query)                         9_753.86       382.07    10_135.93       0.9999          1.0001       152.59
NSG-GPU-R32-L150 (self)                                9_753.86     3_045.47    12_799.33       1.0000          1.0000       152.59
NSG-GPU-R48-L50-ef50 (query)                           6_509.42       197.98     6_707.41       0.9983          1.0065       167.85
NSG-GPU-R48-L50-efauto (query)                         6_509.42       291.29     6_800.71       0.9995          1.0021       167.85
NSG-GPU-R48-L50-ef150 (query)                          6_509.42       389.06     6_898.48       0.9999          1.0006       167.85
NSG-GPU-R48-L50 (self)                                 6_509.42     3_097.00     9_606.42       1.0000          1.0000       167.85
NSG-GPU-R48-L100-ef50 (query)                          8_393.65       193.73     8_587.38       0.9988          1.0040       167.85
NSG-GPU-R48-L100-efauto (query)                        8_393.65       294.99     8_688.64       0.9999          1.0004       167.85
NSG-GPU-R48-L100-ef150 (query)                         8_393.65       389.84     8_783.49       0.9999          1.0001       167.85
NSG-GPU-R48-L100 (self)                                8_393.65     3_134.55    11_528.20       1.0000          1.0000       167.85
NSG-GPU-R48-L150-ef50 (query)                          9_829.22       195.61    10_024.84       0.9988          1.0040       167.85
NSG-GPU-R48-L150-efauto (query)                        9_829.22       293.72    10_122.95       0.9999          1.0004       167.85
NSG-GPU-R48-L150-ef150 (query)                         9_829.22       391.46    10_220.68       0.9999          1.0001       167.85
NSG-GPU-R48-L150 (self)                                9_829.22     3_141.30    12_970.53       1.0000          1.0000       167.85
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
Exhaustive (query)                                       252.35    24_482.68    24_735.04       1.0000          1.0000       244.14
Exhaustive (self)                                        252.35   833_484.97   833_737.33       1.0000          1.0000       244.14
NSG-R24-L50-ef50 (query)                              26_593.83       202.89    26_796.71       0.9960          1.0203       289.92
NSG-R24-L50-efauto (query)                            26_593.83       312.01    26_905.84       0.9987          1.0066       289.92
NSG-R24-L50-ef150 (query)                             26_593.83       401.26    26_995.09       0.9994          1.0031       289.92
NSG-R24-L50 (self)                                    26_593.83     6_158.27    32_752.10       1.0000          1.0000       289.92
NSG-R24-L100-ef50 (query)                             31_052.98       203.57    31_256.55       0.9963          1.0186       289.92
NSG-R24-L100-efauto (query)                           31_052.98       323.29    31_376.27       0.9989          1.0056       289.92
NSG-R24-L100-ef150 (query)                            31_052.98       404.51    31_457.49       0.9997          1.0019       289.92
NSG-R24-L100 (self)                                   31_052.98     6_208.61    37_261.59       1.0000          1.0000       289.92
NSG-R24-L150-ef50 (query)                             35_002.35       205.66    35_208.01       0.9983          1.0082       289.92
NSG-R24-L150-efauto (query)                           35_002.35       299.81    35_302.17       0.9993          1.0035       289.92
NSG-R24-L150-ef150 (query)                            35_002.35       396.77    35_399.12       0.9995          1.0029       289.92
NSG-R24-L150 (self)                                   35_002.35     6_166.73    41_169.08       1.0000          1.0000       289.92
NSG-R32-L50-ef50 (query)                              26_312.16       217.49    26_529.65       0.9965          1.0179       305.18
NSG-R32-L50-efauto (query)                            26_312.16       329.13    26_641.29       0.9987          1.0066       305.18
NSG-R32-L50-ef150 (query)                             26_312.16       430.96    26_743.12       0.9994          1.0031       305.18
NSG-R32-L50 (self)                                    26_312.16     6_798.43    33_110.59       1.0000          1.0000       305.18
NSG-R32-L100-ef50 (query)                             31_108.59       218.30    31_326.90       0.9964          1.0178       305.18
NSG-R32-L100-efauto (query)                           31_108.59       339.38    31_447.97       0.9989          1.0056       305.18
NSG-R32-L100-ef150 (query)                            31_108.59       445.06    31_553.65       0.9997          1.0019       305.18
NSG-R32-L100 (self)                                   31_108.59     6_838.46    37_947.05       1.0000          1.0000       305.18
NSG-R32-L150-ef50 (query)                             34_557.78       216.42    34_774.20       0.9985          1.0071       305.18
NSG-R32-L150-efauto (query)                           34_557.78       323.34    34_881.12       0.9993          1.0035       305.18
NSG-R32-L150-ef150 (query)                            34_557.78       432.59    34_990.38       0.9995          1.0029       305.18
NSG-R32-L150 (self)                                   34_557.78     6_787.75    41_345.53       1.0000          1.0000       305.18
NSG-R48-L50-ef50 (query)                              26_491.98       220.68    26_712.66       0.9965          1.0179       335.69
NSG-R48-L50-efauto (query)                            26_491.98       333.22    26_825.20       0.9987          1.0066       335.69
NSG-R48-L50-ef150 (query)                             26_491.98       439.31    26_931.30       0.9994          1.0031       335.69
NSG-R48-L50 (self)                                    26_491.98     6_998.12    33_490.10       1.0000          1.0000       335.69
NSG-R48-L100-ef50 (query)                             30_964.68       223.63    31_188.31       0.9964          1.0178       335.69
NSG-R48-L100-efauto (query)                           30_964.68       345.01    31_309.69       0.9989          1.0056       335.69
NSG-R48-L100-ef150 (query)                            30_964.68       450.52    31_415.20       0.9997          1.0019       335.69
NSG-R48-L100 (self)                                   30_964.68     7_071.35    38_036.03       1.0000          1.0000       335.69
NSG-R48-L150-ef50 (query)                             34_883.34       231.10    35_114.44       0.9985          1.0069       335.69
NSG-R48-L150-efauto (query)                           34_883.34       340.38    35_223.72       0.9993          1.0035       335.69
NSG-R48-L150-ef150 (query)                            34_883.34       450.72    35_334.06       0.9995          1.0029       335.69
NSG-R48-L150 (self)                                   34_883.34     7_098.95    41_982.29       1.0000          1.0000       335.69
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
Exhaustive (query)                                       260.02    24_656.97    24_916.98       1.0000          1.0000       244.14
Exhaustive (self)                                        260.02   832_406.69   832_666.71       1.0000          1.0000       244.14
NSG-GPU-R24-L50-ef50 (query)                          13_771.63       202.56    13_974.20       0.9956          1.0217       289.92
NSG-GPU-R24-L50-efauto (query)                        13_771.63       312.38    14_084.01       0.9987          1.0061       289.92
NSG-GPU-R24-L50-ef150 (query)                         13_771.63       408.36    14_179.99       0.9995          1.0024       289.92
NSG-GPU-R24-L50 (self)                                13_771.63     6_194.51    19_966.15       1.0000          1.0000       289.92
NSG-GPU-R24-L100-ef50 (query)                         18_014.43       202.81    18_217.24       0.9956          1.0210       289.92
NSG-GPU-R24-L100-efauto (query)                       18_014.43       306.65    18_321.09       0.9989          1.0054       289.92
NSG-GPU-R24-L100-ef150 (query)                        18_014.43       410.02    18_424.45       0.9998          1.0012       289.92
NSG-GPU-R24-L100 (self)                               18_014.43     6_183.14    24_197.57       1.0000          1.0000       289.92
NSG-GPU-R24-L150-ef50 (query)                         22_076.46       205.69    22_282.15       0.9980          1.0091       289.92
NSG-GPU-R24-L150-efauto (query)                       22_076.46       309.17    22_385.63       0.9993          1.0037       289.92
NSG-GPU-R24-L150-ef150 (query)                        22_076.46       416.50    22_492.95       0.9995          1.0029       289.92
NSG-GPU-R24-L150 (self)                               22_076.46     6_243.07    28_319.53       1.0000          1.0000       289.92
NSG-GPU-R32-L50-ef50 (query)                          13_899.61       232.79    14_132.41       0.9961          1.0193       305.18
NSG-GPU-R32-L50-efauto (query)                        13_899.61       330.40    14_230.01       0.9987          1.0061       305.18
NSG-GPU-R32-L50-ef150 (query)                         13_899.61       442.22    14_341.84       0.9995          1.0024       305.18
NSG-GPU-R32-L50 (self)                                13_899.61     6_790.33    20_689.94       1.0000          1.0000       305.18
NSG-GPU-R32-L100-ef50 (query)                         18_117.25       222.38    18_339.63       0.9961          1.0185       305.18
NSG-GPU-R32-L100-efauto (query)                       18_117.25       331.48    18_448.74       0.9989          1.0054       305.18
NSG-GPU-R32-L100-ef150 (query)                        18_117.25       438.86    18_556.11       0.9998          1.0012       305.18
NSG-GPU-R32-L100 (self)                               18_117.25     6_845.59    24_962.84       1.0000          1.0000       305.18
NSG-GPU-R32-L150-ef50 (query)                         21_946.63       226.59    22_173.22       0.9985          1.0073       305.18
NSG-GPU-R32-L150-efauto (query)                       21_946.63       333.52    22_280.15       0.9993          1.0037       305.18
NSG-GPU-R32-L150-ef150 (query)                        21_946.63       436.68    22_383.32       0.9995          1.0029       305.18
NSG-GPU-R32-L150 (self)                               21_946.63     6_828.65    28_775.28       1.0000          1.0000       305.18
NSG-GPU-R48-L50-ef50 (query)                          13_939.06       225.26    14_164.31       0.9962          1.0186       335.69
NSG-GPU-R48-L50-efauto (query)                        13_939.06       344.36    14_283.42       0.9987          1.0061       335.69
NSG-GPU-R48-L50-ef150 (query)                         13_939.06       442.71    14_381.77       0.9995          1.0024       335.69
NSG-GPU-R48-L50 (self)                                13_939.06     7_029.75    20_968.80       1.0000          1.0000       335.69
NSG-GPU-R48-L100-ef50 (query)                         18_183.90       223.21    18_407.11       0.9961          1.0185       335.69
NSG-GPU-R48-L100-efauto (query)                       18_183.90       339.41    18_523.31       0.9989          1.0054       335.69
NSG-GPU-R48-L100-ef150 (query)                        18_183.90       446.89    18_630.79       0.9998          1.0012       335.69
NSG-GPU-R48-L100 (self)                               18_183.90     7_851.04    26_034.94       1.0000          1.0000       335.69
NSG-GPU-R48-L150-ef50 (query)                         23_353.72       227.62    23_581.34       0.9985          1.0067       335.69
NSG-GPU-R48-L150-efauto (query)                       23_353.72       331.50    23_685.22       0.9993          1.0037       335.69
NSG-GPU-R48-L150-ef150 (query)                        23_353.72       444.26    23_797.98       0.9995          1.0029       335.69
NSG-GPU-R48-L150 (self)                               23_353.72     6_929.12    30_282.84       1.0000          1.0000       335.69
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
