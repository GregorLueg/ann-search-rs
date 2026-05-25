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
Exhaustive (query)                                         3.20     1_483.91     1_487.11       1.0000          1.0000        18.31
Exhaustive (self)                                          3.20    15_087.37    15_090.56       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.54       670.16       675.70       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.54     5_497.51     5_503.05       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               393.47       356.31       749.77       0.9972          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               393.47       377.55       771.02       0.9996          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               393.47       444.86       838.33       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     393.47     1_561.20     1_954.66       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               754.37       385.00     1_139.38       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               754.37       420.87     1_175.25       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     754.37     1_383.79     2_138.16       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_483.01       387.30     1_870.30       0.9931          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_483.01       354.59     1_837.60       0.9984          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_483.01       378.60     1_861.61       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_483.01     1_356.67     2_839.68       1.0000          1.0000         1.15
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
Exhaustive (query)                                         4.25     1_607.62     1_611.88       1.0000          1.0000        18.88
Exhaustive (self)                                          4.25    17_201.34    17_205.59       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.44       678.89       685.33       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      6.44     5_702.12     5_708.56       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               395.83       383.58       779.41       0.9977          1.0002         1.15
IVF-GPU-nl273-np16 (query)                               395.83       283.12       678.95       0.9998          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               395.83       413.79       809.62       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     395.83     1_633.04     2_028.87       0.9999          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               803.63       374.08     1_177.70       0.9990          1.0001         1.15
IVF-GPU-nl387-np27 (query)                               803.63       437.84     1_241.47       0.9999          1.0000         1.15
IVF-GPU-nl387 (self)                                     803.63     1_429.54     2_233.17       0.9999          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_501.71       376.46     1_878.18       0.9941          1.0004         1.15
IVF-GPU-nl547-np27 (query)                             1_501.71       350.06     1_851.77       0.9987          1.0001         1.15
IVF-GPU-nl547-np33 (query)                             1_501.71       386.07     1_887.78       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_501.71     1_495.11     2_996.82       0.9999          1.0000         1.15
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
Exhaustive (query)                                         3.24     1_622.84     1_626.08       1.0000          1.0000        18.31
Exhaustive (self)                                          3.24    16_950.84    16_954.08       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.67       664.05       669.71       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.67     5_467.92     5_473.59       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               425.47       296.76       722.23       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               425.47       375.57       801.04       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               425.47       432.26       857.74       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     425.47     1_568.97     1_994.45       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               769.00       370.68     1_139.68       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               769.00       433.44     1_202.44       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     769.00     1_372.75     2_141.75       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_504.99       368.79     1_873.79       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_504.99       348.17     1_853.16       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_504.99       356.04     1_861.03       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_504.99     1_302.60     2_807.59       1.0000          1.0000         1.15
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
Exhaustive (query)                                         3.42     1_600.09     1_603.51       1.0000          1.0000        18.31
Exhaustive (self)                                          3.42    16_736.66    16_740.08       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.94       659.26       665.21       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.94     5_477.41     5_483.36       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               398.19       306.81       705.00       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               398.19       367.38       765.57       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               398.19       422.20       820.39       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     398.19     1_427.62     1_825.81       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               779.82       215.21       995.04       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               779.82       363.94     1_143.77       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     779.82     1_306.15     2_085.98       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_510.55       253.92     1_764.47       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_510.55       314.71     1_825.26       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_510.55       360.21     1_870.76       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_510.55     1_302.00     2_812.55       1.0000          1.0000         1.15
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
Exhaustive (query)                                        15.43     6_238.10     6_253.54       1.0000          1.0000        73.24
Exhaustive (self)                                         15.43    65_824.28    65_839.71       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    24.33     1_450.75     1_475.08       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     24.33    12_678.85    12_703.18       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               751.36       459.61     1_210.97       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               751.36       490.87     1_242.23       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               751.36       605.37     1_356.73       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     751.36     3_400.83     4_152.19       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                             1_320.86       453.30     1_774.16       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                             1_320.86       547.83     1_868.69       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                   1_320.86     3_113.09     4_433.95       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             2_577.05       399.34     2_976.38       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             2_577.05       454.34     3_031.38       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             2_577.05       552.15     3_129.20       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   2_577.05     2_925.25     5_502.30       1.0000          1.0000         1.15
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
Exhaustive (query)                                        11.75     4_990.74     5_002.50       1.0000          1.0000        61.04
Exhaustive (self)                                         11.75    87_628.81    87_640.56       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                 1_203.69       312.24     1_515.92       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                 1_203.69       325.66     1_529.35       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                 1_203.69       446.90     1_650.58       1.0000          1.0000        61.12
IVF-nl353 (self)                                       1_203.69     6_571.63     7_775.32       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 2_360.10       309.81     2_669.91       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 2_360.10       336.04     2_696.14       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 2_360.10       410.92     2_771.02       1.0000          1.0000        61.16
IVF-nl500 (self)                                       2_360.10     5_599.26     7_959.36       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 4_561.94       293.30     4_855.24       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 4_561.94       352.16     4_914.09       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 4_561.94       368.37     4_930.31       1.0000          1.0000        61.21
IVF-nl707 (self)                                       4_561.94     4_835.12     9_397.05       1.0000          1.0000        61.21
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
Exhaustive (query)                                        11.50     5_120.91     5_132.41       1.0000          1.0000        61.04
Exhaustive (self)                                         11.50    86_116.93    86_128.43       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    18.99     1_498.62     1_517.61       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     18.99    21_623.79    21_642.79       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                             1_193.14       498.15     1_691.29       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_193.14       523.77     1_716.91       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_193.14       604.73     1_797.87       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_193.14     4_857.81     6_050.95       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_178.49       569.09     2_747.58       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_178.49       540.28     2_718.77       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_178.49       519.58     2_698.07       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_178.49     4_414.08     6_592.58       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_278.05       524.46     4_802.51       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_278.05       542.52     4_820.57       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_278.05       558.83     4_836.88       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_278.05     4_037.75     8_315.80       1.0000          1.0000         1.91
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
Exhaustive (query)                                        24.72    11_040.72    11_065.44       1.0000          1.0000       122.07
Exhaustive (self)                                         24.72   188_253.23   188_277.95       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                 1_121.89       655.90     1_777.80       0.9999          1.0000       122.25
IVF-nl353-np18 (query)                                 1_121.89       675.91     1_797.80       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                 1_121.89       933.81     2_055.71       1.0000          1.0000       122.25
IVF-nl353 (self)                                       1_121.89    15_090.58    16_212.48       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 2_217.77       640.83     2_858.59       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 2_217.77       695.61     2_913.37       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 2_217.77       836.19     3_053.95       1.0000          1.0000       122.32
IVF-nl500 (self)                                       2_217.77    13_393.94    15_611.70       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 4_446.36       585.86     5_032.22       1.0000          1.0000       122.42
IVF-nl707-np35 (query)                                 4_446.36       736.84     5_183.21       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 4_446.36       751.82     5_198.19       1.0000          1.0000       122.42
IVF-nl707 (self)                                       4_446.36    11_954.88    16_401.25       1.0000          1.0000       122.42
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
Exhaustive (query)                                        24.86    10_801.24    10_826.10       1.0000          1.0000       122.07
Exhaustive (self)                                         24.86   188_280.42   188_305.28       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    37.44     2_311.42     2_348.86       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     37.44    34_938.77    34_976.20       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                             1_196.61       633.49     1_830.10       0.9999          1.0000         1.91
IVF-GPU-nl353-np18 (query)                             1_196.61       437.42     1_634.03       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                             1_196.61       731.63     1_928.24       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                   1_196.61     8_244.77     9_441.38       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             2_262.74       751.03     3_013.77       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             2_262.74       651.62     2_914.36       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             2_262.74       708.11     2_970.84       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   2_262.74     7_536.79     9_799.53       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             4_547.97       720.08     5_268.04       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             4_547.97       655.07     5_203.04       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             4_547.97       671.85     5_219.82       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   4_547.97     6_778.40    11_326.37       1.0000          1.0000         1.91
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
Exhaustive (query)                                        21.40    11_373.54    11_394.94       1.0000          1.0000       122.07
Exhaustive (self)                                         21.40   389_492.54   389_513.94       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 2_452.24       648.71     3_100.95       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 2_452.24       696.00     3_148.23       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 2_452.24       848.28     3_300.52       1.0000          1.0000       122.20
IVF-nl500 (self)                                       2_452.24    26_373.98    28_826.22       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 4_685.12       583.46     5_268.58       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 4_685.12       706.66     5_391.78       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 4_685.12       744.39     5_429.51       1.0000          1.0000       122.25
IVF-nl707 (self)                                       4_685.12    23_231.53    27_916.65       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                8_571.39       522.52     9_093.91       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                8_571.39       687.13     9_258.51       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                8_571.39       762.74     9_334.12       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      8_571.39    21_446.76    30_018.15       1.0000          1.0000       122.32
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
Exhaustive (query)                                        22.03    10_937.80    10_959.83       1.0000          1.0000       122.07
Exhaustive (self)                                         22.03   384_990.63   385_012.66       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    37.22     2_777.27     2_814.49       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     37.22    85_809.05    85_846.27       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             2_375.53       652.15     3_027.68       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_375.53       658.65     3_034.18       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_375.53       739.03     3_114.55       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_375.53    15_476.00    17_851.53       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_448.69       743.70     5_192.39       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_448.69       720.35     5_169.04       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_448.69       678.19     5_126.88       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_448.69    13_780.21    18_228.90       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            8_119.09       789.19     8_908.29       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            8_119.09       658.45     8_777.55       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            8_119.09       725.94     8_845.03       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  8_119.09    12_448.36    20_567.45       1.0000          1.0000         3.82
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
Exhaustive (query)                                        47.82    25_567.78    25_615.60       1.0000          1.0000       244.14
Exhaustive (self)                                         47.82   856_172.43   856_220.25       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 2_398.55     1_303.43     3_701.97       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 2_398.55     1_400.49     3_799.03       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 2_398.55     1_702.08     4_100.63       1.0000          1.0000       244.39
IVF-nl500 (self)                                       2_398.55    54_550.15    56_948.70       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 4_670.10     1_176.15     5_846.25       1.0000          1.0000       244.49
IVF-nl707-np35 (query)                                 4_670.10     1_430.76     6_100.86       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 4_670.10     1_486.92     6_157.02       1.0000          1.0000       244.49
IVF-nl707 (self)                                       4_670.10    49_113.92    53_784.02       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                               10_545.94     1_050.47    11_596.41       0.9999          1.0000       244.64
IVF-nl1000-np44 (query)                               10_545.94     1_387.60    11_933.54       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                               10_545.94     1_508.04    12_053.98       1.0000          1.0000       244.64
IVF-nl1000 (self)                                     10_545.94    43_790.54    54_336.48       1.0000          1.0000       244.64
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
Exhaustive (query)                                        48.84    25_641.87    25_690.71       1.0000          1.0000       244.14
Exhaustive (self)                                         48.84   851_802.21   851_851.05       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    81.26     4_461.46     4_542.72       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     81.26   138_498.21   138_579.47       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             2_556.27       982.95     3_539.22       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             2_556.27       973.00     3_529.27       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             2_556.27     1_156.78     3_713.05       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   2_556.27    27_509.87    30_066.14       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             4_905.20     1_039.12     5_944.32       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             4_905.20     1_062.47     5_967.67       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             4_905.20     1_006.39     5_911.59       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   4_905.20    24_597.54    29_502.74       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                           10_757.12       987.09    11_744.21       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                           10_757.12     1_015.89    11_773.01       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                           10_757.12     1_069.94    11_827.05       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                 10_757.12    22_183.21    32_940.33       1.0000          1.0000         3.82
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
CPU-Exhaustive (query)                                     3.62     1_679.02     1_682.64       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.62    15_789.85    15_793.47       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     6.00       676.04       682.04       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      6.00     5_477.05     5_483.05       1.0000          1.0000        18.31
CAGRA-auto (query)                                       987.41       125.32     1_112.73       0.9332          1.0045        86.98
CAGRA-auto (self)                                        987.41       689.88     1_677.29       0.9185          1.0124        86.98
CAGRA-bw16 (query)                                       987.41       114.04     1_101.45       0.9171          1.0054        86.98
CAGRA-bw16 (self)                                        987.41       342.72     1_330.14       0.8988          1.0145        86.98
CAGRA-bw30 (query)                                       987.41       149.30     1_136.72       0.9319          1.0045        86.98
CAGRA-bw30 (self)                                        987.41       632.76     1_620.18       0.9171          1.0125        86.98
CAGRA-bw48 (query)                                       987.41       200.50     1_187.92       0.9416          1.0039        86.98
CAGRA-bw48 (self)                                        987.41     1_147.70     2_135.12       0.9283          1.0111        86.98
CAGRA-bw64 (query)                                       987.41       262.24     1_249.65       0.9473          1.0035        86.98
CAGRA-bw64 (self)                                        987.41     1_729.56     2_716.98       0.9353          1.0102        86.98
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
CPU-Exhaustive (query)                                     4.26     1_540.19     1_544.45       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      4.26    16_294.68    16_298.94       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.01       689.34       695.36       1.0000          1.0000        18.88
GPU-Exhaustive (self)                                      6.01     5_717.08     5_723.10       1.0000          1.0000        18.88
CAGRA-auto (query)                                       911.76       199.69     1_111.45       0.9319          1.0047        87.55
CAGRA-auto (self)                                        911.76       700.35     1_612.11       0.9186          1.0114        87.55
CAGRA-bw16 (query)                                       911.76       127.91     1_039.67       0.9141          1.0058        87.55
CAGRA-bw16 (self)                                        911.76       344.47     1_256.23       0.8977          1.0137        87.55
CAGRA-bw30 (query)                                       911.76       154.51     1_066.28       0.9305          1.0047        87.55
CAGRA-bw30 (self)                                        911.76       643.36     1_555.13       0.9170          1.0116        87.55
CAGRA-bw48 (query)                                       911.76       229.19     1_140.95       0.9406          1.0041        87.55
CAGRA-bw48 (self)                                        911.76     1_172.07     2_083.83       0.9293          1.0101        87.55
CAGRA-bw64 (query)                                       911.76       311.00     1_222.77       0.9472          1.0036        87.55
CAGRA-bw64 (self)                                        911.76     1_764.95     2_676.71       0.9369          1.0091        87.55
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
CPU-Exhaustive (query)                                     3.28     1_616.10     1_619.38       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.28    18_065.72    18_069.00       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.32       656.14       661.46       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.32     5_477.85     5_483.17       1.0000          1.0000        18.31
CAGRA-auto (query)                                       904.52       117.71     1_022.23       0.9804          1.0011        86.98
CAGRA-auto (self)                                        904.52       674.70     1_579.22       0.9899          1.0007        86.98
CAGRA-bw16 (query)                                       904.52       126.96     1_031.48       0.9557          1.0025        86.98
CAGRA-bw16 (self)                                        904.52       337.21     1_241.73       0.9798          1.0013        86.98
CAGRA-bw30 (query)                                       904.52       184.28     1_088.80       0.9789          1.0011        86.98
CAGRA-bw30 (self)                                        904.52       616.26     1_520.78       0.9892          1.0008        86.98
CAGRA-bw48 (query)                                       904.52       220.33     1_124.85       0.9892          1.0006        86.98
CAGRA-bw48 (self)                                        904.52     1_115.50     2_020.02       0.9941          1.0005        86.98
CAGRA-bw64 (query)                                       904.52       276.94     1_181.46       0.9932          1.0004        86.98
CAGRA-bw64 (self)                                        904.52     1_680.04     2_584.56       0.9962          1.0003        86.98
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
CPU-Exhaustive (query)                                     3.24     1_600.67     1_603.91       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.24    17_227.07    17_230.31       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     6.29       668.23       674.52       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      6.29     5_499.61     5_505.90       1.0000          1.0000        18.31
CAGRA-auto (query)                                       966.33       143.68     1_110.02       0.9866          1.0008        86.98
CAGRA-auto (self)                                        966.33       670.33     1_636.66       0.9916          1.0006        86.98
CAGRA-bw16 (query)                                       966.33        94.73     1_061.06       0.9687          1.0020        86.98
CAGRA-bw16 (self)                                        966.33       338.92     1_305.26       0.9829          1.0012        86.98
CAGRA-bw30 (query)                                       966.33       146.18     1_112.52       0.9854          1.0009        86.98
CAGRA-bw30 (self)                                        966.33       618.00     1_584.33       0.9909          1.0007        86.98
CAGRA-bw48 (query)                                       966.33       205.81     1_172.14       0.9927          1.0004        86.98
CAGRA-bw48 (self)                                        966.33     1_120.21     2_086.55       0.9953          1.0004        86.98
CAGRA-bw64 (query)                                       966.33       261.88     1_228.22       0.9955          1.0003        86.98
CAGRA-bw64 (self)                                        966.33     1_690.00     2_656.34       0.9971          1.0003        86.98
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
CPU-Exhaustive (query)                                    14.56     6_152.82     6_167.38       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     14.56    67_943.79    67_958.35       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    24.30     1_401.63     1_425.93       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     24.30    12_664.38    12_688.68       1.0000          1.0000        73.24
CAGRA-auto (query)                                     2_939.73       283.16     3_222.89       0.9861          1.0007       141.91
CAGRA-auto (self)                                      2_939.73       832.48     3_772.21       0.9916          1.0006       141.91
CAGRA-bw16 (query)                                     2_939.73       247.61     3_187.34       0.9681          1.0016       141.91
CAGRA-bw16 (self)                                      2_939.73       435.80     3_375.53       0.9829          1.0012       141.91
CAGRA-bw30 (query)                                     2_939.73       279.83     3_219.56       0.9850          1.0007       141.91
CAGRA-bw30 (self)                                      2_939.73       774.19     3_713.92       0.9910          1.0007       141.91
CAGRA-bw48 (query)                                     2_939.73       330.60     3_270.33       0.9924          1.0004       141.91
CAGRA-bw48 (self)                                      2_939.73     1_314.06     4_253.79       0.9954          1.0004       141.91
CAGRA-bw64 (query)                                     2_939.73       391.45     3_331.18       0.9952          1.0002       141.91
CAGRA-bw64 (self)                                      2_939.73     1_926.03     4_865.76       0.9971          1.0002       141.91
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
CPU-Exhaustive (query)                                    11.06     8_365.52     8_376.58       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     11.06    88_531.29    88_542.35       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    19.16     2_363.75     2_382.91       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     19.16    21_631.90    21_651.06       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_741.55       410.00     3_151.55       0.9823          1.0010       175.48
CAGRA-auto (self)                                      2_741.55     1_230.12     3_971.67       0.9887          1.0008       175.48
CAGRA-bw16 (query)                                     2_741.55       346.92     3_088.47       0.9610          1.0022       175.48
CAGRA-bw16 (self)                                      2_741.55       621.69     3_363.24       0.9783          1.0015       175.48
CAGRA-bw30 (query)                                     2_741.55       398.02     3_139.57       0.9810          1.0010       175.48
CAGRA-bw30 (self)                                      2_741.55     1_135.05     3_876.60       0.9878          1.0009       175.48
CAGRA-bw48 (query)                                     2_741.55       485.41     3_226.96       0.9899          1.0006       175.48
CAGRA-bw48 (self)                                      2_741.55     2_023.30     4_764.85       0.9935          1.0005       175.48
CAGRA-bw64 (query)                                     2_741.55       587.74     3_329.29       0.9936          1.0004       175.48
CAGRA-bw64 (self)                                      2_741.55     3_032.54     5_774.09       0.9958          1.0003       175.48
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
CPU-Exhaustive (query)                                    25.83    18_175.46    18_201.30       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     25.83   188_365.74   188_391.58       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    39.66     3_720.63     3_760.29       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     39.66    34_958.30    34_997.97       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_041.76       620.27     6_662.03       0.9811          1.0009       236.51
CAGRA-auto (self)                                      6_041.76     1_416.67     7_458.43       0.9886          1.0008       236.51
CAGRA-bw16 (query)                                     6_041.76       512.96     6_554.72       0.9595          1.0020       236.51
CAGRA-bw16 (self)                                      6_041.76       739.14     6_780.90       0.9782          1.0015       236.51
CAGRA-bw30 (query)                                     6_041.76       615.57     6_657.33       0.9796          1.0010       236.51
CAGRA-bw30 (self)                                      6_041.76     1_313.39     7_355.15       0.9878          1.0009       236.51
CAGRA-bw48 (query)                                     6_041.76       717.92     6_759.68       0.9895          1.0005       236.51
CAGRA-bw48 (self)                                      6_041.76     2_256.54     8_298.30       0.9935          1.0005       236.51
CAGRA-bw64 (query)                                     6_041.76       818.36     6_860.12       0.9932          1.0003       236.51
CAGRA-bw64 (self)                                      6_041.76     3_298.99     9_340.75       0.9958          1.0003       236.51
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
CPU-Exhaustive (query)                                    21.17    37_309.46    37_330.64       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     21.17   387_480.62   387_501.79       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    36.35     8_795.40     8_831.75       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     36.35    85_847.75    85_884.10       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_224.86       732.40     6_957.26       0.9722          1.0017       350.95
CAGRA-auto (self)                                      6_224.86     2_542.24     8_767.10       0.9811          1.0014       350.95
CAGRA-bw16 (query)                                     6_224.86       618.20     6_843.05       0.9448          1.0035       350.95
CAGRA-bw16 (self)                                      6_224.86     1_249.56     7_474.42       0.9670          1.0025       350.95
CAGRA-bw30 (query)                                     6_224.86       764.49     6_989.35       0.9702          1.0018       350.95
CAGRA-bw30 (self)                                      6_224.86     2_318.01     8_542.87       0.9800          1.0015       350.95
CAGRA-bw48 (query)                                     6_224.86       942.04     7_166.90       0.9833          1.0010       350.95
CAGRA-bw48 (self)                                      6_224.86     4_154.61    10_379.47       0.9882          1.0009       350.95
CAGRA-bw64 (query)                                     6_224.86     1_164.59     7_389.45       0.9886          1.0007       350.95
CAGRA-bw64 (self)                                      6_224.86     6_246.95    12_471.81       0.9920          1.0006       350.95
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
CPU-Exhaustive (query)                                    48.91    84_905.28    84_954.19       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     48.91   849_159.99   849_208.90       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    77.36    14_198.39    14_275.75       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     77.36   138_906.01   138_983.37       1.0000          1.0000       244.14
CAGRA-auto (query)                                    15_166.58     1_209.29    16_375.88       0.9707          1.0015       473.02
CAGRA-auto (self)                                     15_166.58     2_871.85    18_038.44       0.9810          1.0014       473.02
CAGRA-bw16 (query)                                    15_166.58     1_088.04    16_254.63       0.9428          1.0031       473.02
CAGRA-bw16 (self)                                     15_166.58     1_511.62    16_678.21       0.9667          1.0025       473.02
CAGRA-bw30 (query)                                    15_166.58     1_189.85    16_356.43       0.9687          1.0016       473.02
CAGRA-bw30 (self)                                     15_166.58     2_659.90    17_826.48       0.9799          1.0015       473.02
CAGRA-bw48 (query)                                    15_166.58     1_403.19    16_569.78       0.9820          1.0009       473.02
CAGRA-bw48 (self)                                     15_166.58     4_612.68    19_779.26       0.9880          1.0009       473.02
CAGRA-bw64 (query)                                    15_166.58     1_590.72    16_757.31       0.9878          1.0006       473.02
CAGRA-bw64 (self)                                     15_166.58     6_815.22    21_981.80       0.9919          1.0006       473.02
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
GPU-Exhaustive (ground truth)                              9.29    15_140.38    15_149.67       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   5_428.42     1_244.45     6_672.87       1.0000          1.0000       276.93
GPU-NND bk=1x refine=0 (extract)                       1_245.57        41.36     1_286.93       0.7423          1.1127       144.96
GPU-NND bk=1x refine=0 (self-beam)                     1_245.57     1_183.09     2_428.66       0.9780          1.0020       144.96
GPU-NND bk=1x refine=1 (extract)                       1_215.04        39.88     1_254.92       0.8853          1.0873       144.96
GPU-NND bk=1x refine=1 (self-beam)                     1_215.04     1_152.79     2_367.83       0.9828          1.0015       144.96
GPU-NND bk=1x refine=2 (extract)                       1_251.78        40.14     1_291.93       0.9012          1.0853       144.96
GPU-NND bk=1x refine=2 (self-beam)                     1_251.78     1_164.75     2_416.53       0.9837          1.0014       144.96
GPU-NND bk=2x refine=0 (extract)                       1_365.21        41.50     1_406.72       0.7422          1.1127       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_365.21     1_165.53     2_530.75       0.9800          1.0017       144.96
GPU-NND bk=2x refine=1 (extract)                       1_697.01        40.44     1_737.45       0.9247          1.0827       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_697.01     1_151.27     2_848.28       0.9919          1.0005       144.96
GPU-NND bk=2x refine=2 (extract)                       2_011.42        40.08     2_051.50       0.9297          1.0822       144.96
GPU-NND bk=2x refine=2 (self-beam)                     2_011.42     1_156.02     3_167.44       0.9927          1.0004       144.96
GPU-NND bk=3x refine=0 (extract)                       1_759.88        41.05     1_800.92       0.7399          1.1132       144.96
GPU-NND bk=3x refine=0 (self-beam)                     1_759.88     1_168.00     2_927.88       0.9799          1.0017       144.96
GPU-NND bk=3x refine=1 (extract)                       2_569.35        40.51     2_609.86       0.9305          1.0821       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_569.35     1_162.34     3_731.69       0.9933          1.0004       144.96
GPU-NND bk=3x refine=2 (extract)                       3_286.66        40.60     3_327.27       0.9326          1.0819       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_286.66     1_170.39     4_457.05       0.9937          1.0003       144.96
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
GPU-Exhaustive (ground truth)                             21.34    21_731.71    21_753.05       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   6_701.14     1_803.77     8_504.91       1.0000          1.0000       365.97
GPU-NND bk=1x refine=0 (extract)                       1_595.58        41.45     1_637.02       0.7398          1.1129       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_595.58     1_308.97     2_904.55       0.9777          1.0020       175.48
GPU-NND bk=1x refine=1 (extract)                       1_998.35        40.54     2_038.89       0.8848          1.0873       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_998.35     1_241.36     3_239.71       0.9827          1.0015       175.48
GPU-NND bk=1x refine=2 (extract)                       2_336.55        40.36     2_376.91       0.9009          1.0853       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_336.55     1_239.20     3_575.75       0.9836          1.0014       175.48
GPU-NND bk=2x refine=0 (extract)                       1_819.13        39.99     1_859.12       0.7398          1.1129       175.48
GPU-NND bk=2x refine=0 (self-beam)                     1_819.13     1_252.21     3_071.34       0.9797          1.0017       175.48
GPU-NND bk=2x refine=1 (extract)                       3_315.74        39.77     3_355.52       0.9245          1.0826       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_315.74     1_243.84     4_559.58       0.9918          1.0005       175.48
GPU-NND bk=2x refine=2 (extract)                       4_702.01        40.64     4_742.65       0.9297          1.0821       175.48
GPU-NND bk=2x refine=2 (self-beam)                     4_702.01     1_246.56     5_948.57       0.9927          1.0004       175.48
GPU-NND bk=3x refine=0 (extract)                       2_267.16        40.08     2_307.24       0.7380          1.1133       175.48
GPU-NND bk=3x refine=0 (self-beam)                     2_267.16     1_266.76     3_533.92       0.9796          1.0018       175.48
GPU-NND bk=3x refine=1 (extract)                       5_127.47        40.78     5_168.25       0.9304          1.0820       175.48
GPU-NND bk=3x refine=1 (self-beam)                     5_127.47     1_249.80     6_377.26       0.9933          1.0004       175.48
GPU-NND bk=3x refine=2 (extract)                       7_629.55        40.95     7_670.49       0.9326          1.0818       175.48
GPU-NND bk=3x refine=2 (self-beam)                     7_629.55     1_238.29     8_867.84       0.9937          1.0003       175.48
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
GPU-Exhaustive (ground truth)                             20.21    59_763.59    59_783.80       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  12_362.38     3_034.65    15_397.02       0.9999          1.0000       631.89
GPU-NND bk=1x refine=0 (extract)                       2_002.73        83.70     2_086.43       0.6794          1.1260       289.92
GPU-NND bk=1x refine=0 (self-beam)                     2_002.73     2_415.08     4_417.81       0.9617          1.0037       289.92
GPU-NND bk=1x refine=1 (extract)                       2_223.89        81.30     2_305.18       0.8547          1.0912       289.92
GPU-NND bk=1x refine=1 (self-beam)                     2_223.89     2_334.08     4_557.97       0.9712          1.0026       289.92
GPU-NND bk=1x refine=2 (extract)                       2_515.69        83.26     2_598.95       0.8821          1.0874       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_515.69     2_338.48     4_854.17       0.9731          1.0023       289.92
GPU-NND bk=2x refine=0 (extract)                       2_387.15        80.82     2_467.97       0.6792          1.1260       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_387.15     2_380.35     4_767.50       0.9644          1.0033       289.92
GPU-NND bk=2x refine=1 (extract)                       3_569.56        81.70     3_651.26       0.9151          1.0835       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_569.56     2_335.14     5_904.70       0.9869          1.0008       289.92
GPU-NND bk=2x refine=2 (extract)                       4_644.71        81.08     4_725.78       0.9267          1.0822       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_644.71     2_341.39     6_986.09       0.9887          1.0006       289.92
GPU-NND bk=3x refine=0 (extract)                       3_168.29        82.01     3_250.30       0.6767          1.1266       289.92
GPU-NND bk=3x refine=0 (self-beam)                     3_168.29     2_385.13     5_553.41       0.9643          1.0033       289.92
GPU-NND bk=3x refine=1 (extract)                       5_574.56        81.37     5_655.93       0.9250          1.0824       289.92
GPU-NND bk=3x refine=1 (self-beam)                     5_574.56     2_336.83     7_911.40       0.9896          1.0006       289.92
GPU-NND bk=3x refine=2 (extract)                       7_602.03        81.23     7_683.26       0.9318          1.0817       289.92
GPU-NND bk=3x refine=2 (self-beam)                     7_602.03     2_339.61     9_941.64       0.9907          1.0005       289.92
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
GPU-Exhaustive (ground truth)                             38.34    85_928.15    85_966.49       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  15_504.85     4_294.61    19_799.46       0.9999          1.0000       803.96
GPU-NND bk=1x refine=0 (extract)                       2_958.23        83.41     3_041.64       0.6746          1.1267       350.95
GPU-NND bk=1x refine=0 (self-beam)                     2_958.23     2_575.96     5_534.19       0.9610          1.0037       350.95
GPU-NND bk=1x refine=1 (extract)                       4_328.08        81.19     4_409.27       0.8533          1.0912       350.95
GPU-NND bk=1x refine=1 (self-beam)                     4_328.08     2_524.52     6_852.60       0.9709          1.0026       350.95
GPU-NND bk=1x refine=2 (extract)                       5_620.84        83.76     5_704.60       0.8815          1.0873       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_620.84     2_508.84     8_129.68       0.9728          1.0023       350.95
GPU-NND bk=2x refine=0 (extract)                       3_401.43        82.53     3_483.96       0.6745          1.1267       350.95
GPU-NND bk=2x refine=0 (self-beam)                     3_401.43     2_548.43     5_949.85       0.9638          1.0033       350.95
GPU-NND bk=2x refine=1 (extract)                       8_289.10        81.02     8_370.13       0.9149          1.0834       350.95
GPU-NND bk=2x refine=1 (self-beam)                     8_289.10     2_500.23    10_789.34       0.9868          1.0008       350.95
GPU-NND bk=2x refine=2 (extract)                      12_839.39        81.07    12_920.46       0.9267          1.0821       350.95
GPU-NND bk=2x refine=2 (self-beam)                    12_839.39     2_509.70    15_349.09       0.9886          1.0006       350.95
GPU-NND bk=3x refine=0 (extract)                       4_319.59        83.55     4_403.13       0.6722          1.1273       350.95
GPU-NND bk=3x refine=0 (self-beam)                     4_319.59     2_562.01     6_881.60       0.9636          1.0034       350.95
GPU-NND bk=3x refine=1 (extract)                      13_372.17        80.19    13_452.36       0.9246          1.0824       350.95
GPU-NND bk=3x refine=1 (self-beam)                    13_372.17     2_518.37    15_890.54       0.9895          1.0006       350.95
GPU-NND bk=3x refine=2 (extract)                      21_154.01        81.32    21_235.32       0.9318          1.0816       350.95
GPU-NND bk=3x refine=2 (self-beam)                    21_154.01     2_515.94    23_669.94       0.9906          1.0005       350.95
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
GPU-Exhaustive (ground truth)                             36.83   237_409.97   237_446.80       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  28_200.35     7_317.35    35_517.70       0.9999          1.0000      1295.77
GPU-NND bk=1x refine=0 (extract)                       3_642.51       166.56     3_809.07       0.6161          1.1417       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_642.51     4_901.51     8_544.02       0.9400          1.0060       579.83
GPU-NND bk=1x refine=1 (extract)                       4_674.63       166.66     4_841.29       0.8166          1.0966       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_674.63     4_754.79     9_429.42       0.9562          1.0041       579.83
GPU-NND bk=1x refine=2 (extract)                       5_680.19       162.64     5_842.83       0.8582          1.0903       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_680.19     4_749.49    10_429.68       0.9598          1.0036       579.83
GPU-NND bk=2x refine=0 (extract)                       4_652.02       163.45     4_815.47       0.6160          1.1417       579.83
GPU-NND bk=2x refine=0 (self-beam)                     4_652.02     4_873.42     9_525.44       0.9433          1.0056       579.83
GPU-NND bk=2x refine=1 (extract)                       8_277.30       162.78     8_440.08       0.8995          1.0850       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_277.30     4_752.46    13_029.76       0.9802          1.0013       579.83
GPU-NND bk=2x refine=2 (extract)                      11_592.71       163.45    11_756.16       0.9222          1.0825       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_592.71     4_750.87    16_343.58       0.9837          1.0010       579.83
GPU-NND bk=3x refine=0 (extract)                       6_238.90       163.44     6_402.34       0.6138          1.1423       579.83
GPU-NND bk=3x refine=0 (self-beam)                     6_238.90     4_871.64    11_110.54       0.9430          1.0057       579.83
GPU-NND bk=3x refine=1 (extract)                      13_296.88       163.38    13_460.26       0.9134          1.0835       579.83
GPU-NND bk=3x refine=1 (self-beam)                    13_296.88     4_752.16    18_049.04       0.9845          1.0009       579.83
GPU-NND bk=3x refine=2 (extract)                      19_503.47       162.47    19_665.93       0.9304          1.0817       579.83
GPU-NND bk=3x refine=2 (self-beam)                    19_503.47     4_771.64    24_275.11       0.9873          1.0006       579.83
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
GPU-Exhaustive (ground truth)                             75.33   341_591.63   341_666.96       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  34_090.33    10_394.83    44_485.16       0.9999          1.0000      1487.90
GPU-NND bk=1x refine=0 (extract)                       5_383.11       166.65     5_549.76       0.6166          1.1408       701.90
GPU-NND bk=1x refine=0 (self-beam)                     5_383.11     5_262.03    10_645.14       0.9402          1.0060       701.90
GPU-NND bk=1x refine=1 (extract)                       9_264.88       164.51     9_429.39       0.8169          1.0963       701.90
GPU-NND bk=1x refine=1 (self-beam)                     9_264.88     5_144.94    14_409.81       0.9563          1.0040       701.90
GPU-NND bk=1x refine=2 (extract)                      12_946.27       164.41    13_110.68       0.8583          1.0901       701.90
GPU-NND bk=1x refine=2 (self-beam)                    12_946.27     5_117.74    18_064.01       0.9599          1.0036       701.90
GPU-NND bk=2x refine=0 (extract)                       6_634.13       163.83     6_797.96       0.6165          1.1409       701.90
GPU-NND bk=2x refine=0 (self-beam)                     6_634.13     5_262.38    11_896.51       0.9435          1.0056       701.90
GPU-NND bk=2x refine=1 (extract)                      19_512.92       162.03    19_674.95       0.8998          1.0849       701.90
GPU-NND bk=2x refine=1 (self-beam)                    19_512.92     5_161.30    24_674.22       0.9803          1.0013       701.90
GPU-NND bk=2x refine=2 (extract)                      31_328.30       166.99    31_495.30       0.9223          1.0824       701.90
GPU-NND bk=2x refine=2 (self-beam)                    31_328.30     5_116.21    36_444.51       0.9837          1.0010       701.90
GPU-NND bk=3x refine=0 (extract)                       8_338.83       177.07     8_515.90       0.6140          1.1415       701.90
GPU-NND bk=3x refine=0 (self-beam)                     8_338.83     5_279.02    13_617.85       0.9431          1.0056       701.90
GPU-NND bk=3x refine=1 (extract)                      31_697.74       163.89    31_861.63       0.9136          1.0834       701.90
GPU-NND bk=3x refine=1 (self-beam)                    31_697.74     5_184.04    36_881.77       0.9845          1.0009       701.90
GPU-NND bk=3x refine=2 (extract)                      52_460.58       163.10    52_623.68       0.9305          1.0816       701.90
GPU-NND bk=3x refine=2 (self-beam)                    52_460.58     5_130.89    57_591.47       0.9872          1.0006       701.90
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
GPU-Exhaustive (ground truth)                             91.77 1_485_312.68 1_485_404.45       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  73_796.55    22_392.67    96_189.22       0.9997          1.0000      3487.41
GPU-NND bk=1x refine=0 (extract)                       7_520.71       419.61     7_940.32       0.4813          1.1868      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_520.71    12_590.62    20_111.33       0.8861          1.0129      1449.59
GPU-NND bk=1x refine=1 (extract)                      11_129.75       407.71    11_537.45       0.7243          1.1131      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    11_129.75    12_132.72    23_262.47       0.9224          1.0080      1449.59
GPU-NND bk=1x refine=2 (extract)                      14_521.09       415.09    14_936.18       0.8021          1.0984      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    14_521.09    12_099.86    26_620.95       0.9319          1.0067      1449.59
GPU-NND bk=2x refine=0 (extract)                       9_432.15       420.41     9_852.56       0.4813          1.1868      1449.59
GPU-NND bk=2x refine=0 (self-beam)                     9_432.15    12_571.98    22_004.13       0.8896          1.0124      1449.59
GPU-NND bk=2x refine=1 (extract)                      22_543.37       412.14    22_955.51       0.8496          1.0914      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    22_543.37    12_078.42    34_621.79       0.9660          1.0027      1449.59
GPU-NND bk=2x refine=2 (extract)                      34_005.83       410.42    34_416.26       0.9102          1.0836      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    34_005.83    12_055.41    46_061.24       0.9750          1.0016      1449.59
GPU-NND bk=3x refine=0 (extract)                      13_534.58       408.48    13_943.05       0.4813          1.1868      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    13_534.58    12_584.26    26_118.83       0.8897          1.0124      1449.59
GPU-NND bk=3x refine=1 (extract)                      37_338.51       408.84    37_747.35       0.8497          1.0919      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    37_338.51    12_116.83    49_455.34       0.9698          1.0023      1449.59
GPU-NND bk=3x refine=2 (extract)                      59_296.29       407.93    59_704.23       0.9249          1.0820      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    59_296.29    12_101.38    71_397.67       0.9813          1.0010      1449.59
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
